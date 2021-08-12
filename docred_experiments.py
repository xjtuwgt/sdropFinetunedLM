from argparse import ArgumentParser
from collections import namedtuple
from eval.docred_evaluation import official_evaluate
from data.docred import DocREDDataset, docred_validate_fn
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModel

from data.docred import DocREDDataset, docred_collate_fn, docred_validate_fn
from data.dataset import SentenceDropDataset
from eval.docred_evaluation import official_evaluate

DocREDModelOutput = namedtuple("DocREDModelOutput", ["loss", "pair_logits", "pair_sent_score"])

class TransformerModelForDocRED(nn.Module):
    def __init__(self, model_type, rel_vocab_size, ner_vocab_size, sentence_emb_size=32, lambda_evi=1):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.model.config.hidden_size
        self.lambda_evi = lambda_evi

        self.ner_embedding = nn.Embedding(ner_vocab_size, self.hidden_size, padding_idx=0)
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size * 2, rel_vocab_size)
        )

        self.sentence_embedder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size * 2, sentence_emb_size)
        )
        self.pair_embedder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size * 2, sentence_emb_size)
        )

        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, batch):
        inputs_embeds = self.model.get_input_embeddings()(batch['context'])
        inputs_embeds += self.ner_embedding(batch['ner'])
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'])
        hid = output.last_hidden_state

        # relation prediction
        ent_mask = batch['entity_mask'].float()
        ent_hid = ent_mask.bmm(hid) / (ent_mask.sum(2, keepdim=True) + 1e-6)
        pairs = batch['entity_pairs'].clamp(min=0)
        head_hid = ent_hid.gather(1, pairs[:, :, :1].repeat(1, 1, self.hidden_size))
        tail_hid = ent_hid.gather(1, pairs[:, :, 1:].repeat(1, 1, self.hidden_size))

        pair_logits = self.rel_classifier(torch.cat([head_hid, tail_hid], 2))
        loss0 = self.crit(pair_logits, batch['pair_labels'].float())
        loss = loss0.masked_select(pairs[:, :, :1] >= 0).mean()

        # evidence prediction
        sent_start = batch['sentence_start'].clamp(min=0)
        sent_start_hid = hid.gather(1, sent_start.unsqueeze(-1).repeat(1, 1, self.hidden_size))
        sent_end = batch['sentence_end'].clamp(min=0)
        sent_end_hid = hid.gather(1, sent_end.unsqueeze(-1).repeat(1, 1, self.hidden_size))
        sent_emb = self.sentence_embedder(torch.cat([sent_start_hid, sent_end_hid], 2))

        pair_emb = self.pair_embedder(torch.cat([head_hid, tail_hid], 2))        

        pair_sent_score0 = pair_emb.bmm(sent_emb.transpose(1, 2))
        pair_sent_score = pair_sent_score0.unsqueeze(2).masked_select((batch['pair_labels'] > 0).unsqueeze(-1))
        pair_sent_label = batch['pair_evidence'].masked_select((batch['pair_labels'] > 0).unsqueeze(-1))

        loss += self.lambda_evi * self.crit(pair_sent_score, pair_sent_label.float()).mean()

        return DocREDModelOutput(loss=loss, pair_logits=pair_logits, pair_sent_score=pair_sent_score0)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--span_dropout', type=float, default=.1)
    parser.add_argument('--train_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--report_every', type=int, default=10)

    parser.add_argument('--train_seqlen', type=int, default=300)
    parser.add_argument('--test_seqlen', type=int, default=300)
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--beta_drop', action='store_true')
    parser.add_argument('--beta_drop_scale', default=1, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)

    parser.add_argument('--model_type', type=str, default="google/electra-base-discriminator")

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DocREDDataset("dataset/docred/train_annotated.json", tokenizer_class=args.model_type)
    dev_dataset = DocREDDataset("dataset/docred/dev.json", tokenizer_class=args.model_type, ner_to_idx=dataset.ner_to_idx, relation_to_idx=dataset.relation_to_idx, eval=True)

    model = TransformerModelForDocRED(args.model_type, len(dataset.relation_to_idx), len(dataset.ner_to_idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    model.cuda()

    validation_fn = docred_validate_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.span_dropout,
        example_validate_fn=validation_fn, beta_drop=args.beta_drop, beta_drop_scale=args.beta_drop_scale)

    if args.span_dropout > 0:
        failed, total = sdrop_dataset.estimate_label_noise(reps=1, validation_fn=docred_validate_fn)
        print(f"Label noise: {failed / total * 100 :6.3f}% ({failed:7d} / {total:7d})", flush=True)
    else:
        print(f"Label noise: 0% (- / -)", flush=True)

    collate_fn = lambda examples: docred_collate_fn(examples, dataset=dataset)
    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    step = 0
    best_dev_score = -1
    best_dev_result = None
    best_step = -1

    while True:
        model.train()
        for batch in dataloader:
            batch = {k: batch[k].cuda() if k != 'doc_title' else batch[k] for k in batch}
            step += 1
            output = model(batch)

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
            if step % args.report_every == 0:
                print(f"Step {step}: loss={output.loss.item():6f}")

            if step % args.eval_every == 0:
                model.eval()
                predictions = []
                print("Evaluating on the dev set...")
                with torch.no_grad():
                    for batch in tqdm(dev_dataloader):
                        pair_ids = batch['entity_pairs'].numpy()
                        sent_start = batch['sentence_start'].numpy()
                        batch = {k: batch[k].cuda() if k != 'doc_title' else batch[k] for k in batch}
                        output = model(batch)

                        # decode output
                        pair_pred = (output.pair_logits > 0).detach().cpu().numpy()
                        pair_sent_pred = (output.pair_sent_score > 0).detach().cpu().numpy()
                        for ex_i in range(pair_pred.shape[0]):
                            for pair_i in range(pair_pred.shape[1]):
                                if pair_ids[ex_i, pair_i, 0] <= 0: continue
                                head, tail = pair_ids[ex_i, pair_i]
                                for r in range(pair_pred.shape[2]):
                                    if not pair_pred[ex_i, pair_i, r]: continue
                                    evidence = []
                                    for sent_i in range(sent_start.shape[1]):
                                        if sent_start[ex_i, sent_i] <= 0:
                                            break
                                        if pair_sent_pred[ex_i, pair_i, sent_i]:
                                            evidence.append(sent_i)
                                    pred = {
                                        "title": batch["doc_title"][ex_i],
                                        "h_idx": head,
                                        "t_idx": tail,
                                        "r": dataset.idx_to_relation[r],
                                        "evidence": evidence
                                    }
                                    predictions.append(pred)

                    dev_result = official_evaluate(predictions, 'dataset/docred')

                    re_f1 = dev_result['re_f1']
                    evi_f1 = dev_result['evi_f1']
                    print(dev_result)
                    dev_score = 2 * re_f1 * evi_f1 / (re_f1 + evi_f1 + 1e-6)
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        best_dev_result = dev_result
                        best_step = step

                model.train()

            if step >= args.steps:
                break

        if step >= args.steps:
            break

    print(f"Best dev result: dev scores={best_dev_result} at step {best_step}")
