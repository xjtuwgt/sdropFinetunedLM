from argparse import ArgumentParser
from collections import namedtuple
from eval.docred_evaluation import official_evaluate
from data.docred import DocREDDataset, docred_validate_fn
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import AutoModel

from data.docred import DocREDDataset, docred_collate_fn, docred_validate_fn, docred_sent_drop_postproc
from data.dataset import SentenceDropDataset
from eval.docred_evaluation import official_evaluate

DocREDModelOutput = namedtuple("DocREDModelOutput", ["loss", "pair_logits", "pair_sent_score"])

class TransformerModelForDocRED(nn.Module):
    def __init__(self, model_type, rel_vocab_size, ner_vocab_size, ent_emb_size=32, sentence_emb_size=32, lambda_evi=1):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.model.config.hidden_size
        self.lambda_evi = lambda_evi

        self.ner_embedding = nn.Embedding(ner_vocab_size, self.hidden_size, padding_idx=0)
        # self.rel_classifier = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Dropout(.1),
        #     nn.Linear(self.hidden_size * 2, rel_vocab_size)
        # )
        self.rel_hid_h = nn.Linear(self.hidden_size, ent_emb_size)
        self.rel_hid_t = nn.Linear(self.hidden_size, ent_emb_size)
        self.rel_biaffine = nn.Parameter(torch.empty(ent_emb_size, ent_emb_size, rel_vocab_size + 1))
        nn.init.normal_(self.rel_biaffine.data, 0.03)

        self.sentence_embedder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size * 2, sentence_emb_size),
            nn.Tanh()
        )
        self.pair_embedder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size * 2, sentence_emb_size),
            nn.Tanh()
        )

        self.evi_biaffine = nn.Parameter(torch.empty(sentence_emb_size, sentence_emb_size, rel_vocab_size))
        nn.init.normal_(self.evi_biaffine.data, 0.03)

        self.crit = nn.BCEWithLogitsLoss()

    def masked_atlop_loss(self, input, target, mask):
        if not mask.any():
            return 0

        # using 0 as the threshold value for simplicity, and assuming that the last dimension is what the softmax is over
        input_size = list(input.size())
        mask_size = list(mask.size())

        threshold = input.new_zeros(input_size[:-1] + [1])
        new_mask = torch.cat([mask.new_ones(mask_size[:-1] + [1]), mask.expand_as(input)], -1)

        new_input = torch.cat([threshold, input], -1).masked_fill(new_mask.logical_not(), -1e10)

        pos_label = torch.cat([target.new_zeros(input_size[:-1] + [1]), target], -1)
        neg_label = torch.cat([target.new_ones(input_size[:-1] + [1]), target.new_zeros(target.size())], -1)

        pos_loss = -torch.log_softmax(new_input - (1 - pos_label - neg_label) * 1e10, -1) * pos_label
        neg_loss = -torch.log_softmax(new_input - pos_label * 1e10, -1) * neg_label

        return (pos_loss + neg_loss).sum(-1).masked_select(new_mask.any(-1)).mean()

    def masked_bce_loss(self, input, target, mask):
        if not mask.any():
            return 0
        return self.crit(input.masked_select(mask), target.masked_select(mask))

    def forward(self, batch):
        MAXLEN = 512
        inputs_embeds = self.model.get_input_embeddings()(batch['context'])
        inputs_embeds += self.ner_embedding(batch['ner'])
        inputs_embeds = inputs_embeds[:, :MAXLEN]
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'][:, :MAXLEN])
        hid = output.last_hidden_state

        # relation prediction
        ent_mask = batch['entity_mask'][:, :, :MAXLEN].float()
        # # average pool
        avg_mask = ent_mask / (ent_mask.sum(2, keepdim=True) + 1e-6)
        ent_hid = avg_mask.bmm(hid)
        # attention pool
        # attn_w = hid[:, 0].unsqueeze(1).bmm(hid.transpose(1, 2))
        # attn = torch.softmax(attn_w - (1 - ent_mask.float()) * 1e10, 2)
        # ent_hid = attn.bmm(hid)

        pairs0 = batch['entity_pairs']
        pairs = pairs0.clamp(min=0)
        head_hid = ent_hid.gather(1, pairs[:, :, :1].repeat(1, 1, self.hidden_size))
        tail_hid = ent_hid.gather(1, pairs[:, :, 1:].repeat(1, 1, self.hidden_size))
        head_emb = torch.tanh(self.rel_hid_h(head_hid))
        tail_emb = torch.tanh(self.rel_hid_t(tail_hid))

        pair_logits0 = torch.einsum('bpd,bpe,der->bpr', head_emb, tail_emb, self.rel_biaffine)
        # pair_logits0 = self.rel_classifier(torch.cat([head_hid, tail_hid], 2))
        pair_logits0 = pair_logits0[:, :, 1:] - pair_logits0[:, :, :1]

        loss = self.masked_atlop_loss(pair_logits0, batch['pair_labels'].float(), pairs0[:, :, :1] >= 0)
        # loss = self.masked_bce_loss(pair_logits0, batch['pair_labels'].float(), pairs0[:, :, :1] >= 0)

        # evidence prediction
        sent_start = batch['sentence_start'].clamp(min=0, max=MAXLEN-1)
        sent_start = torch.cat([sent_start.new_zeros([sent_start.size(0), 1]), sent_start], 1)
        sent_start_hid = hid.gather(1, sent_start.unsqueeze(-1).repeat(1, 1, self.hidden_size))
        sent_end = batch['sentence_end'].clamp(min=0, max=MAXLEN-1)
        sent_end = torch.cat([sent_end.new_zeros([sent_end.size(0), 1]), sent_end], 1)
        sent_end_hid = hid.gather(1, sent_end.unsqueeze(-1).repeat(1, 1, self.hidden_size))
        sent_emb = self.sentence_embedder(torch.cat([sent_start_hid, sent_end_hid], 2))

        pair_emb = self.pair_embedder(torch.cat([head_hid, tail_hid], 2))

        pair_sent_score0 = torch.einsum('bpd,bse,der->bprs', pair_emb, sent_emb, self.evi_biaffine)
        pair_sent_score0 = pair_sent_score0[:, :, :, 1:] - pair_sent_score0[:, :, :, :1]

        loss += self.lambda_evi * self.masked_atlop_loss(pair_sent_score0, batch['pair_evidence'].float(), ((batch['pair_labels'].unsqueeze(-1) > 0) & (batch['sentence_start'].unsqueeze(1).unsqueeze(1) >= 0)))
        # loss += self.lambda_evi * self.masked_bce_loss(pair_sent_score0, batch['pair_evidence'].float(), ((batch['pair_labels'].unsqueeze(-1) > 0) & (batch['sentence_start'].unsqueeze(1).unsqueeze(1) >= 0)))

        return DocREDModelOutput(loss=loss, pair_logits=pair_logits0, pair_sent_score=pair_sent_score0)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--span_dropout', type=float, default=.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--report_every', type=int, default=20)
    parser.add_argument('--lr_warmup_portion', type=float, default=0.01)

    parser.add_argument('--train_seqlen', type=int, default=300)
    parser.add_argument('--test_seqlen', type=int, default=300)
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--beta_drop', action='store_true')
    parser.add_argument('--beta_drop_scale', default=1, type=float)
    parser.add_argument('--lr', default=3e-5, type=float)

    parser.add_argument('--model_type', type=str, default="google/electra-base-discriminator")

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DocREDDataset("dataset/docred/train_annotated.json", tokenizer_class=args.model_type)
    dev_dataset = DocREDDataset("dataset/docred/dev.json", tokenizer_class=args.model_type,
        ner_to_idx=dataset.ner_to_idx, relation_to_idx=dataset.relation_to_idx, eval=True)

    model = TransformerModelForDocRED(args.model_type, len(dataset.relation_to_idx), len(dataset.ner_to_idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(args.lr_warmup_portion * args.steps), args.steps)

    model.cuda()

    validation_fn = docred_validate_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.span_dropout,
        sent_drop_postproc=docred_sent_drop_postproc,
        example_validate_fn=validation_fn, beta_drop=args.beta_drop, beta_drop_scale=args.beta_drop_scale)

    if args.span_dropout > 0:
        failed, total = sdrop_dataset.estimate_label_noise(reps=1, validation_fn=docred_validate_fn)
        print(f"Label noise: {failed / total * 100 :6.3f}% ({failed:7d} / {total:7d})", flush=True)
    else:
        print(f"Label noise:   0.000% (      0 / {len(sdrop_dataset):7d})", flush=True)

    collate_fn = lambda examples: docred_collate_fn(examples, dataset=dataset)
    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    step = 0
    best_dev_score = -1
    best_dev_result = None
    best_step = -1

    while True:
        model.train()
        for batch in dataloader:
            batch = {k: batch[k].cuda() if isinstance(batch[k], torch.Tensor) else batch[k] for k in batch}
            step += 1
            output = model(batch)

            gnorm = 0
            if isinstance(output.loss, torch.Tensor):
                optimizer.zero_grad()
                output.loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()

            if step % args.report_every == 0:
                print(f"Step {step:6d}: loss={output.loss.item():6f}, lr={optimizer.param_groups[0]['lr']:3e}, gradnorm={gnorm:6f}", flush=True)

            if step % args.eval_every == 0:
                model.eval()
                predictions = []
                print("Evaluating on the dev set...", flush=True)
                with torch.no_grad():
                    for batch in tqdm(dev_dataloader):
                        pair_ids = batch['entity_pairs'].numpy()
                        sent_start = batch['sentence_start'].numpy()
                        batch = {k: batch[k].cuda() if isinstance(batch[k], torch.Tensor) else batch[k] for k in batch}
                        output = model(batch)

                        # decode output
                        pair_pred = (output.pair_logits > 0).detach().cpu().numpy()
                        pair_sent_pred = (output.pair_sent_score > 0).detach().cpu().numpy()

                        for ex_i, pair_i, r in zip(*np.where(pair_pred)):
                            head, tail = pair_ids[ex_i, pair_i]
                            if head < 0: continue
                            evidence = [sent_i for sent_i in np.where(pair_sent_pred[ex_i, pair_i, r])[0] if sent_start[ex_i, sent_i] >= 0]

                            pred = {
                                "title": batch["doc_title"][ex_i],
                                "h_idx": head,
                                "t_idx": tail,
                                "r": dataset.idx_to_relation[r],
                                "evidence": evidence
                            }
                            predictions.append(pred)

                print(f"{len(predictions)} predictions made", flush=True)
                dev_result = official_evaluate(predictions, 'dataset/docred')

                re_f1 = dev_result['re_f1']
                evi_f1 = dev_result['evi_f1']
                dev_score = 2 * re_f1 * evi_f1 / (re_f1 + evi_f1 + 1e-6)
                if dev_score > best_dev_score:
                    print("New best dev performance: ", dev_result, flush=True)
                    best_dev_score = dev_score
                    best_dev_result = dev_result
                    best_step = step
                else:
                    print(dev_result, flush=True)

                model.train()

            if step >= args.steps:
                break

        if step >= args.steps:
            break

    print(f"Best dev result: dev scores={best_dev_result} at step {best_step}")
