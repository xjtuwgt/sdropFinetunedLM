from argparse import ArgumentParser
from collections import namedtuple
from data.docred import DocREDDataset, docred_validate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModel

from data.docred import DocREDDataset, docred_collate_fn, docred_validate_fn
from data.dataset import SentenceDropDataset

DocREDModelOutput = namedtuple("DocREDModelOutput", ["loss"])

class TransformerModelForDocRED(nn.Module):
    def __init__(self, model_type, rel_vocab_size, ner_vocab_size):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.model.config.hidden_size

        self.ner_embedding = nn.Embedding(ner_vocab_size, self.hidden_size, padding_idx=0)
        self.rel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(self.hidden_size * 2, rel_vocab_size)
        )

        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, batch):
        print({k: batch[k].size() for k in batch})
        inputs_embeds = self.model.get_input_embeddings()(batch['context'])
        inputs_embeds += self.ner_embedding(batch['ner'])
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'])
        hid = output.last_hidden_state
        ent_mask = batch['entity_mask'].float()
        ent_hid = ent_mask.bmm(hid) / (ent_mask.sum(2, keepdim=True) + 1e-6)
        pairs = batch['entity_pairs'].clamp(min=0)
        head_hid = ent_hid.gather(1, pairs[:, :, :1].repeat(1, 1, self.hidden_size))
        tail_hid = ent_hid.gather(1, pairs[:, :, 1:].repeat(1, 1, self.hidden_size))

        pair_logits = self.rel_classifier(torch.cat([head_hid, tail_hid], 2))
        loss0 = self.crit(pair_logits, batch['pair_labels'].float())
        loss = loss0.masked_select(pairs[:, :, :1] >= 0).mean()
        
        return DocREDModelOutput(loss=loss)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--span_dropout', type=float, default=.1)
    parser.add_argument('--train_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--report_every', type=int, default=1)

    parser.add_argument('--train_seqlen', type=int, default=300)
    parser.add_argument('--test_seqlen', type=int, default=300)
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--beta_drop', action='store_true')
    parser.add_argument('--beta_drop_scale', default=1, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)

    parser.add_argument('--model_type', type=str, default="google/electra-base-discriminator")

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.cuda.manual_seed_all(args.seed)

    dataset = DocREDDataset("dataset/docred/train_annotated.json", tokenizer_class=args.model_type)
    dev_dataset = DocREDDataset("dataset/docred/dev.json", tokenizer_class=args.model_type, ner_to_idx=dataset.ner_to_idx, relation_to_idx=dataset.relation_to_idx)

    model = TransformerModelForDocRED(args.model_type, len(dataset.relation_to_idx), len(dataset.ner_to_idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # model.cuda()

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
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    step = 0
    best_dev_acc = -1
    best_step = -1

    total = 0
    correct = 0
    total_loss = 0

    while True:
        model.train()
        for batch in dataloader:
            # batch = {k: batch[k].cuda() for k in batch}
            step += 1
            output = model(batch)

            total_loss += output.loss.item()

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
            if step % args.report_every == 0:
                print(f"Step {step}: loss={output.loss.item():6f}")

            if step % args.eval_every == 0:
                print(f"Epoch {step}: train accuracy={correct / total:.6f}, train loss={total_loss / total}", flush=True)

                model.eval()
                total = 0
                correct = 0
                with torch.no_grad():
                    for batch in dev_dataloader:
                        batch = {k: batch[k].cuda() for k in batch}
                        input = batch['input'].clamp(min=0)
                        attn_mask = (input >= 0)
                        output = model(input, attention_mask=attn_mask, labels=batch['labels'])
                        pred = output.logits.max(1)[1]

                        total += len(pred)
                        correct += (pred == batch['labels']).sum()

                print(f"Step {step}: dev accuracy={correct / total:.6f}", flush=True)

                if correct / total > best_dev_acc:
                    best_dev_acc = correct / total
                    best_step = step

                total = 0
                correct = 0
                total_loss = 0

                model.train()

            if step >= args.steps:
                break

        if step >= args.steps:
            break

    print(f"Best dev result: dev accuracy={best_dev_acc:.6f} at step {best_step}")
