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

class TransformerModelForDocRED(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.model.config.hidden_size

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--sent_dropout', type=float, default=.1)
    parser.add_argument('--train_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=100)

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

    model = TransformerModelForDocRED(args.model_type)
    optimizer = nn.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    model.cuda()

    dataset = DocREDDataset("/Users/peng.qi/Downloads/train_annotated.json", tokenizer_class=args.model_type)
    dev_dataset = DocREDDataset("/Users/peng.qi/Downloads/dev.json", tokenizer_class=args.model_type)

    validation_fn = docred_validate_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.sent_dropout,
        example_validate_fn=validation_fn, beta_drop=args.beta_drop, beta_drop_scale=args.beta_drop_scale)

    if args.sent_dropout > 0:
        failed, total = sdrop_dataset.estimate_label_noise(reps=10, validation_fn=docred_validate_fn)
        print(f"Label noise: {failed / total * 100 :6.3f}% ({failed:7d} / {total:7d})", flush=True)
    else:
        print(f"Label noise: 0% (- / -)", flush=True)

    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=docred_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=docred_collate_fn)

    step = 0
    best_dev_acc = -1
    best_step = -1

    total = 0
    correct = 0
    total_loss = 0

    while True:
        model.train()
        for batch in dataloader:
            batch = {k: batch[k].cuda() for k in batch}
            import pdb; pdb.set_trace()
            step += 1
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            output = model(input, attention_mask=attn_mask, labels=batch['labels'])

            total_loss += output.loss.item()

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            pred = output.logits.max(1)[1]

            total += len(pred)
            correct += (pred == batch['labels']).sum()

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
