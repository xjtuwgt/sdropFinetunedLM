from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

from data.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data.dataset import SentenceDropDataset

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--sent_dropout', type=float, default=.1)
    parser.add_argument('--train_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10)

    parser.add_argument('--validate_examples', action='store_true')

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()

    torch.cuda.manual_seed_all(args.seed)

    bert_config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
    bert_config.num_hidden_layers = 3
    bert_config.vocab_size = args.vocab_size

    model = transformers.AutoModelForSequenceClassification.from_config(bert_config)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    dataset = FindCatDataset(seed=args.seed, total_examples=args.train_examples)
    dev_dataset = FindCatDataset(seed=314, total_examples=args.test_examples)
    validation_fn = find_cat_validation_fn if args.validate_examples else lambda ex: True
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.sent_dropout,
        example_validate_fn=validation_fn)

    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)

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
            step += 1
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            output = model(input, attention_mask=attn_mask, labels=batch['labels'])

            total_loss += output.loss.item()
            #print(f"Step {step:6d}: loss={output.loss.item()}")

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

            if step >= args.steps:
                break

        if step >= args.steps:
            break

    print(f"Best dev result: dev accuracy={best_dev_acc:.6f} at step {best_step}")
