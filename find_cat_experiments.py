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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=100)

    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()

    dataset = FindCatDataset()
    dev_dataset = FindCatDataset(seed=314)
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.sent_dropout,
        example_validate_fn=lambda ex: find_cat_validation_fn(ex, dataset=dataset))

    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn)

    bert_config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
    bert_config.num_hidden_layers = 3
    bert_config.vocab_size = args.vocab_size

    model = transformers.AutoModelForSequenceClassification.from_config(bert_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    crit = torch.nn.BCEWithLogitsLoss()
    step = 0

    for epoch in range(10):
        model.train()
        total = 0
        correct = 0
        for batch in tqdm(dataloader):
            step += 1
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            output = model(input, attention_mask=attn_mask, labels=batch['labels'])

            print(f"Step {step:6d}: loss={output.loss.item()}")

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            pred = output.logits.max(1)[1]

            total += len(pred)
            correct += (pred == batch['labels']).sum()
        print(f"Epoch {epoch}: train accuracy={correct / total:.6f}")

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                input = batch['input'].clamp(min=0)
                attn_mask = (input >= 0)
                output = model(input, attention_mask=attn_mask, labels=batch['labels'])
                pred = output.logits.max(1)[1]

                total += len(pred)
                correct += (pred == batch['labels']).sum()

        print(f"Epoch {epoch}: dev accuracy={correct / total:.6f}")