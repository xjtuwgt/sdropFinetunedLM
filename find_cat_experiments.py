from argparse import ArgumentParser
from collections import namedtuple
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

from data.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn, PAD, FindCatSentence
from data.dataset import SentenceDropDataset, SpanChunkDataset

BiLSTMClassifierOutput = namedtuple("BiLSTMClassifierOutput", ["loss", "logits"])

class BiLSTMClassifier(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_dim, dropout=0):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD)

        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers, bidirectional=True, 
            dropout=dropout, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, 2)

        self.crit = nn.CrossEntropyLoss()

    def forward(self, input, labels, attention_mask=None):
        emb = self.embeddings(input)
        emb = nn.utils.rnn.pack_padded_sequence(emb, lengths=attention_mask.sum(-1).cpu(), batch_first=True, enforce_sorted=False)
        hid = self.bilstm(emb)[0]
        hid = nn.utils.rnn.pad_packed_sequence(hid, batch_first=True)[0]
        first_hid = hid[:, 0]

        logits = self.classifier(first_hid)
        loss = self.crit(logits, labels)
        return BiLSTMClassifierOutput(logits=logits, loss=loss)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--sent_dropout', type=float, default=.1)
    parser.add_argument('--train_examples', type=int, default=1000)
    parser.add_argument('--test_examples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval_every', type=int, default=100)

    parser.add_argument('--train_seqlen', type=int, default=300)
    parser.add_argument('--test_seqlen', type=int, default=300)
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--beta_drop', action='store_true')
    parser.add_argument('--beta_drop_scale', default=1, type=float)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model_type', default='bert-like', choices=['bert-like', 'bilstm'])

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.model_type == 'bert-like':
        bert_config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
        bert_config.num_hidden_layers = 3
        bert_config.vocab_size = args.vocab_size

        model = transformers.AutoModelForSequenceClassification.from_config(bert_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    elif args.model_type == 'bilstm':
        model = BiLSTMClassifier(3, args.vocab_size, 512, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    model.cuda()

    dataset = FindCatDataset(seed=args.seed, total_examples=args.train_examples, seqlen=args.train_seqlen)
    dev_dataset = FindCatDataset(seed=314, total_examples=args.test_examples, seqlen=args.test_seqlen)

    dataset = SpanChunkDataset(dataset, FindCatSentence, chunk_size=args.chunk_size)
    dev_dataset = SpanChunkDataset(dev_dataset, FindCatSentence, chunk_size=args.chunk_size)

    validation_fn = find_cat_validation_fn if args.validate_examples else lambda ex: True

    random.seed(args.seed)
    np.random.seed(args.seed)
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=args.sent_dropout,
        example_validate_fn=validation_fn, beta_drop=args.beta_drop, beta_drop_scale=args.beta_drop_scale)

    if args.sent_dropout > 0:
        failed, total = sdrop_dataset.estimate_label_noise(reps=max(1, 10000 // len(dataset)), validation_fn=find_cat_validation_fn)
        print(f"Label noise: {failed / total * 100 :6.3f}% ({failed:7d} / {total:7d})", flush=True)
    else:
        print(f"Label noise: 0% (- / -)", flush=True)

    dataloader = DataLoader(sdrop_dataset, batch_size=args.batch_size, collate_fn=find_cat_collate_fn, shuffle=True)
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
            attn_mask = (batch['input'] > 0)
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
                print(f"Step {step}: train accuracy={correct / total:.6f}, train loss={total_loss / total}", flush=True)

                model.eval()
                total = 0
                correct = 0
                with torch.no_grad():
                    for batch in dev_dataloader:
                        batch = {k: batch[k].cuda() for k in batch}
                        input = batch['input'].clamp(min=0)
                        attn_mask = (batch['input'] > 0)
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
