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
INF_NUM = 1e12

class TransformerModelForDocRED(nn.Module):
    def __init__(self, model_type, rel_vocab_size, ner_vocab_size, ent_emb_block_size=64, sentence_emb_size=128, rel_emb_size=32, lambda_evi=1, num_labels=4):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.model.config.hidden_size
        self.lambda_evi = lambda_evi

        ent_emb_size = 768
        self.ner_embedding = nn.Embedding(ner_vocab_size, self.hidden_size, padding_idx=0)
        self.ner_embedding.weight.data.zero_()
        self.rel_hid_h = nn.Linear(self.hidden_size, ent_emb_size)
        self.rel_hid_t = nn.Linear(self.hidden_size, ent_emb_size)
        self.ent_emb_block_size = ent_emb_block_size
        self.ent_emb_blocks = ent_emb_size // ent_emb_block_size
        self.rel_bilinear = nn.Linear(ent_emb_block_size * ent_emb_size, rel_vocab_size + 1)
        self.num_labels = num_labels # max number of predictions per context
        self.emb_dropout = nn.Dropout(.2)
        self.dropout = nn.Dropout(.1)

        sentence_emb_size = 768
        self.sentence_embedder = nn.Linear(self.hidden_size, sentence_emb_size)
        self.pair_embedder = nn.Linear(self.hidden_size * 2, sentence_emb_size)

        self.sent_to_hid = nn.Linear(self.hidden_size, sentence_emb_size)
        self.pair_to_hid = nn.Linear(self.hidden_size * 2, sentence_emb_size)
            
        self.sent_to_evi = nn.Linear(sentence_emb_size, rel_vocab_size)

        # self.rel_embedding = nn.Embedding(rel_vocab_size, rel_emb_size)
        # nn.init.normal_(self.rel_embedding.weight.data, std=rel_emb_size ** -0.5)
        # self.evi_trilinear = nn.Parameter(torch.empty(sentence_emb_size, sentence_emb_size, rel_emb_size))
        # nn.init.normal_(self.evi_trilinear.data, std=(sentence_emb_size ** -1))
        # self.sentence_to_hid = nn.Linear(sentence_emb_size, sentence_emb_size)
        # self.pair_to_hid = nn.Linear(sentence_emb_size, sentence_emb_size)
        # self.rel_to_hid = nn.Linear(rel_emb_size, sentence_emb_size)
        # self.hid_to_score = nn.Linear(sentence_emb_size, 1)

        self.crit = nn.BCEWithLogitsLoss()

    def masked_atlop_loss(self, input, target, mask=None):
        if mask is not None:
            if not mask.any():
                return 0
            input = input.masked_fill(mask.logical_not(), -INF_NUM)

        # using 0 as the threshold value for simplicity, and assuming that the last dimension is what the softmax is over
        input_size = list(input.size())

        threshold = input.new_zeros(input_size[:-1] + [1])

        new_input = torch.cat([threshold, input], -1)

        pos_label = torch.cat([target.new_zeros(input_size[:-1] + [1]), target], -1)
        neg_label = torch.cat([target.new_ones(input_size[:-1] + [1]), target.new_zeros(target.size())], -1)

        pos_loss = -(torch.log_softmax(new_input.masked_fill(pos_label + neg_label == 0, -INF_NUM), -1) * pos_label).sum(-1)
        neg_loss = -(torch.log_softmax(new_input.masked_fill(pos_label > 0, -INF_NUM), -1) * neg_label).sum(-1)

        loss =  (pos_loss + neg_loss)
        if mask is not None:
            loss = loss.masked_select(mask.any(-1, keepdim=True))
        return loss.mean()

    def masked_bce_loss(self, input, target, mask):
        if not mask.any():
            return 0
        return self.crit(input.masked_select(mask), target.masked_select(mask))

    def forward(self, batch):
        MAXLEN = 512
        inputs_embeds = self.model.get_input_embeddings()(batch['context'][:, :MAXLEN])
        inputs_embeds += self.emb_dropout(self.ner_embedding(batch['ner'][:, :MAXLEN]))
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'][:, :MAXLEN])
        # output = self.model(input_ids=batch['context'][:, :MAXLEN], attention_mask=batch['attention_mask'][:, :MAXLEN])
        hid = output.last_hidden_state

        # relation prediction
        ent_start = batch['entity_start'].clamp(min=0, max=MAXLEN-1)
        ent_start_mask = (batch['entity_start'] <  0) | (batch['entity_start'] >= MAXLEN)
        # # average pool
        # avg_mask = ent_mask / (ent_mask.sum(2, keepdim=True) + 1e-6)
        # ent_hid = avg_mask.bmm(hid)
        # logsumexp pool
        ent_hid = self.dropout(hid).gather(1, ent_start.view(hid.size(0), -1).unsqueeze(-1).repeat(1, 1, self.hidden_size)).view(hid.size(0), ent_start.size(1), ent_start.size(2), self.hidden_size)
        ent_hid.masked_fill_(ent_start_mask.unsqueeze(-1), -INF_NUM) 
        ent_hid = self.dropout(ent_hid.logsumexp(2))
        
        pairs = batch['entity_pairs'].clamp(min=0)
        pair_hid = ent_hid.gather(1, pairs.view(pairs.size(0), -1, 1).repeat(1, 1, self.hidden_size)).view(pairs.size(0), pairs.size(1), pairs.size(2), self.hidden_size)
        # head_hid = ent_hid.gather(1, pairs[:, :, :1].repeat(1, 1, self.hidden_size))
        # tail_hid = ent_hid.gather(1, pairs[:, :, 1:].repeat(1, 1, self.hidden_size))
        head_emb = torch.tanh(self.rel_hid_h(pair_hid[:, :, 0])).view(pair_hid.size(0), pair_hid.size(1), self.ent_emb_blocks, self.ent_emb_block_size)
        tail_emb = torch.tanh(self.rel_hid_t(pair_hid[:, :, 1])).view(pair_hid.size(0), pair_hid.size(1), self.ent_emb_blocks, self.ent_emb_block_size)
        head_emb = self.dropout(head_emb)
        tail_emb = self.dropout(tail_emb)

        pair_intermediate = (head_emb.unsqueeze(-1) * tail_emb.unsqueeze(-2)).view(pair_hid.size(0), pair_hid.size(1), -1)
        pair_logits0 = self.rel_bilinear(pair_intermediate)
        pair_logits0 = pair_logits0[:, :, 1:] - pair_logits0[:, :, :1]

        valid_pair_mask = (batch['entity_pairs'][:, :, :1] >= 0)
        pair_logits = pair_logits0.masked_select(valid_pair_mask).view(-1, pair_logits0.size(-1))
        pair_labels = batch['pair_labels'].masked_select(valid_pair_mask).view(-1, pair_logits0.size(-1)).float()
        loss = self.masked_atlop_loss(pair_logits, pair_labels)
        # loss = self.masked_bce_loss(pair_logits0, batch['pair_labels'].float(), batch['entity_pairs'][:, :, :1] >= 0)

        pair_logits0 = pair_logits0.masked_fill(pair_logits0 < torch.topk(pair_logits0, self.num_labels, dim=-1)[0][:, :, -1:], -INF_NUM)

        # evidence prediction
        
        # sent_idx = batch['sentence_start_end'].clamp(min=0, max=MAXLEN-1).view(hid.size(0), -1)
        # sent_hid = hid.gather(1, sent_idx.unsqueeze(-1).repeat(1, 1, self.hidden_size)).view(hid.size(0), -1, 2, self.hidden_size)
        # sent_hid = torch.cat([sent_hid.view(hid.size(0), -1, self.hidden_size * 2), sent_hid[:, :, 0] * sent_hid[:, :, 1]], -1)
        # sent_hid = sent_hid.view(hid.size(0), -1, self.hidden_size * 2)
        # clf_hid = hid[:, :1]
        # sent_hid_clf = torch.cat([clf_hid, clf_hid], -1)

        # pair_hid = torch.cat([pair_hid.view(hid.size(0), -1, self.hidden_size * 2), pair_hid[:, :, 0] * pair_hid[:, :, 1]], -1)

        sent_emb = self.sentence_embedder(self.dropout(hid))# - self.sentence_embedder(self.dropout(sent_hid_clf))
        pair_emb = self.pair_embedder(pair_hid.view(pair_hid.size(0), pair_hid.size(1), self.hidden_size * 2))

        attn = pair_emb.bmm(sent_emb.transpose(1, 2)) * (sent_emb.size(-1) ** -0.5)
        attn = attn.unsqueeze(2).masked_fill(batch['sentence_mask'][:, :, :MAXLEN].unsqueeze(1) == 0, -INF_NUM)
        attn = torch.softmax(attn, -1)
        sent_hid = attn.view(attn.size(0), -1, sent_emb.size(-2)).bmm(self.dropout(hid)).view(attn.size(0), attn.size(1), attn.size(2), -1)

        pair_sent_hid = self.sent_to_hid(sent_hid)
        # pair_sent_hid += self.pair_to_hid(pair_hid.view(pair_hid.size(0), pair_hid.size(1), self.hidden_size * 2)).unsqueeze(-2)
        pair_sent_hid = torch.relu(pair_sent_hid)

        pair_sent_score0 = self.sent_to_evi(self.dropout(pair_sent_hid)).transpose(-1, -2)
        
        # hid = self.pair_to_hid(pair_emb).unsqueeze(2).unsqueeze(2)
        # hid = hid + self.sentence_to_hid(sent_emb).unsqueeze(1).unsqueeze(1)
        # hid = hid + self.rel_to_hid(self.rel_embedding.weight.data).unsqueeze(1).unsqueeze(0).unsqueeze(0)
        # pair_sent_score0 = self.hid_to_score(self.dropout(torch.relu(hid / 3))).squeeze(-1)

        # intermediate = torch.einsum('def,rf,bse->drbs', self.evi_trilinear, self.dropout(torch.tanh(self.rel_embedding.weight)), sent_emb)
        # pair_sent_score0 = torch.einsum('drbs,bpd->bprs', intermediate, pair_emb)

        positive_pair_mask = batch['pair_labels'].unsqueeze(-1) > 0
        pair_sent_score = pair_sent_score0.masked_select(positive_pair_mask).view(-1, pair_sent_score0.size(-1))
        pair_sent_label = batch['pair_evidence'].float().masked_select(positive_pair_mask).view(-1, pair_sent_score0.size(-1))
        pair_sent_mask = (batch['sentence_mask'].any(-1).unsqueeze(1).unsqueeze(1) == 1).masked_select(positive_pair_mask).view(-1, pair_sent_score0.size(-1))
        loss += self.lambda_evi * self.masked_atlop_loss(pair_sent_score, pair_sent_label, pair_sent_mask) * positive_pair_mask.sum() / valid_pair_mask.sum()
        # loss += self.lambda_evi * self.masked_bce_loss(pair_sent_score0, batch['pair_evidence'].float(), ((batch['pair_labels'].unsqueeze(-1) > 0) & (batch['sentence_start_end'][:, :, 0].unsqueeze(1).unsqueeze(1) >= 0)))

        return DocREDModelOutput(loss=loss, pair_logits=pair_logits0, pair_sent_score=pair_sent_score0)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--span_dropout', type=float, default=.1)
    parser.add_argument('--train_file', type=str, default="train_annotated.json")
    parser.add_argument('--dev_file', type=str, default="dev.json")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=450000)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--report_every', type=int, default=20)
    parser.add_argument('--lr_warmup_portion', type=float, default=0.06)
    parser.add_argument('--gradient_acc_steps', type=int, default=20)

    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--validate_examples', action='store_true')
    parser.add_argument('--beta_drop', action='store_true')
    parser.add_argument('--beta_drop_scale', default=1, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lambda_evi', default=1, type=float)

    parser.add_argument('--model_type', type=str, default="google/electra-base-discriminator")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_tqdm', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DocREDDataset(f"dataset/docred/{args.train_file}", tokenizer_class=args.model_type)
    dev_dataset = DocREDDataset(f"dataset/docred/{args.dev_file}", tokenizer_class=args.model_type,
        ner_to_idx=dataset.ner_to_idx, relation_to_idx=dataset.relation_to_idx, eval=True)

    model = TransformerModelForDocRED(args.model_type, len(dataset.relation_to_idx), len(dataset.ner_to_idx), lambda_evi=args.lambda_evi, num_labels=args.num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(args.lr_warmup_portion * args.steps / args.gradient_acc_steps), args.steps // args.gradient_acc_steps)

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
    dataloader = DataLoader(sdrop_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    step = 0
    best_dev_score = -1
    best_dev_result = None
    best_step = -1

    mytqdm = (lambda x: x) if args.no_tqdm else tqdm

    optimizer.zero_grad()
    while True:
        model.train()
        for batch in dataloader:
            if batch is None: continue
            batch = {k: batch[k].cuda() if isinstance(batch[k], torch.Tensor) else batch[k] for k in batch}
            step += 1
            output = model(batch)

            loss = output.loss / args.gradient_acc_steps
            loss.backward()

            gnorm = 0
            if isinstance(output.loss, torch.Tensor) and step % args.gradient_acc_steps == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.report_every == 0:
                print(f"Step {step:6d}: loss={output.loss.item():6f}, lr={optimizer.param_groups[0]['lr']:3e}, gradnorm={gnorm:6f}", flush=True)

            if step % args.eval_every == 0:
                model.eval()
                predictions = []
                print("Evaluating on the dev set...", flush=True)
                with torch.no_grad():
                    for batch in mytqdm(dev_dataloader):
                        pair_ids = batch['entity_pairs'].numpy()
                        is_sent = batch['sentence_mask'].any(-1).numpy()
                        batch = {k: batch[k].cuda() if isinstance(batch[k], torch.Tensor) else batch[k] for k in batch}
                        output = model(batch)

                        # decode output
                        pair_pred = (output.pair_logits > 0).detach().cpu().numpy()
                        pair_sent_pred = (output.pair_sent_score > 0).detach().cpu().numpy()

                        for ex_i, pair_i, r in zip(*np.where(pair_pred)):
                            head, tail = pair_ids[ex_i, pair_i]
                            if head < 0: continue
                            evidence = [sent_i for sent_i in np.where(pair_sent_pred[ex_i, pair_i, r])[0] if is_sent[ex_i, sent_i]]

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

                re_f1 = dev_result['re_f1_ignore_train_annotated']
                evi_f1 = dev_result['evi_f1']
                dev_score = re_f1 + evi_f1
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
