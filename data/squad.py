from bisect import bisect_left
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Iterable, List
import unicodedata

from .dataset import TokenizedDataset, SentenceDropDataset
from .sentence import Sentence
from .example import ExampleWithSentences

@dataclass
class SQuADSpan(Sentence):
    start_labels: List[int] = field(default_factory=list)
    end_labels: List[int] = field(default_factory=list)

@dataclass
class SQuADExample(ExampleWithSentences):
    doc_title: str
    question_ids: List[int]

def tokens_to_char_offsets(context, tokens, UNK):
    offsets = []
    start = 0
    for t in tokens:
        if t.startswith('Ġ'):
            t = t.lstrip('Ġ')
        elif t.startswith('##'):
            t = t[2:]

        try:
            ofs = context.index(t, start)
        except:
            if t == UNK:
                ofs = start
                t = ''
            else:
                print(f"|{t}|\t|{context[start:]}|")
                raise
        offsets.append(ofs)
        start = ofs + len(t)
    return offsets
    
# From stackoverflow: https://stackoverflow.com/a/518232
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

class SQuADDataset(TokenizedDataset):
    def __init__(self, json_file, tokenizer_class="bert-base-uncased", eval=False, ner_to_idx=None, relation_to_idx=None):
        super().__init__(tokenizer_class=tokenizer_class)
        with open(json_file) as f:
            data = json.load(f)
        self.eval = eval
        
        self.data = []
        
        for datum in data['data']:
            doc_title = datum['title']
            for para in datum['paragraphs']:
                tokenized_para = self.tokenizer(para['context'], add_special_tokens=False)['input_ids']
                tokens = [self.tokenizer.decode([x]) for x in tokenized_para]
                ctx = para['context']
                if 'uncased' in tokenizer_class:
                    ctx = strip_accents(ctx.lower())
                char_offsets = tokens_to_char_offsets(ctx, tokens, self.tokenizer.unk_token)

                for qa in para['qas']:
                    question_ids = self.tokenizer(qa['question'], add_special_tokens=False)['input_ids']
                    ans_start = qa['answers'][0]['answer_start']
                    ans_text = qa['answers'][0]['text']

                    st = bisect_left(char_offsets, ans_start)
                    en = bisect_left(char_offsets, ans_start + len(ans_text)) - 1
                    
                    self.data.append(SQuADExample(tokenized_sentences=[SQuADSpan(i, [t], start_labels=[1 if i == st else 0], end_labels=[1 if i == en else 0])
                        for i, t in enumerate(tokenized_para)], doc_title=doc_title, question_ids=question_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

def squad_validate_fn(example):
    return any(any(x > 0 for x in s.start_labels) for s in example.tokenized_sentences) and any(any(x > 0 for x in s.end_labels) for s in example.tokenized_sentences)

def squad_collate_fn(examples: Iterable[SQuADExample], dataset: SQuADDataset):
    # filter out examples where the head entity is no longer available
    examples = list(filter(squad_validate_fn, examples))

    if len(examples) == 0:
        return

    context_lens = [sum(len(s.token_ids) for s in ex.tokenized_sentences) + len(ex.question_ids) + 3 for ex in examples]
    max_ctx_len = max(context_lens)
    
    context = np.zeros((len(examples), max_ctx_len), dtype=np.int64)
    attention_mask = np.zeros((len(examples), max_ctx_len), dtype=np.uint8)
    start_labels = np.zeros((len(examples), ), dtype=np.int64)
    end_labels = np.zeros((len(examples), ), dtype=np.int64)
    
    tokenizer = dataset.tokenizer
    CLS = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    SEP = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    for ex_i, ex in enumerate(examples):
        context[ex_i, 0] = CLS
        question = ex.question_ids
        context[ex_i, 1:1+len(question)] = question
        context[ex_i, len(question)+1] = SEP
        offset = len(question) + 2

        for span in ex.tokenized_sentences:
            context[ex_i, offset:offset+len(span.token_ids)] = span.token_ids

            for t_i in range(len(span.token_ids)):
                if span.start_labels[t_i] > 0:
                    start_labels[ex_i] = offset + t_i
                if span.end_labels[t_i] > 0:
                    end_labels[ex_i] = offset + t_i

            offset += len(span.token_ids)

        context[ex_i, offset] = SEP
        attention_mask[ex_i, :offset+1] = 1

    retval = {
        'context': context,
        'attention_mask': attention_mask,
        'start_labels': start_labels,
        'end_labels': end_labels
    }

    retval = {k: torch.from_numpy(retval[k]) for k in retval}

    retval['doc_title'] = [ex.doc_title for ex in examples]

    return retval

def squad_chunking_fn(ex, chunked_spans):
    # just construct new spans by concatenating tokenized spans in each chunk

    new_spans = []
    for s_i, chunk in enumerate(chunked_spans):
        toks = []
        start = []
        end = []
        for s in chunk:
            toks.extend(s.token_ids)
            start.extend(s.start_labels)
            end.extend(s.end_labels)
        new_spans.append(SQuADSpan(sentence_idx=s_i, token_ids=toks, start_labels=start, end_labels=end))

    new_ex = deepcopy(ex)
    new_ex.tokenized_sentences = new_spans
    return new_ex

if __name__ == "__main__":
    dataset = SQuADDataset("dataset/squad/dev-v1.1.json")
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1, 
        example_validate_fn=squad_validate_fn)

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataloader = DataLoader(sdrop_dataset, batch_size=2, 
        collate_fn=lambda examples: squad_collate_fn(examples, dataset=dataset))
    for batch in tqdm(dataloader):
        pass