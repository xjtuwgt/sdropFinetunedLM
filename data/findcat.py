from dataclasses import dataclass
import numpy as np
import os
import pickle
import random
import re
import torch
from typing import List

from .dataset import TokenizedDataset, SentenceDropDataset
from .sentence import Sentence
from .example import ExampleWithSentences

@dataclass
class FindCatSentence(Sentence):
    pass

@dataclass
class FindCatExample(ExampleWithSentences):
    target_tokens: List[int]
    positions: List[int]
    label : int = 0

def contains_subsequence(target, sequence):
    if len(target) == 0:
        return True
    
    remaining = sequence
    matched = 0
    for t in target:
        idx = 0
        while idx < len(remaining) and remaining[idx] != t:
            idx += 1
        if idx >= len(remaining):
            return False
        else:
            matched += 1
            if matched == len(target):
                return True
            remaining = remaining[idx+1:]

RESERVED_TOKENS = 10
PAD = 0
CLS = 1
SEP = 2
MASK = 3

class FindCatDataset(TokenizedDataset):
    def __init__(self, tokenizer_class="bert-base-uncased", 
        total_examples=1000, seqlen=300, vocab=list(range(RESERVED_TOKENS, RESERVED_TOKENS+26)), target_tokens=[[ord(x)-ord('a')+RESERVED_TOKENS for x in 'cat']], 
        fixed_positions=None, eval=False, seed=42, cache_dir="dataset/findcat"):
        super().__init__(tokenizer_class=tokenizer_class)

        self.seqlen = seqlen
        self.vocab = vocab
        self.target_tokens = target_tokens
        self.fixed_positions = fixed_positions
        random.seed(seed)

        if cache_dir is not None:
            CACHE_FILE = f"{cache_dir}/findcat_total_{total_examples}_seqlen_{seqlen}_vocab_[{min(vocab)}-{max(vocab)}]_target_{repr(target_tokens)}_fixed_{repr(fixed_positions)}_seed_{seed}_v2.pkl"
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                self.data = [self._generate_example() for _ in range(total_examples)]
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(self.data, f)
        else:
            self.data = [self._generate_example() for _ in range(total_examples)]

    def _generate_negative_example(self, target_tokens):
        V = len(self.vocab)
        if not hasattr(self, 'pos_count_over_V_n'):
            #                             total number of length-n sequences that don't contain a specific length-m subsequence
            # pos_count_over_V_n[n, m] = ---------------------------------------------------------------------------------------
            #                                                                     V^n
            pos_count_over_V_n = np.zeros((self.seqlen+1, len(target_tokens)+1))
            for n in range(1, self.seqlen+1):
                for m in range(1, min(n+1, max(len(t) for t in self.target_tokens)+1)):
                    if m == n:
                        pos_count_over_V_n[n, m] = 1 - V ** (-n)
                    elif m == 1:
                        pos_count_over_V_n[n, m] = ((V - 1) / V) ** n
                    else:
                        pos_count_over_V_n[n, m] = (pos_count_over_V_n[n-1, m-1] + (V - 1) * pos_count_over_V_n[n-1, m]) / V

            self.pos_count_over_V_n = pos_count_over_V_n

        retval = []
        matched = 0
        for i in range(self.seqlen):
            n = self.seqlen - i
            m = len(target_tokens) - matched

            if m > n:
                # remaining target is shorter than remaining whole sequence
                retval.append(random.choice(self.vocab))
            else:
                match_weight = self.pos_count_over_V_n[n-1, m-1]
                unmatch_weight = self.pos_count_over_V_n[n-1, m]
                p = np.full(V, unmatch_weight)
                p[self.vocab.index(target_tokens[matched])] = match_weight
                retval.extend(random.choices(self.vocab, weights=p))

            if retval[-1] == target_tokens[matched]:
                matched += 1

        assert not contains_subsequence(target_tokens, retval)

        return retval

    def _generate_example(self):
        target = int(random.random() > 0.5)
        target_tokens = random.choice(self.target_tokens) # randomly choose one animal if more than one is provided

        retval = self._generate_negative_example(target_tokens) # start from a negative example that doesn't already contain the target subsequence

        positions = []
        if target == 1:
            if self.fixed_positions is not None:
                assert len(self.fixed_positions) == len(target_tokens)
                positions = self.fixed_positions
            else:
                positions = sorted(random.choices(list(range(self.seqlen)), k=len(target_tokens)))

            for p_i, p in enumerate(positions):
                retval[p] = target_tokens[p_i]
        
        return FindCatExample(tokenized_sentences=[FindCatSentence(sentence_idx=s_i, token_ids=[s]) for s_i, s in enumerate(retval)], 
            target_tokens=target_tokens, positions=positions, label=target)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

def find_cat_collate_fn(examples):
    ex_lens = [3 + len(ex.target_tokens) + sum(len(s.token_ids) for s in ex.tokenized_sentences) for ex in examples]
    max_ex_len = max(ex_lens)

    batched_input = np.full((len(examples), max_ex_len), -1, dtype=np.int64)
    batched_labels = np.zeros((len(examples),), dtype=np.int64)

    for ex_i, ex in enumerate(examples):
        batched_input[ex_i, :ex_lens[ex_i]] = [CLS] + ex.target_tokens + [SEP] + [t for s in ex.tokenized_sentences for t in s.token_ids] + [SEP]
        batched_labels[ex_i] = ex.label

    retval = {
        'input': batched_input,
        'labels': batched_labels
    }

    retval = {k: torch.from_numpy(retval[k]) for k in retval}

    return retval

def find_cat_validation_fn(ex):
    return (ex.label == 0) or contains_subsequence(ex.target_tokens, [t for s in ex.tokenized_sentences for t in s.token_ids])

if __name__ == "__main__":
    dataset = FindCatDataset()
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1, 
        example_validate_fn=lambda ex: find_cat_validation_fn(ex, dataset=dataset))

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataloader = DataLoader(sdrop_dataset, batch_size=32, collate_fn=find_cat_collate_fn)

    for batch in tqdm(dataloader):
        pass