from dataclasses import dataclass, field
import numpy as np
import random
import re
import torch

from .dataset import TokenizedDataset, SentenceDropDataset
from .sentence import Sentence
from .example import ExampleWithSentences

@dataclass
class FindCatSentence(Sentence):
    pass

@dataclass
class FindCatExample(ExampleWithSentences):
    label : int = 0

def contains_subsequence(target, sequence):
    if len(target) == 0:
        return True
    
    if target[0] not in sequence:
        return False
    else:
        for i in range(len(sequence)-len(target)+1):
            if sequence[i] == target[0]:
                subresult = contains_subsequence(target[1:], sequence[i+1:])
                if subresult:
                    return True
        return False


class FindCatDataset(TokenizedDataset):
    def __init__(self, tokenizer_class="bert-base-uncased", 
        total_examples=1000, seqlen=300, vocab=list(range(1, 27)), target=[ord(x)-ord('a')+1 for x in 'cat'], 
        fixed_positions=None, eval=False, seed=42):
        super().__init__(tokenizer_class=tokenizer_class)

        self.seqlen = seqlen
        self.vocab = vocab
        self.target = target
        self.fixed_positions = fixed_positions
        random.seed(seed)
        self.data = [self._generate_example() for _ in range(total_examples)]

    def _generate_example(self):
        target = int(random.random() > 0.5)

        if target == 0:
            retval = random.choices(self.vocab, k=self.seqlen)
            while contains_subsequence(self.target, retval):
                retval = random.choices(self.vocab, k=self.seqlen)
        else:
            retval = random.choices(self.vocab, k=self.seqlen)
            if self.fixed_positions is not None:
                assert len(self.fixed_positions) == len(self.target)
                positions = self.fixed_positions
            else:
                positions = sorted(random.choices(list(range(self.seqlen)), k=len(self.target)))

            for p_i, p in enumerate(positions):
                retval[p] = self.target[p_i]
        
        return FindCatExample(tokenized_sentences=[FindCatSentence(sentence_idx=s_i, token_ids=[s]) for s_i, s in enumerate(retval)], label=target)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

def find_cat_collate_fn(examples):
    sent_lens = [len(ex.tokenized_sentences) for ex in examples]
    max_sent_len = max(sent_lens)

    batched_input = np.full((len(examples), max_sent_len), -1, dtype=np.int64)
    batched_labels = np.zeros((len(examples),), dtype=np.int64)

    for ex_i, ex in enumerate(examples):
        batched_input[ex_i, :len(ex.tokenized_sentences)] = [s.token_ids[0] for s in ex.tokenized_sentences]
        batched_labels[ex_i] = ex.label

    retval = {
        'input': batched_input,
        'labels': batched_labels
    }

    retval = {k: torch.from_numpy(retval[k]) for k in retval}

    return retval

def find_cat_validation_fn(ex, dataset):
    return (ex.label == 0) or contains_subsequence(dataset.target, [s.token_ids[0] for s in ex.tokenized_sentences])

if __name__ == "__main__":
    dataset = FindCatDataset()
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1, 
        example_validate_fn=lambda ex: find_cat_validation_fn(ex, dataset=dataset))

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataloader = DataLoader(sdrop_dataset, batch_size=32, collate_fn=find_cat_collate_fn)

    for batch in tqdm(dataloader):
        pass