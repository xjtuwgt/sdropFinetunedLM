from copy import deepcopy
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TokenizedDataset(Dataset):
    def __init__(self, *args, tokenizer_class="bert-base-uncased", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_class)

class SentenceDropDataset(Dataset):
    def __init__(self, 
        dataset,
        *args,
        sent_drop_prob=0, 
        sent_keep_fn=lambda sentence: False,
        sent_drop_postproc=lambda example: example,
        example_validate_fn=lambda example: True,
        **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        # probability a sentence is dropped
        self.sent_drop_prob = sent_drop_prob
        # test whether a sentence should be kept regardless of the random dropping process,
        # function must return true for sentences that are kept whatsoever
        self.sent_keep_fn = sent_keep_fn
        # dataset-specific postprocessing for examples after sentence drop
        self.sent_drop_postproc = sent_drop_postproc
        # make sure examples are actually well-formed for training
        self.example_validate_fn = example_validate_fn

    def _sentence_drop_on_example(self, example):
        new_ex = deepcopy(example)
        for sentence in new_ex.tokenized_sentences:
            if not self.sent_keep_fn(sentence) and random.random() < self.sent_drop_prob:
                sentence.marked_for_deletion = True

        # perform dataset-specific postprocessing to propagate the effect of sentence removal if necessary
        new_ex = self.sent_drop_postproc(new_ex)

        new_ex.tokenized_sentences = list(filter(lambda sentence: not sentence.marked_for_deletion, new_ex.tokenized_sentences))
        
        # renumber sentences
        for s_i, s in enumerate(new_ex.tokenized_sentences):
            s.sentence_idx = s_i

        return new_ex

    def __getitem__(self, key):
        # try different sentence drop patterns until we end up with at least a valid example
        ex = self._sentence_drop_on_example(self.dataset[key])
        while not self.example_validate_fn(ex):
            ex = self._sentence_drop_on_example(self.dataset[key])
        return ex

    def __len__(self):
        return len(self.dataset)
