from copy import deepcopy
import random
from scipy.stats import beta
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TokenizedDataset(Dataset):
    def __init__(self, *args, tokenizer_class="bert-base-uncased", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_class)

class SpanChunkDataset(Dataset):
    def __init__(self, dataset, span_type, *args, chunk_size=1, chunking_fn=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = dataset
        self.span_type = span_type
        self.chunk_size = chunk_size
        self.chunking_fn = chunking_fn

    def _chunk_spans(self, spans):
        res = []
        cur = []
        for s in spans:
            cur.append(s)
            if len(cur) >= self.chunk_size:
                res.append(cur)
                cur = []

        if len(cur) > 0:
            res.append(cur)

        return res

    def _naive_chunking_fn(self, ex, chunked_spans):
        # just construct new spans by concatenating tokenized spans in each chunk

        new_spans = []
        for s_i, chunk in enumerate(chunked_spans):
            toks = []
            for s in chunk:
                toks.extend(s.token_ids)
            new_spans.append(self.span_type(sentence_idx=s_i, token_ids=toks))

        new_ex = deepcopy(ex)
        new_ex.tokenized_sentences = new_spans
        return new_ex

    def __getitem__(self, key):
        ex = self.dataset[key]

        if self.chunk_size == 1:
            return ex

        chunking_fn = self.chunking_fn if self.chunking_fn is not None else lambda ex, chunks: self._naive_chunking_fn(ex, chunks)

        return chunking_fn(ex, self._chunk_spans(ex.tokenized_sentences))

    def __len__(self):
        return len(self.dataset)

class SentenceDropDataset(Dataset):
    def __init__(self, 
        dataset,
        *args,
        sent_drop_prob=0, 
        sent_keep_fn=lambda sentence: False,
        sent_drop_postproc=lambda example: example,
        example_validate_fn=lambda example: True,
        beta_drop=False,
        beta_drop_scale=1,
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
        # use the beta distribution
        self.beta_drop = beta_drop
        self.beta_drop_scale = beta_drop_scale

    def _sentence_drop_on_example(self, example):
        if self.sent_drop_prob == 0:
            return example
        new_ex = deepcopy(example)
        if self.sent_drop_prob > 0 and self.beta_drop:
            a = max(1, self.sent_drop_prob / (1 - self.sent_drop_prob))
            b = max(1, (1 - self.sent_drop_prob) / self.sent_drop_prob)
            sent_drop_prob = beta.rvs(a * self.beta_drop_scale, b * self.beta_drop_scale)
        else:
            sent_drop_prob = self.sent_drop_prob
        for sentence in new_ex.tokenized_sentences:
            if not self.sent_keep_fn(sentence) and random.random() < sent_drop_prob:
                sentence.marked_for_deletion = True

        # perform dataset-specific postprocessing to propagate the effect of sentence removal if necessary
        new_ex = self.sent_drop_postproc(new_ex)

        new_ex.tokenized_sentences = list(filter(lambda sentence: not sentence.marked_for_deletion, new_ex.tokenized_sentences))
        
        # renumber sentences
        for s_i in range(len(new_ex.tokenized_sentences)):
            new_ex.tokenized_sentences[s_i].sentence_idx = s_i

        return new_ex

    def __getitem__(self, key):
        # try different sentence drop patterns until we end up with at least a valid example
        retries = 0
        ex = self._sentence_drop_on_example(self.dataset[key])
        while not self.example_validate_fn(ex):
            retries += 1
            ex = self._sentence_drop_on_example(self.dataset[key])
            if retries > 10:
                # don't wait forever, just return the original sample
                return self.dataset[key]
        return ex

    def estimate_label_noise(self, reps=1, validation_fn=lambda x: True):
        failed = 0
        total = 0
        for ex in self.dataset:
            for _ in range(reps):
                total += 1
                if not validation_fn(self._sentence_drop_on_example(ex)):
                    failed += 1
        return failed, total

    def __len__(self):
        return len(self.dataset)
