from collections import defaultdict
from copy import deepcopy
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Iterable

from example import DocREDExample, DocREDMention, DocREDEntity, DocREDRelation, DocREDSentence

class TokenizedDataset(Dataset):
    def __init__(self, *args, tokenizer_class="bert-base-uncased", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_class)

class DocREDDataset(TokenizedDataset):
    def __init__(self, json_file, tokenizer_class="bert-base-uncased", eval=False):
        super().__init__(tokenizer_class=tokenizer_class)
        with open(json_file) as f:
            data = json.load(f)
        self.eval = eval
        
        self.data = []
        self.relation_to_idx = {}
        self.idx_to_relation = []
        self.ner_to_idx = {}
        self.idx_to_ner = []
        for datum in data:
            sentences = datum['sents']
            # tokenize each word with the wordpiece tokenizer, so each sentence will become a list of lists of wordpiece ids
            tokenized_sentences = [self.tokenizer(s, add_special_tokens=False)['input_ids'] for s in sentences]
            token_to_subword = [dict() for _ in sentences]
            # concatenate wordpieces together for each sentence
            for s_i, s in enumerate(tokenized_sentences):
                subword_offset = 0
                s1 = []
                for w_i, subw in enumerate(s):
                    token_to_subword[s_i][w_i] = subword_offset
                    subword_offset += len(subw)
                    s1.extend(subw)
                tokenized_sentences[s_i] = s1
                
            sentences = [DocREDSentence(sentence_idx=sent_id, token_ids=sent) for sent_id, sent in enumerate(tokenized_sentences)]
            
            # map mention offsets to wordpiece tokenization
            entities = []
            for entity in datum['vertexSet']:
                mentions = []
                for mention in entity:
                    m1 = {k: mention[k] for k in mention}
                    sent_id = m1['sent_id']
                    st, en = m1['pos']
                    st = token_to_subword[sent_id][st]
                    if en < len(token_to_subword[sent_id]):
                        en = token_to_subword[sent_id][en]
                    else:
                        en = len(tokenized_sentences[sent_id])

                    if m1['type'] not in self.ner_to_idx:
                        self.ner_to_idx[m1['type']] = len(self.ner_to_idx)
                        self.idx_to_ner.append(m1['type'])
                    m1['type'] = self.ner_to_idx[m1['type']]
                    m1 = DocREDMention(sentence=sentences[sent_id], start=st, end=en, ner_type=m1['type'])
                    sentences[sent_id].mentions.append(m1)
                    mentions.append(m1)
                e1 = DocREDEntity(mentions=mentions)
                # add backpointers from mentions to entities for convenience
                for m1 in mentions:
                    m1.parent = e1
                entities.append(e1)

            head_to_examples = defaultdict(list)

            for r in datum['labels']:
                head, tail, relation, evidence = r['h'], r['t'], r['r'], r['evidence']
                if relation not in self.relation_to_idx:
                    self.relation_to_idx[relation] = len(self.relation_to_idx)
                    self.idx_to_relation.append(relation)
                relation = self.relation_to_idx[relation]
                head_to_examples[head].append(DocREDRelation(head_entity=entities[head], tail_entity=entities[tail], relation=relation, evidence=[sentences[si] for si in evidence]))

            for head in head_to_examples:
                example = DocREDExample(sentences, entities[head], entities, head_to_examples[head])
                self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

def docred_sent_drop_postproc(example: DocREDExample):
    # remove entities that no longer have mentions in the remaining sentences,
    # and remove relations that have lost either entity (head or tail)

    for sentence in example.tokenized_sentences:
        if sentence.marked_for_deletion:
            for mention in sentence.mentions:
                mention.marked_for_deletion = True

    for entity in example.entities:
        entity.mentions = list(filter(lambda m: not m.marked_for_deletion, entity.mentions))

    example.relations = list(filter(lambda r: len(r.head_entity.mentions) > 0 and len(r.tail_entity.mentions) > 0, example.relations))
    example.entities = list(filter(lambda e: len(e.mentions) > 0, example.entities))

    return example

def docred_collate_fn(examples: Iterable[DocREDExample]):
    # TODO: implement the actual collate function to put examples together into a batch of tensors
    return examples

class SentenceDropDataLoader(DataLoader):
    def __init__(self, 
        *args,
        sent_drop_prob=0, 
        sent_keep_fn=lambda sentence: False,
        sent_drop_postproc=lambda example: example,
        example_collate_fn=None,
        **kwargs
        ):

        kwargs['collate_fn'] = self._collate_fn

        super().__init__(*args, **kwargs)

        assert example_collate_fn is not None, "Must specify collate function (example_collate_fn) " \
            "for SentenceDropDataLoader to accommodate data examples from different datasets."

        # probability a sentence is dropped
        self.sent_drop_prob = sent_drop_prob
        # test whether a sentence should be kept regardless of the random dropping process,
        # function must return true for sentences that are kept whatsoever
        self.sent_keep_fn = sent_keep_fn
        # dataset-specific postprocessing for examples after sentence drop
        self.sent_drop_postproc = sent_drop_postproc
        # collate function to convert a list of examples in a batch to pytorch tensors
        self.example_collate_fn = example_collate_fn

    def _sentence_drop_on_example(self, example):
        new_ex = deepcopy(example)
        for sentence in new_ex.tokenized_sentences:
            if not self.sent_keep_fn(sentence) and random.random() < self.sent_drop_prob:
                sentence.marked_for_deletion = True

        # perform dataset-specific postprocessing to propagate the effect of sentence removal if necessary
        new_ex = self.sent_drop_postproc(new_ex)

        new_ex.tokenized_sentences = list(filter(lambda sentence: not sentence.marked_for_deletion, new_ex.tokenized_sentences))

        return new_ex

    def _collate_fn(self, examples):
        if self.sent_drop_prob > 0:
            # perform sentence drop
            examples = [self._sentence_drop_on_example(ex) for ex in examples]
        
        return self.example_collate_fn(examples)

if __name__ == "__main__":
    d = DocREDDataset("/Users/peng.qi/Downloads/dev.json")
    dl = SentenceDropDataLoader(d, batch_size=2, sent_drop_prob=.1, 
        sent_drop_postproc=docred_sent_drop_postproc, example_collate_fn=docred_collate_fn)
    for batch in dl:
        pass