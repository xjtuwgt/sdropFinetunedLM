from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Iterable, List

from .dataset import TokenizedDataset, SentenceDropDataset
from .sentence import Sentence
from .example import ExampleWithSentences

@dataclass
class DocREDMention:
    sentence: Sentence
    start: int
    end: int
    ner_type: int
    marked_for_deletion: bool = False

@dataclass
class DocREDSentence(Sentence):
    mentions: List[DocREDMention] = field(default_factory=list)

@dataclass
class DocREDEntity:
    name: List[int]
    mentions: List[DocREDMention]

@dataclass
class DocREDRelation:
    head_entity: DocREDEntity
    tail_entity: DocREDEntity
    relation: int
    evidence: List[DocREDSentence]

@dataclass
class DocREDExample(ExampleWithSentences):
    doc_title: str
    head_entity: DocREDEntity
    entities: List[DocREDEntity]
    relations: List[DocREDRelation]

class DocREDDataset(TokenizedDataset):
    def __init__(self, json_file, tokenizer_class="bert-base-uncased", eval=False, ner_to_idx=None, relation_to_idx=None):
        super().__init__(tokenizer_class=tokenizer_class)
        with open(json_file) as f:
            data = json.load(f)
        self.eval = eval
        
        self.data = []
        self.relation_to_idx = {}
        self.idx_to_relation = []
        if relation_to_idx is not None:
            self.relation_to_idx = relation_to_idx
            self.idx_to_relation = [None] * len(relation_to_idx)
            for k in relation_to_idx:
                self.idx_to_relation[relation_to_idx[k]] = k
        else:
            rel_count = Counter([y['r'] for x in data for y in x['labels']])
            self.idx_to_relation = list(rel_count.keys())
            for i, r in enumerate(self.idx_to_relation):
                self.relation_to_idx[r] = i

        self.ner_to_idx = {"PAD": 0}
        self.idx_to_ner = ["PAD"]
        if ner_to_idx is not None:
            self.ner_to_idx = ner_to_idx
            self.idx_to_ner = [None] * len(ner_to_idx)
            for k in ner_to_idx:
                self.idx_to_ner[ner_to_idx[k]] = k
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
                e1 = DocREDEntity(name=self.tokenizer(entity[0]['name'], add_special_tokens=False)['input_ids'], mentions=mentions)
                # add backpointers from mentions to entities for convenience
                for m1 in mentions:
                    m1.parent = e1
                entities.append(e1)

            head_to_examples = defaultdict(list)
            if not eval:
                for r in datum['labels']:
                    head, tail, relation, evidence = r['h'], r['t'], r['r'], r['evidence']
                    if relation not in self.relation_to_idx:
                        continue
                    relation = self.relation_to_idx[relation]
                    head_to_examples[head].append(DocREDRelation(head_entity=entities[head], tail_entity=entities[tail], relation=relation, evidence=[sentences[si] for si in evidence]))

            for head in range(len(entities)):
                if head not in head_to_examples:
                    head_to_examples[head].extend([DocREDRelation(head_entity=entities[head], tail_entity=entities[tail], relation=-1, evidence=[]) for tail in range(len(entities)) if tail != head])

            for head in head_to_examples:
                example = DocREDExample(sentences, datum['title'], entities[head], entities, head_to_examples[head])
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

    example.relations = list(filter(lambda r: len(r.head_entity.mentions) > 0 and len(r.tail_entity.mentions) > 0 and not any(s.marked_for_deletion for s in r.evidence), example.relations))
    example.entities = list(filter(lambda e: len(e.mentions) > 0, example.entities))

    return example

def docred_collate_fn(examples: Iterable[DocREDExample], dataset: DocREDDataset):
    # filter out examples where the head entity is no longer available
    examples = list(filter(lambda ex: ex.head_entity in ex.entities, examples))

    if len(examples) == 0:
        return

    context_lens = [sum(len(s.token_ids) for s in ex.tokenized_sentences) + len(ex.head_entity.name) + 3 for ex in examples]
    max_ctx_len = max(context_lens)
    entity_counts = [len(ex.entities) for ex in examples]
    max_ent_count = max(entity_counts)
    max_mention_count = max([len(entity.mentions) for ex in examples for entity in ex.entities])
    max_sentences = max(len(ex.tokenized_sentences) for ex in examples)

    context = np.zeros((len(examples), max_ctx_len), dtype=np.int64)
    ner = np.zeros((len(examples), max_ctx_len), dtype=np.int64)
    attention_mask = np.zeros((len(examples), max_ctx_len), dtype=np.uint8)
    entity_start = np.full((len(examples), max_ent_count, max_mention_count), -1, dtype=np.int64)
    sentence_mask = np.zeros((len(examples), max_sentences, max_ctx_len), dtype=np.uint8)

    pairs = []

    tokenizer = dataset.tokenizer
    CLS = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    SEP = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    for ex_i, ex in enumerate(examples):
        context[ex_i, 0] = CLS
        head_mention = ex.head_entity.name
        context[ex_i, 1:1+len(head_mention)] = head_mention
        context[ex_i, len(head_mention)+1] = SEP
        offset = len(head_mention) + 2

        # assign each entity an index for quick reference later
        for ent_i, ent in enumerate(ex.entities):
            ent.idx = ent_i
            
        for s_i, sentence in enumerate(ex.tokenized_sentences):
            ex.tokenized_sentences[s_i].sentence_idx = s_i
            sentence_mask[ex_i, s_i, offset:offset+len(sentence.token_ids)] = 1
            # sentence_start_end[ex_i, s_i, 0] = offset
            # sentence_start_end[ex_i, s_i, 1] = offset + len(sentence.token_ids) - 1
            context[ex_i, offset:offset+len(sentence.token_ids)] = sentence.token_ids
            for mention in sentence.mentions:
                parent_idx = mention.parent.idx
                m_i = 0
                while entity_start[ex_i, parent_idx, m_i] >= 0:
                    m_i += 1
                entity_start[ex_i, parent_idx, m_i] = offset + mention.start
                ner[ex_i, offset+mention.start:offset+mention.end] = mention.ner_type

            offset += len(sentence.token_ids)

        context[ex_i, offset] = SEP
        attention_mask[ex_i, :offset+1] = 1
    
        ex_pairs = defaultdict(list)
        positive_entities = set()

        for r in ex.relations:
            ex_pairs[(r.head_entity.idx, r.tail_entity.idx)].append((r.relation, r.evidence))
            positive_entities.add(r.tail_entity.idx)

        # add negative entity pairs
        for ent in ex.entities:
            if ent.idx not in positive_entities and ent.idx != ex.head_entity.idx:
                ex_pairs[(ex.head_entity.idx, ent.idx)].append((-1, []))

        pairs.append(ex_pairs)

    max_pairs = max(len(p) for p in pairs)

    entity_pairs = np.full((len(examples), max_pairs, 2), -1, dtype=np.int64)
    pair_labels = np.zeros((len(examples), max_pairs, len(dataset.relation_to_idx)), dtype=np.int64)
    pair_evidence = np.zeros((len(examples), max_pairs, len(dataset.relation_to_idx), max_sentences), dtype=np.uint8)

    for ex_i, ex in enumerate(examples):
        for pair_i, pair in enumerate(pairs[ex_i]):
            entity_pairs[ex_i, pair_i] = pair
            for r, e in pairs[ex_i][pair]:
                if r >= 0:
                    pair_labels[ex_i, pair_i, r] = 1
                    for sentence in e:
                        pair_evidence[ex_i, pair_i, r, sentence.sentence_idx] = 1

    retval = {
        'context': context,
        'attention_mask': attention_mask,
        'entity_start': entity_start,
        # 'sentence_start_end': sentence_start_end,
        'sentence_mask': sentence_mask,
        'entity_pairs': entity_pairs, 
        'pair_labels': pair_labels,
        'pair_evidence': pair_evidence,
        'ner': ner,
    }

    retval = {k: torch.from_numpy(retval[k]) for k in retval}

    retval['doc_title'] = [ex.doc_title for ex in examples]

    return retval

def docred_validate_fn(example):
    return example.head_entity in example.entities

if __name__ == "__main__":
    dataset = DocREDDataset("dataset/docred/dev.json")
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1, 
        sent_drop_postproc=docred_sent_drop_postproc, 
        example_validate_fn=lambda ex: ex.head_entity in ex.entities)

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataloader = DataLoader(sdrop_dataset, batch_size=2, 
        collate_fn=lambda examples: docred_collate_fn(examples, dataset=dataset))
    for batch in tqdm(dataloader):
        pass