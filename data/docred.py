from collections import defaultdict
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Iterable

from dataset import TokenizedDataset, SentenceDropDataset
from sentence import Sentence
from example import ExampleWithSentences

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
    mentions: List[DocREDMention]

@dataclass
class DocREDRelation:
    head_entity: DocREDEntity
    tail_entity: DocREDEntity
    relation: int
    evidence: List[Sentence]

@dataclass
class DocREDExample(ExampleWithSentences):
    head_entity: DocREDEntity
    entities: List[DocREDEntity]
    relations: List[DocREDRelation]

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

    example.relations = list(filter(lambda r: len(r.head_entity.mentions) > 0 and len(r.tail_entity.mentions) > 0 and not any(s.marked_for_deletion for s in r.evidence), example.relations))
    example.entities = list(filter(lambda e: len(e.mentions) > 0, example.entities))

    return example

def docred_collate_fn(examples: Iterable[DocREDExample], ner_vocab_size: int, relation_vocab_size: int, tokenizer: AutoTokenizer):
    # filter out examples where the head entity is no longer available
    examples = list(filter(lambda ex: ex.head_entity in ex.entities, examples))

    if len(examples) == 0:
        return

    # use the first mention of the head entity to guide the input
    head_mentions = []
    for ex in examples:
        mention = ex.head_entity.mentions[0]
        mention_text = mention.sentence.token_ids[mention.start:mention.end]
        head_mentions.append(mention_text)

    context_lens = [sum(len(s.token_ids) for s in ex.tokenized_sentences) + len(m) + 3 for ex, m in zip(examples, head_mentions)]
    max_ctx_len = max(context_lens)
    entity_counts = [len(ex.entities) for ex in examples]
    max_ent_count = max(entity_counts)
    max_sentences = max(len(ex.tokenized_sentences) for ex in examples)

    context = np.zeros((len(examples), max_ctx_len), dtype=np.int64)
    attention_mask = np.zeros((len(examples), max_ctx_len), dtype=np.uint8)
    entity_mask = np.zeros((len(examples), max_ent_count, max_ctx_len), dtype=np.uint8)
    sentence_start = np.full((len(examples), max_sentences), -1, dtype=np.int64)
    sentence_end = np.full((len(examples), max_sentences), -1, dtype=np.int64)

    pairs = []

    CLS = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    SEP = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    for ex_i, (ex, head_mention) in enumerate(zip(examples, head_mentions)):
        context[ex_i, 0] = CLS
        context[ex_i, 1:1+len(head_mention)] = head_mention
        context[ex_i, len(head_mention)+1] = SEP
        offset = len(head_mention) + 2

        # assign each entity an index for quick reference later
        for ent_i, ent in enumerate(ex.entities):
            ent.idx = ent_i
            
        for s_i, sentence in enumerate(ex.tokenized_sentences):
            sentence_start[ex_i, s_i] = offset
            sentence_end[ex_i, s_i] = offset + len(sentence.token_ids) - 1
            context[ex_i, offset:offset+len(sentence.token_ids)] = sentence.token_ids
            for mention in sentence.mentions:
                parent_idx = mention.parent.idx
                entity_mask[ex_i, ent_i, offset+mention.start:offset+mention.end] = 1

            offset += len(sentence.token_ids)

        context[ex_i, offset] = SEP
        attention_mask[ex_i, :offset+1] = 1
    
        ex_pairs = defaultdict(list)
        for r in ex.relations:
            ex_pairs[(r.head_entity.idx, r.tail_entity.idx)].append((r.relation, r.evidence))

        pairs.append(ex_pairs)

    max_pairs = max(len(p) for p in pairs)

    entity_pairs = np.full((len(examples), max_pairs, 2), -1, dtype=np.int64)
    pair_labels = np.zeros((len(examples), max_pairs, relation_vocab_size), dtype=np.int64)
    pair_evidence = np.zeros((len(examples), max_pairs, relation_vocab_size, max_sentences), dtype=np.uint8)

    for ex_i, ex in enumerate(examples):
        for pair_i, pair in enumerate(pairs[ex_i]):
            entity_pairs[ex_i, pair_i] = pair
            for r, e in pairs[ex_i][pair]:
                pair_labels[ex_i, pair_i, r] = 1
                for sentence in e:
                    pair_evidence[ex_i, pair_i, r, sentence.sentence_idx] = 1

    retval = {
        'context': context,
        'attention_mask': attention_mask,
        'entity_mask': entity_mask,
        'sentence_start': sentence_start,
        'sentence_end': sentence_end,
        'entity_pairs': entity_pairs, 
        'pair_labels': pair_labels,
        'pair_evidence': pair_evidence
    }

    retval = {k: torch.from_numpy(retval[k]) for k in retval}

    return retval

if __name__ == "__main__":
    dataset = DocREDDataset("/Users/peng.qi/Downloads/dev.json")
    sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1, 
        sent_drop_postproc=docred_sent_drop_postproc, 
        example_validate_fn=lambda ex: ex.head_entity in ex.entities)

    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataloader = DataLoader(sdrop_dataset, batch_size=2, 
        collate_fn=lambda examples: docred_collate_fn(examples, ner_vocab_size=len(dataset.ner_to_idx), relation_vocab_size=len(dataset.relation_to_idx), tokenizer=dataset.tokenizer))
    for batch in tqdm(dataloader):
        pass