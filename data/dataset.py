from collections import defaultdict
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from example import DocREDExample, DocREDMention, DocREDEntity, DocREDRelation, DocREDSentence

class TokenizedDataset(Dataset):
    def __init__(self, tokenizer_class="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_class)

class DocREDDataset(TokenizedDataset):
    def __init__(self, json_file, tokenizer_class="bert-base-uncased", eval=False):
        super().__init__(tokenizer_class=tokenizer_class)
        with open(json_file) as f:
            data = json.load(f)
        self.eval = eval
        
        self.data = []
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
                
            sentences = [DocREDSentence(sentence_idx=sent_id, token_ids=sent, mentions=[]) for sent_id, sent in enumerate(tokenized_sentences)]
            
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
                head_to_examples[head].append(DocREDRelation(head_entity=entities[head], tail_entity=entities[tail], relation=relation, evidence=[sentences[si] for si in evidence]))

            for head in head_to_examples:
                example = DocREDExample(sentences, entities[head], entities, head_to_examples[head])
                self.data.append(example)

if __name__ == "__main__":
    d = DocREDDataset("/Users/peng.qi/Downloads/dev.json")