import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from sentence import DocREDSentence
from example import DocREDExample

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
                    m1['pos'] = [st, en]
                    mentions.append(m1)
                entities.append(mentions)

            # TODO: build actual examples from mapped entities
            for entity_i, head in enumerate(datum['vertexSet']):
                # build one example for each head entity
                
                for entity_j, tail in enumerate(datum['vertexSet']):
                    if entity_i == entity_j: continue


if __name__ == "__main__":
    d = DocREDDataset("/Users/peng.qi/Downloads/dev.json")