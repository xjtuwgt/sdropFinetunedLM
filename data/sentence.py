class Sentence:
    def __init__(self, example_idx, sentence_idx, token_ids):
        self.example_idx = example_idx
        self.sentence_idx = sentence_idx
        self.token_ids = token_ids

class DocREDSentence(Sentence):
    def __init__(self, example_idx, sentence_idx, token_ids, mentions):
        super().__init__(example_idx, sentence_idx, token_ids)

        self.mentions = mentions