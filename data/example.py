from dataclasses import dataclass, field
from typing import List

from .sentence import Sentence

@dataclass
class ExampleWithSentences:
    tokenized_sentences: List[Sentence]