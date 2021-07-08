from dataclasses import dataclass
from typing import List

from sentence import Sentence

@dataclass
class DocREDMention:
    sentence: Sentence
    start: int
    end: int
    ner_type: str

@dataclass
class DocREDSentence(Sentence):
    mentions: List[DocREDMention]

@dataclass
class DocREDEntity:
    mentions: List[DocREDMention]
    
@dataclass
class DocREDRelation:
    head_entity: DocREDEntity
    tail_entity: DocREDEntity
    relation: str
    evidence: List[Sentence]

@dataclass
class DocREDExample:
    tokenized_sentences: List[Sentence]
    head_entity: DocREDEntity
    entities: List[DocREDEntity]
    relations: List[DocREDRelation]