from dataclasses import dataclass, field
from typing import List

from sentence import Sentence

@dataclass
class ExampleWithSentences:
    tokenized_sentences: List[Sentence]

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