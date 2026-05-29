from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class RetrievalDoc:
    source: str
    content: str


@dataclass
class CaseInput:
    query: str


@dataclass
class CaseOutput:
    risk: str
    cause: str
    recommended_actions: List[str] = field(default_factory=list)
    evidence: List[RetrievalDoc] = field(default_factory=list)
    confidence: float = 0.0
    approved: bool = False
    feedback_note: str = ""
    memory_updates: Dict[str, str] = field(default_factory=dict)
