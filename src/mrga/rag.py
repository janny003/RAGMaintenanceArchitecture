from typing import List
from .models import RetrievalDoc


class KeywordRetriever:
    """Deterministic fallback retriever.
    Replace this with Chroma vector retrieval adapter later.
    """

    def __init__(self, corpus: List[RetrievalDoc]):
        self.corpus = corpus

    def search(self, query: str, top_k: int = 5) -> List[RetrievalDoc]:
        q = query.lower()
        scored = []
        for d in self.corpus:
            score = sum(1 for tok in q.split() if tok and tok in d.content.lower())
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]
