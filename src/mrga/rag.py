from __future__ import annotations

from pathlib import Path
from typing import List

from .models import RetrievalDoc


class KeywordRetriever:
    """Deterministic fallback retriever."""

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


class TFIDFVectorDBRetriever:
    """TRD vector_db adapter (sqlite + sklearn tf-idf matrix)."""

    def __init__(self, vector_db_dir: str | Path):
        self.vector_db_dir = Path(vector_db_dir)
        self.db_path = self.vector_db_dir / "trd_chunks.sqlite"
        self.vectorizer_path = self.vector_db_dir / "tfidf_vectorizer.joblib"
        self.matrix_path = self.vector_db_dir / "tfidf_matrix.npz"
        self._ready = False
        self._init_error = ""
        self._vectorizer = None
        self._matrix = None
        self._sqlite3 = None
        self._normalize = None
        self._cosine_similarity = None
        self._load_npz = None
        self._bootstrap()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def init_error(self) -> str:
        return self._init_error

    def _bootstrap(self) -> None:
        required = [self.db_path, self.vectorizer_path, self.matrix_path]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            self._init_error = f"missing vector_db files: {', '.join(missing)}"
            return
        try:
            import sqlite3
            import joblib
            from scipy import sparse
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import normalize

            self._sqlite3 = sqlite3
            self._normalize = normalize
            self._cosine_similarity = cosine_similarity
            self._load_npz = sparse.load_npz
            self._vectorizer = joblib.load(self.vectorizer_path)
            self._matrix = self._load_npz(self.matrix_path)
            self._ready = True
        except Exception as e:
            self._init_error = str(e)
            self._ready = False

    def _load_chunk(self, rowid: int) -> dict:
        conn = self._sqlite3.connect(self.db_path)
        conn.row_factory = self._sqlite3.Row
        try:
            row = conn.execute(
                "SELECT rowid, chunk_id, source_path, source_name, chunk_index, text FROM chunks WHERE rowid = ?",
                (rowid,),
            ).fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()

    def search(self, query: str, top_k: int = 5) -> List[RetrievalDoc]:
        if not self._ready:
            return []
        qv = self._normalize(self._vectorizer.transform([query]), norm="l2", copy=False)
        scores = self._cosine_similarity(qv, self._matrix).ravel()
        top_indices = scores.argsort()[::-1][:top_k]
        docs: List[RetrievalDoc] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            row = self._load_chunk(int(idx) + 1)
            if not row:
                continue
            content = row.get("text", "")
            source = row.get("source_name", "TRD")
            docs.append(RetrievalDoc(source=source, content=content))
        return docs


class HybridRetriever:
    """Use vector_db first; fallback to keyword corpus if unavailable/empty."""

    def __init__(self, vector_db_dir: str | Path, fallback: KeywordRetriever):
        self.vector = TFIDFVectorDBRetriever(vector_db_dir)
        self.fallback = fallback

    @property
    def uses_vector_db(self) -> bool:
        return self.vector.ready

    @property
    def vector_db_error(self) -> str:
        return self.vector.init_error

    def search(self, query: str, top_k: int = 5) -> List[RetrievalDoc]:
        docs = self.vector.search(query, top_k=top_k)
        if docs:
            return docs
        return self.fallback.search(query, top_k=top_k)
