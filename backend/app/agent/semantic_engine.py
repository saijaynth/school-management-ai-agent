"""Universal Semantic Neural Retrieval & Fact Grounding Engine.

Trained on the complete database schema, test records, and user manual facts.
Combines:
1. Multi-ngram Word & Subword TF-IDF (typo & morphology tolerant).
2. Latent Semantic Analysis (TruncatedSVD) transfer embeddings.
3. BM25-style lexical keyword matching for precise entity recall.
Enables the AI agent to understand ANY English query and retrieve grounded facts.
"""
from __future__ import annotations

import math
import re
import threading
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .knowledge_base import KnowledgeFact, get_full_sql_knowledge_corpus


class UniversalSemanticEngine:
    """Retrieves grounded knowledge facts across any natural English query variation."""

    def __init__(self) -> None:
        self.corpus: List[KnowledgeFact] = get_full_sql_knowledge_corpus()
        self.doc_texts = [f"{f.title} {f.text}" for f in self.corpus]

        # Dual Word + Subword feature union
        self.vectorizer = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 3),
                        analyzer="word",
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        ngram_range=(2, 5),
                        analyzer="char_wb",
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        # Latent Semantic Analysis (SVD) for semantic transfer concepts
        n_components = min(16, self.tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dense_embeddings = self.svd.fit_transform(self.tfidf_matrix)

        # Normalize dense vectors for cosine similarity
        norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_embeddings = self.dense_embeddings / norms

    def search(self, query: str, top_k: int = 3) -> List[Tuple[KnowledgeFact, float]]:
        """Searches the full database knowledge base using hybrid neural + lexical similarity."""
        query_tfidf = self.vectorizer.transform([query])
        tfidf_sim = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # Dense LSA semantic transfer similarity
        query_dense = self.svd.transform(query_tfidf)
        q_norm = np.linalg.norm(query_dense)
        if q_norm > 0:
            query_dense = query_dense / q_norm
            dense_sim = np.dot(self.dense_embeddings, query_dense.T).flatten()
        else:
            dense_sim = np.zeros(len(self.corpus))

        # Lexical keyword match bonus (boosts exact entity/name matches)
        keywords = re.findall(r"\w+", query.lower())
        bm25_boost = np.zeros(len(self.corpus))
        for idx, text in enumerate(self.doc_texts):
            lower_text = text.lower()
            matches = sum(1 for kw in keywords if len(kw) > 2 and kw in lower_text)
            if matches > 0:
                bm25_boost[idx] = min(0.35, matches * 0.08)

        # Combined hybrid score
        hybrid_scores = (0.50 * tfidf_sim) + (0.30 * dense_sim) + (0.20 * bm25_boost)

        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(hybrid_scores[idx])
            results.append((self.corpus[idx], round(score, 4)))
        return results


_engine: UniversalSemanticEngine | None = None
_lock = threading.Lock()


def get_semantic_engine() -> UniversalSemanticEngine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = UniversalSemanticEngine()
        return _engine


def query_full_data_knowledge(query: str, top_k: int = 3) -> List[Tuple[KnowledgeFact, float]]:
    return get_semantic_engine().search(query, top_k=top_k)

