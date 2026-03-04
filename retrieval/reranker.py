"""Cross-Encoder Reranker for improving retrieval quality."""

import os
from typing import List, Dict, Optional
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model for better relevance."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enabled: bool = True):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            enabled: Whether reranking is enabled
        """
        self.model_name = model_name
        self._model = None

        # Auto-disable on memory-constrained environments (Render free tier, etc.)
        # Set DISABLE_RERANKER=true as an env var on Render to save ~100 MB RAM.
        if os.environ.get("DISABLE_RERANKER", "").lower() in ("1", "true", "yes"):
            logger.info("Reranker disabled via DISABLE_RERANKER env var (memory saving mode)")
            self.enabled = False
        else:
            self.enabled = enabled
            # NOTE: Model is NOT loaded at init — it is lazy-loaded on first rerank() call.
            # This saves ~100 MB at startup on constrained deployments.

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("✓ Cross-encoder reranker loaded")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder model: {e}. Reranking disabled.")
            self.enabled = False

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank retrieved chunks using cross-encoder scores.

        Args:
            query: User query
            chunks: List of retrieved chunk dicts (must have 'text' key)
            top_k: Number of results to return (None = return all reranked)

        Returns:
            Reranked list of chunks with 'rerank_score' added
        """
        # Lazy-load model on first actual rerank call
        if self.enabled and self._model is None:
            self._load_model()

        if not self.enabled or not chunks or self._model is None:
            return chunks

        try:
            # Build (query, passage) pairs for cross-encoder
            pairs = [(query, chunk["text"]) for chunk in chunks]

            # Get cross-encoder scores
            scores = self._model.predict(pairs)

            # Attach rerank_score to each chunk
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            # Sort by rerank score descending
            reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

            if top_k:
                reranked = reranked[:top_k]

            logger.info(
                f"Reranked {len(chunks)} chunks → top {len(reranked)} | "
                f"Top score: {reranked[0]['rerank_score']:.3f} | "
                f"Bottom score: {reranked[-1]['rerank_score']:.3f}"
            )
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks  # Fall back to original order

    def get_relevance_scores(self, query: str, chunks: List[Dict]) -> List[float]:
        """Get raw relevance scores without reordering.

        Args:
            query: User query
            chunks: Retrieved chunks

        Returns:
            List of relevance scores (same order as input)
        """
        if not self.enabled or not chunks or self._model is None:
            return [0.5] * len(chunks)

        try:
            pairs = [(query, chunk["text"]) for chunk in chunks]
            scores = self._model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Score computation failed: {e}")
            return [0.5] * len(chunks)
