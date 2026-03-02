"""Hybrid search combining vector similarity and BM25."""

from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np

from .vector_store import VectorStore
from utils.logger import setup_logger


logger = setup_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining vector search and BM25."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store instance
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # BM25 index (will be built on demand)
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        
        logger.info("Initialized hybrid retriever")
    
    def _build_bm25_index(self):
        """Build BM25 index from vector store."""
        # Get all documents from vector store
        all_results = self.vector_store.collection.get()
        
        if not all_results['documents']:
            logger.warning("No documents found in vector store for BM25 indexing")
            return
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in all_results['documents']]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = all_results['documents']
        self.bm25_metadata = [
            {
                "chunk_id": all_results['ids'][i],
                "metadata": all_results['metadatas'][i]
            }
            for i in range(len(all_results['ids']))
        ]
        
        logger.info(f"Built BM25 index with {len(self.bm25_documents)} documents")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search using hybrid retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search
            filter_dict: Metadata filters
            
        Returns:
            List of search results
        """
        if not use_hybrid:
            # Use only vector search
            return self.vector_store.search(query, top_k, filter_dict)
        
        # Build BM25 index if not exists
        if self.bm25_index is None:
            self._build_bm25_index()
        
        if self.bm25_index is None:
            # Fallback to vector search
            logger.warning("BM25 index not available, using vector search only")
            return self.vector_store.search(query, top_k, filter_dict)
        
        # Vector search
        vector_results = self.vector_store.search(query, top_k * 2, filter_dict)
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Combine results
        combined_scores = {}
        
        # Add vector results
        vector_distances = [r['distance'] for r in vector_results]
        normalized_vector_scores = self._normalize_scores(
            [1 - d for d in vector_distances]  # Convert distance to similarity
        )
        
        for i, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {
                'score': normalized_vector_scores[i] * self.vector_weight,
                'text': result['text'],
                'metadata': result['metadata'],
                'vector_score': normalized_vector_scores[i],
                'bm25_score': 0.0
            }
        
        # Add BM25 results
        bm25_top_scores = [bm25_scores[i] for i in top_bm25_indices]
        normalized_bm25_scores = self._normalize_scores(bm25_top_scores)
        
        for i, idx in enumerate(top_bm25_indices):
            chunk_id = self.bm25_metadata[idx]['chunk_id']
            
            if chunk_id in combined_scores:
                # Update existing entry
                combined_scores[chunk_id]['score'] += normalized_bm25_scores[i] * self.bm25_weight
                combined_scores[chunk_id]['bm25_score'] = normalized_bm25_scores[i]
            else:
                # Add new entry
                combined_scores[chunk_id] = {
                    'score': normalized_bm25_scores[i] * self.bm25_weight,
                    'text': self.bm25_documents[idx],
                    'metadata': self.bm25_metadata[idx]['metadata'],
                    'vector_score': 0.0,
                    'bm25_score': normalized_bm25_scores[i]
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]
        
        # Format results
        final_results = []
        for chunk_id, data in sorted_results:
            result = {
                'chunk_id': chunk_id,
                'text': data['text'],
                'metadata': data['metadata'],
                'combined_score': data['score'],
                'vector_score': data['vector_score'],
                'bm25_score': data['bm25_score']
            }
            final_results.append(result)
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results
    
    def rebuild_index(self):
        """Rebuild BM25 index."""
        self.bm25_index = None
        self._build_bm25_index()
