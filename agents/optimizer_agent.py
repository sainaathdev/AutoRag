"""Optimizer agent for continuous system improvement."""

from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

from .llm_client import DeepSeekClient
from utils.logger import setup_logger


logger = setup_logger(__name__)


OPTIMIZER_PROMPT = """You are a Self-Improving RAG Optimization Agent.

Your goal is to continuously improve the retrieval and answer quality of the RAG system.

You have access to:
- Query history
- Retrieved document chunks
- Generated answers
- Confidence scores
- User feedback
- Retrieval statistics

Given a failure case (low confidence answer), diagnose the root cause:

1. **Poor Chunking**: Document chunks are too large/small or poorly segmented
2. **Insufficient Context**: Not enough relevant chunks retrieved
3. **Wrong Retrieval Method**: Vector search vs hybrid search vs keyword search
4. **Ambiguous Query**: Query needs rewriting or clarification
5. **Embedding Drift**: Embeddings don't capture semantic meaning well
6. **Source Quality**: The source document itself is unclear or incomplete

Provide diagnosis and recommendations in JSON format:
- failure_type: one of the above categories
- confidence: 0-1 confidence in diagnosis
- recommended_actions: list of specific actions to take
- rechunk_needed: true/false
- new_chunk_size: suggested chunk size if rechunking
- new_overlap: suggested overlap if rechunking
- switch_to_hybrid: true/false
- reasoning: detailed explanation"""


class OptimizerAgent:
    """Agent for optimizing RAG system performance."""
    
    def __init__(
        self,
        llm_client: DeepSeekClient,
        enabled: bool = True,
        auto_optimize: bool = True
    ):
        """Initialize optimizer agent.
        
        Args:
            llm_client: DeepSeek client instance
            enabled: Whether optimization is enabled
            auto_optimize: Whether to auto-apply optimizations
        """
        self.llm_client = llm_client
        self.enabled = enabled
        self.auto_optimize = auto_optimize
        
        # Performance tracking
        self.retrieval_stats = defaultdict(lambda: {"queries": 0, "avg_confidence": 0.0, "scores": []})
        self.document_stats = defaultdict(lambda: {"queries": 0, "failures": 0, "avg_confidence": 0.0})
        
        logger.info(f"Initialized OptimizerAgent (enabled={enabled}, auto_optimize={auto_optimize})")
    
    def diagnose_failure(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        answer: str,
        evaluation: Dict
    ) -> Dict:
        """Diagnose why an answer failed.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            answer: Generated answer
            evaluation: Answer evaluation results
            
        Returns:
            Diagnosis with recommended actions
        """
        if not self.enabled:
            return {
                "failure_type": "unknown",
                "confidence": 0.0,
                "recommended_actions": [],
                "rechunk_needed": False,
                "reasoning": "Optimizer disabled"
            }
        
        try:
            # Prepare context
            chunks_str = "\n\n".join([
                f"[Chunk {i+1}] (from {chunk.get('metadata', {}).get('document_id', 'unknown')})\n{chunk.get('text', '')}"
                for i, chunk in enumerate(retrieved_chunks)
            ])
            
            eval_str = f"""Confidence: {evaluation.get('confidence_score', 0)}
Hallucination: {evaluation.get('hallucination_detected', False)}
Completeness: {evaluation.get('completeness_score', 0)}
Failure Reason: {evaluation.get('failure_reason', 'N/A')}"""
            
            user_message = f"""Query: {query}

Retrieved Chunks:
{chunks_str}

Generated Answer:
{answer}

Evaluation Results:
{eval_str}

Diagnose the failure and recommend improvements."""
            
            messages = [
                {"role": "system", "content": OPTIMIZER_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            result = self.llm_client.chat_json(messages, temperature=0.3)
            
            logger.info(f"Diagnosed failure: {result.get('failure_type', 'unknown')}")
            
            return result
        
        except Exception as e:
            logger.error(f"Failure diagnosis failed: {e}")
            return {
                "failure_type": "unknown",
                "confidence": 0.0,
                "recommended_actions": [],
                "rechunk_needed": False,
                "reasoning": f"Diagnosis failed: {str(e)}"
            }
    
    def update_retrieval_stats(
        self,
        retrieval_method: str,
        confidence_score: float
    ):
        """Update retrieval method statistics.
        
        Args:
            retrieval_method: Method used ('vector', 'hybrid', 'bm25')
            confidence_score: Confidence score for this query
        """
        stats = self.retrieval_stats[retrieval_method]
        stats["queries"] += 1
        stats["scores"].append(confidence_score)
        
        # Keep only recent scores
        if len(stats["scores"]) > 100:
            stats["scores"] = stats["scores"][-100:]
        
        stats["avg_confidence"] = np.mean(stats["scores"])
    
    def update_document_stats(
        self,
        document_id: str,
        confidence_score: float,
        is_failure: bool = False
    ):
        """Update document performance statistics.
        
        Args:
            document_id: Document identifier
            confidence_score: Confidence score
            is_failure: Whether this was a failure case
        """
        stats = self.document_stats[document_id]
        stats["queries"] += 1
        
        if is_failure:
            stats["failures"] += 1
        
        # Update average confidence
        current_avg = stats["avg_confidence"]
        n = stats["queries"]
        stats["avg_confidence"] = (current_avg * (n - 1) + confidence_score) / n
    
    def get_best_retrieval_method(self) -> str:
        """Get the best performing retrieval method.
        
        Returns:
            Best retrieval method name
        """
        if not self.retrieval_stats:
            return "hybrid"  # Default
        
        best_method = max(
            self.retrieval_stats.items(),
            key=lambda x: x[1]["avg_confidence"]
        )
        
        return best_method[0]
    
    def get_problematic_documents(self, threshold: float = 0.6) -> List[str]:
        """Get list of documents with poor performance.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            List of document IDs
        """
        problematic = []
        
        for doc_id, stats in self.document_stats.items():
            if stats["avg_confidence"] < threshold and stats["queries"] >= 3:
                problematic.append(doc_id)
        
        return problematic
    
    def should_rechunk_document(
        self,
        document_id: str,
        failure_threshold: int = 3
    ) -> bool:
        """Determine if a document should be re-chunked.
        
        Args:
            document_id: Document identifier
            failure_threshold: Number of failures before rechunking
            
        Returns:
            Whether document should be rechunked
        """
        if document_id not in self.document_stats:
            return False
        
        stats = self.document_stats[document_id]
        return stats["failures"] >= failure_threshold
    
    def get_optimization_report(self) -> Dict:
        """Generate optimization report.
        
        Returns:
            Report dictionary
        """
        return {
            "retrieval_methods": dict(self.retrieval_stats),
            "best_retrieval_method": self.get_best_retrieval_method(),
            "problematic_documents": self.get_problematic_documents(),
            "total_queries_tracked": sum(
                stats["queries"] for stats in self.retrieval_stats.values()
            )
        }
