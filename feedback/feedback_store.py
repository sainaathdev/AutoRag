"""Feedback storage and management."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import pickle

from utils.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class QueryFeedback:
    """Feedback for a single query."""
    query_id: str
    query: str
    rewritten_query: Optional[str]
    retrieved_chunks: List[Dict]
    answer: str
    confidence_score: float
    evaluation: Dict
    timestamp: str
    user_feedback: Optional[str] = None
    is_failure: bool = False
    failure_reason: Optional[str] = None
    retrieval_method: str = "vector"


class FeedbackStore:
    """Store and manage query feedback."""
    
    def __init__(self, store_path: str = "./data/feedback"):
        """Initialize feedback store.
        
        Args:
            store_path: Path to feedback storage directory
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.store_path / "feedback.jsonl"
        self.failure_memory_file = self.store_path / "failure_memory.pkl"
        
        # Load failure memory
        self.failure_memory: List[QueryFeedback] = []
        self._load_failure_memory()
        
        logger.info(f"Initialized FeedbackStore at {store_path}")
    
    def _load_failure_memory(self):
        """Load failure memory from disk."""
        if self.failure_memory_file.exists():
            try:
                with open(self.failure_memory_file, 'rb') as f:
                    self.failure_memory = pickle.load(f)
                logger.info(f"Loaded {len(self.failure_memory)} failure cases")
            except Exception as e:
                logger.error(f"Failed to load failure memory: {e}")
                self.failure_memory = []
    
    def _save_failure_memory(self):
        """Save failure memory to disk."""
        try:
            with open(self.failure_memory_file, 'wb') as f:
                pickle.dump(self.failure_memory, f)
        except Exception as e:
            logger.error(f"Failed to save failure memory: {e}")
    
    def add_feedback(
        self,
        query_id: str,
        query: str,
        retrieved_chunks: List[Dict],
        answer: str,
        confidence_score: float,
        evaluation: Dict,
        rewritten_query: Optional[str] = None,
        user_feedback: Optional[str] = None,
        retrieval_method: str = "vector"
    ):
        """Add feedback for a query.
        
        Args:
            query_id: Unique query identifier
            query: Original query
            retrieved_chunks: Retrieved document chunks
            answer: Generated answer
            confidence_score: Confidence score
            evaluation: Evaluation results
            rewritten_query: Rewritten query (if any)
            user_feedback: User feedback (if any)
            retrieval_method: Retrieval method used
        """
        # Determine if this is a failure
        is_failure = (
            confidence_score < 0.6 or
            evaluation.get("hallucination_detected", False) or
            (user_feedback and user_feedback.lower() in ["bad", "wrong", "incorrect"])
        )
        
        failure_reason = None
        if is_failure:
            failure_reason = evaluation.get("failure_reason") or "Low confidence"
        
        # Create feedback object
        feedback = QueryFeedback(
            query_id=query_id,
            query=query,
            rewritten_query=rewritten_query,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            confidence_score=confidence_score,
            evaluation=evaluation,
            timestamp=datetime.now().isoformat(),
            user_feedback=user_feedback,
            is_failure=is_failure,
            failure_reason=failure_reason,
            retrieval_method=retrieval_method
        )
        
        # Save to JSONL file
        try:
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(feedback)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
        
        # Add to failure memory if it's a failure
        if is_failure:
            self.failure_memory.append(feedback)
            
            # Limit failure memory size
            max_size = 1000
            if len(self.failure_memory) > max_size:
                self.failure_memory = self.failure_memory[-max_size:]
            
            self._save_failure_memory()
            logger.warning(f"Added failure case: {query} (confidence={confidence_score:.2f})")
    
    def get_recent_feedback(self, n: int = 100) -> List[QueryFeedback]:
        """Get recent feedback entries.
        
        Args:
            n: Number of entries to return
            
        Returns:
            List of feedback entries
        """
        feedback_list = []
        
        if not self.feedback_file.exists():
            return feedback_list
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Get last n lines
            for line in lines[-n:]:
                data = json.loads(line)
                feedback_list.append(QueryFeedback(**data))
        
        except Exception as e:
            logger.error(f"Failed to read feedback: {e}")
        
        return feedback_list
    
    def get_failure_cases(self, n: Optional[int] = None) -> List[QueryFeedback]:
        """Get failure cases from memory.
        
        Args:
            n: Number of cases to return (None for all)
            
        Returns:
            List of failure cases
        """
        if n is None:
            return self.failure_memory
        return self.failure_memory[-n:]
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics.
        
        Returns:
            Statistics dictionary
        """
        recent = self.get_recent_feedback(n=100)
        
        if not recent:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "failure_rate": 0.0,
                "total_failures": len(self.failure_memory)
            }
        
        confidences = [f.confidence_score for f in recent]
        failures = [f for f in recent if f.is_failure]
        
        return {
            "total_queries": len(recent),
            "avg_confidence": sum(confidences) / len(confidences),
            "failure_rate": len(failures) / len(recent),
            "total_failures": len(self.failure_memory),
            "recent_failures": len(failures)
        }
    
    def clear_old_feedback(self, keep_days: int = 30):
        """Clear feedback older than specified days.
        
        Args:
            keep_days: Number of days to keep
        """
        # This is a simple implementation
        # In production, you'd want to filter by timestamp
        logger.info(f"Feedback cleanup not implemented (keep_days={keep_days})")
