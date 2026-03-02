"""Confidence tracking and analysis."""

from typing import List, Dict
from collections import deque
import numpy as np

from utils.logger import setup_logger


logger = setup_logger(__name__)


class ConfidenceTracker:
    """Track and analyze confidence scores over time."""
    
    def __init__(self, window_size: int = 100):
        """Initialize confidence tracker.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)
        self.all_scores = []
        
        logger.info(f"Initialized ConfidenceTracker (window_size={window_size})")
    
    def add_score(self, score: float):
        """Add a confidence score.
        
        Args:
            score: Confidence score (0-1)
        """
        self.scores.append(score)
        self.all_scores.append(score)
    
    def get_current_average(self) -> float:
        """Get average confidence in current window.
        
        Returns:
            Average confidence score
        """
        if not self.scores:
            return 0.0
        return np.mean(list(self.scores))
    
    def get_overall_average(self) -> float:
        """Get overall average confidence.
        
        Returns:
            Overall average confidence score
        """
        if not self.all_scores:
            return 0.0
        return np.mean(self.all_scores)
    
    def get_trend(self, recent_n: int = 20) -> str:
        """Get confidence trend.
        
        Args:
            recent_n: Number of recent scores to analyze
            
        Returns:
            Trend description ('improving', 'declining', 'stable')
        """
        if len(self.scores) < recent_n:
            return "insufficient_data"
        
        recent_scores = list(self.scores)[-recent_n:]
        first_half = np.mean(recent_scores[:recent_n//2])
        second_half = np.mean(recent_scores[recent_n//2:])
        
        diff = second_half - first_half
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict:
        """Get confidence statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.scores:
            return {
                "current_avg": 0.0,
                "overall_avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "trend": "insufficient_data",
                "total_queries": 0
            }
        
        scores_array = np.array(list(self.scores))
        
        return {
            "current_avg": float(np.mean(scores_array)),
            "overall_avg": self.get_overall_average(),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "std": float(np.std(scores_array)),
            "trend": self.get_trend(),
            "total_queries": len(self.all_scores)
        }
    
    def is_performing_well(self, threshold: float = 0.75) -> bool:
        """Check if system is performing well.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Whether system is performing well
        """
        return self.get_current_average() >= threshold
    
    def needs_optimization(self, threshold: float = 0.6) -> bool:
        """Check if system needs optimization.
        
        Args:
            threshold: Minimum acceptable confidence
            
        Returns:
            Whether optimization is needed
        """
        current_avg = self.get_current_average()
        trend = self.get_trend()
        
        return current_avg < threshold or trend == "declining"
