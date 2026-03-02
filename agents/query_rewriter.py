"""Query rewriter agent for optimizing user queries."""

from typing import Dict, List
from .llm_client import DeepSeekClient
from utils.logger import setup_logger


logger = setup_logger(__name__)


QUERY_REWRITER_PROMPT = """You are a query optimization agent.

Given a user query, rewrite it to:
- Remove ambiguity
- Expand technical terms
- Add missing context
- Preserve original intent
- Make it more suitable for document retrieval

Analyze the query and determine:
1. Intent type (factual, procedural, conceptual, comparison, etc.)
2. Keywords that should be expanded
3. Any ambiguities that need clarification

Return your analysis in JSON format with these fields:
- original_query: the original query
- rewritten_query: the optimized query
- intent_type: the detected intent
- keywords_expanded: list of keywords that were expanded
- ambiguity_score: 0-1 score of how ambiguous the original query was
- reasoning: brief explanation of changes made"""


class QueryRewriterAgent:
    """Agent for rewriting and optimizing queries."""
    
    def __init__(self, llm_client: DeepSeekClient, enabled: bool = True):
        """Initialize query rewriter agent.
        
        Args:
            llm_client: DeepSeek client instance
            enabled: Whether query rewriting is enabled
        """
        self.llm_client = llm_client
        self.enabled = enabled
        
        logger.info(f"Initialized QueryRewriterAgent (enabled={enabled})")
    
    def rewrite_query(self, query: str) -> Dict:
        """Rewrite a user query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary with rewritten query and metadata
        """
        if not self.enabled:
            return {
                "original_query": query,
                "rewritten_query": query,
                "intent_type": "unknown",
                "keywords_expanded": [],
                "ambiguity_score": 0.0,
                "reasoning": "Query rewriting disabled"
            }
        
        try:
            messages = [
                {"role": "system", "content": QUERY_REWRITER_PROMPT},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            result = self.llm_client.chat_json(messages, temperature=0.3)
            
            logger.info(f"Rewrote query: '{query}' -> '{result.get('rewritten_query', query)}'")
            
            return result
        
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            # Return original query on failure
            return {
                "original_query": query,
                "rewritten_query": query,
                "intent_type": "unknown",
                "keywords_expanded": [],
                "ambiguity_score": 0.0,
                "reasoning": f"Rewriting failed: {str(e)}"
            }
    
    def should_rewrite(self, query: str, ambiguity_threshold: float = 0.7) -> bool:
        """Determine if a query should be rewritten.
        
        Args:
            query: User query
            ambiguity_threshold: Threshold for ambiguity score
            
        Returns:
            Whether query should be rewritten
        """
        if not self.enabled:
            return False
        
        # Simple heuristics (can be enhanced)
        if len(query.split()) <= 2:
            return True  # Very short queries likely need expansion
        
        if "?" not in query and len(query.split()) < 5:
            return True  # Short non-questions may be ambiguous
        
        return False
