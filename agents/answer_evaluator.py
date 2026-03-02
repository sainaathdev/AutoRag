"""Answer evaluator agent for assessing answer quality."""

from typing import Dict, List
from .llm_client import DeepSeekClient
from utils.logger import setup_logger


logger = setup_logger(__name__)


ANSWER_EVALUATOR_PROMPT = """You are an answer evaluation agent.

Given:
- User query
- Retrieved context (documents used to generate answer)
- Generated answer

Evaluate the answer on these criteria:

1. **Context Adherence**: Does the answer strictly use information from the retrieved context?
2. **Hallucination Detection**: Is there any information in the answer that is NOT in the context?
3. **Completeness**: Does the answer fully address the user's query?
4. **Relevance**: Is the answer relevant to the query?

Provide scores and analysis in JSON format:
- confidence_score: 0-1 score (1 = perfect answer, 0 = completely wrong)
- hallucination_detected: true/false
- completeness_score: 0-1 score
- relevance_score: 0-1 score
- context_used: true/false (whether context was actually used)
- failure_reason: string or null (reason if confidence < 0.6)
- suggestions: list of improvement suggestions

Be strict in your evaluation. If the answer contains ANY information not in the context, mark hallucination_detected as true."""


class AnswerEvaluatorAgent:
    """Agent for evaluating answer quality."""
    
    def __init__(self, llm_client: DeepSeekClient, enabled: bool = True):
        """Initialize answer evaluator agent.
        
        Args:
            llm_client: DeepSeek client instance
            enabled: Whether evaluation is enabled
        """
        self.llm_client = llm_client
        self.enabled = enabled
        
        logger.info(f"Initialized AnswerEvaluatorAgent (enabled={enabled})")
    
    def evaluate_answer(
        self,
        query: str,
        context: List[str],
        answer: str
    ) -> Dict:
        """Evaluate an answer against query and context.
        
        Args:
            query: User query
            context: List of context documents
            answer: Generated answer
            
        Returns:
            Evaluation results dictionary
        """
        if not self.enabled:
            return {
                "confidence_score": 0.8,
                "hallucination_detected": False,
                "completeness_score": 0.8,
                "relevance_score": 0.8,
                "context_used": True,
                "failure_reason": None,
                "suggestions": []
            }
        
        try:
            # Prepare context string
            context_str = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(context)])
            
            user_message = f"""Query: {query}

Retrieved Context:
{context_str}

Generated Answer:
{answer}

Evaluate this answer."""
            
            messages = [
                {"role": "system", "content": ANSWER_EVALUATOR_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            result = self.llm_client.chat_json(messages, temperature=0.2)
            
            confidence = result.get("confidence_score", 0.5)
            logger.info(f"Answer evaluation: confidence={confidence:.2f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            # Return conservative scores on failure
            return {
                "confidence_score": 0.5,
                "hallucination_detected": False,
                "completeness_score": 0.5,
                "relevance_score": 0.5,
                "context_used": True,
                "failure_reason": f"Evaluation failed: {str(e)}",
                "suggestions": []
            }
    
    def quick_confidence_check(
        self,
        query: str,
        context: List[str],
        answer: str
    ) -> float:
        """Quick confidence check without full evaluation.
        
        Args:
            query: User query
            context: List of context documents
            answer: Generated answer
            
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristics for quick check
        if not answer or len(answer) < 10:
            return 0.2
        
        if "I don't know" in answer or "cannot answer" in answer.lower():
            return 0.3
        
        if not context:
            return 0.4
        
        # Check if answer uses context (simple keyword overlap)
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(context).lower().split())
        overlap = len(answer_words & context_words)
        
        if overlap < 3:
            return 0.5
        
        return 0.7  # Default moderate confidence
