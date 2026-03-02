"""RAGAS-style evaluation metrics for the RAG system.

Implements four core RAGAS metrics via LLM-as-judge:
  - Faithfulness: Does the answer stay within retrieved context?
  - Answer Relevancy: Is the answer on-topic for the question?
  - Context Precision: Are the TOP retrieved chunks actually relevant?
  - Context Recall: Does the context contain all information needed to answer?
"""

from typing import Dict, List, Optional
from .llm_client import DeepSeekClient
from utils.logger import setup_logger

logger = setup_logger(__name__)


FAITHFULNESS_PROMPT = """You are an expert judge evaluating a RAG system's answer faithfulness.

TASK: Determine whether each statement in the ANSWER can be directly inferred from the CONTEXT.

Rules:
- Score 1.0 = Every claim in the answer is supported by the context
- Score 0.0 = The answer makes claims not supported by the context (hallucination)
- Partial scores for partially supported answers

Respond ONLY with valid JSON:
{
  "score": <float 0.0-1.0>,
  "supported_statements": ["...", "..."],
  "unsupported_statements": ["...", "..."],
  "reasoning": "<short explanation>"
}"""


ANSWER_RELEVANCY_PROMPT = """You are an expert judge evaluating answer relevancy.

TASK: Does the ANSWER actually address the USER QUESTION?

Rules:
- Score 1.0 = Answer directly and completely addresses the question
- Score 0.0 = Answer is completely off-topic or evasive
- Deduct for incomplete answers, unnecessary tangents, or vague responses

Respond ONLY with valid JSON:
{
  "score": <float 0.0-1.0>,
  "addressed_aspects": ["...", "..."],
  "missing_aspects": ["...", "..."],
  "reasoning": "<short explanation>"
}"""


CONTEXT_PRECISION_PROMPT = """You are an expert judge evaluating context precision in a RAG system.

TASK: Of the retrieved CONTEXT CHUNKS, what fraction are actually relevant to answering the QUESTION?

Rules:
- Score 1.0 = All retrieved chunks are highly relevant
- Score 0.0 = None of the chunks are relevant to the question
- Higher-ranked chunks (listed first) matter more

Respond ONLY with valid JSON:
{
  "score": <float 0.0-1.0>,
  "relevant_chunk_indices": [0, 1, ...],
  "irrelevant_chunk_indices": [2, 3, ...],
  "reasoning": "<short explanation>"
}"""


CONTEXT_RECALL_PROMPT = """You are an expert judge evaluating context recall in a RAG system.

TASK: Does the provided CONTEXT contain all the information needed to answer the QUESTION?

Rules:
- Score 1.0 = Context has everything needed to give a perfect answer
- Score 0.0 = Context is missing critical information required to answer
- Partial credit for partially sufficient context

Respond ONLY with valid JSON:
{
  "score": <float 0.0-1.0>,
  "information_present": ["...", "..."],
  "information_missing": ["...", "..."],
  "reasoning": "<short explanation>"
}"""


class RAGASEvaluator:
    """Evaluates RAG system using RAGAS-style metrics via LLM-as-judge."""

    def __init__(self, llm_client: DeepSeekClient, enabled: bool = True):
        """Initialize RAGAS evaluator.

        Args:
            llm_client: LLM client instance
            enabled: Whether evaluation is enabled
        """
        self.llm_client = llm_client
        self.enabled = enabled
        self._history: List[Dict] = []  # Store evaluation history

        logger.info(f"Initialized RAGASEvaluator (enabled={enabled})")

    def _safe_llm_score(self, messages: List[Dict], score_key: str = "score") -> Dict:
        """Run LLM evaluation with fallback on failure."""
        try:
            result = self.llm_client.chat_json(messages, temperature=0.1)
            if score_key not in result:
                result[score_key] = 0.5
            result[score_key] = max(0.0, min(1.0, float(result[score_key])))
            return result
        except Exception as e:
            logger.warning(f"LLM evaluation call failed: {e}")
            return {
                score_key: 0.5,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    def evaluate_faithfulness(self, context: List[str], answer: str) -> Dict:
        """Measure whether the answer is grounded in the context.

        Args:
            context: Retrieved document chunks
            answer: Generated answer

        Returns:
            Dict with score and breakdown
        """
        context_str = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context)])
        messages = [
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nANSWER:\n{answer}"}
        ]
        return self._safe_llm_score(messages)

    def evaluate_answer_relevancy(self, query: str, answer: str) -> Dict:
        """Measure how well the answer addresses the question.

        Args:
            query: User question
            answer: Generated answer

        Returns:
            Dict with score and breakdown
        """
        messages = [
            {"role": "system", "content": ANSWER_RELEVANCY_PROMPT},
            {"role": "user", "content": f"USER QUESTION:\n{query}\n\nANSWER:\n{answer}"}
        ]
        return self._safe_llm_score(messages)

    def evaluate_context_precision(self, query: str, context: List[str]) -> Dict:
        """Measure what fraction of retrieved chunks are relevant.

        Args:
            query: User question
            context: Retrieved document chunks

        Returns:
            Dict with score and breakdown
        """
        context_str = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context)])
        messages = [
            {"role": "system", "content": CONTEXT_PRECISION_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT CHUNKS:\n{context_str}"}
        ]
        return self._safe_llm_score(messages)

    def evaluate_context_recall(self, query: str, context: List[str]) -> Dict:
        """Measure whether context contains all needed information.

        Args:
            query: User question
            context: Retrieved document chunks

        Returns:
            Dict with score and breakdown
        """
        context_str = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context)])
        messages = [
            {"role": "system", "content": CONTEXT_RECALL_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context_str}"}
        ]
        return self._safe_llm_score(messages)

    def full_evaluation(
        self,
        query: str,
        context: List[str],
        answer: str,
        store_history: bool = True
    ) -> Dict:
        """Run all four RAGAS metrics.

        Args:
            query: User query
            context: Retrieved chunks
            answer: Generated answer
            store_history: Whether to store this evaluation in history

        Returns:
            Full evaluation dict with all four metric scores
        """
        if not self.enabled:
            return self._default_scores()

        logger.info("Running RAGAS full evaluation...")

        faithfulness = self.evaluate_faithfulness(context, answer)
        answer_relevancy = self.evaluate_answer_relevancy(query, answer)
        context_precision = self.evaluate_context_precision(query, context)
        context_recall = self.evaluate_context_recall(query, context)

        scores = {
            "faithfulness": faithfulness.get("score", 0.5),
            "answer_relevancy": answer_relevancy.get("score", 0.5),
            "context_precision": context_precision.get("score", 0.5),
            "context_recall": context_recall.get("score", 0.5),
        }
        ragas_score = sum(scores.values()) / len(scores)

        result = {
            "ragas_score": round(ragas_score, 4),
            "scores": scores,
            "details": {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            },
            "query": query,
            "answer_preview": answer[:200],
        }

        if store_history:
            self._history.append(result)
            # Keep last 200 evals in memory
            if len(self._history) > 200:
                self._history = self._history[-200:]

        logger.info(
            f"RAGAS scores — Faithfulness:{scores['faithfulness']:.2f} | "
            f"Relevancy:{scores['answer_relevancy']:.2f} | "
            f"Precision:{scores['context_precision']:.2f} | "
            f"Recall:{scores['context_recall']:.2f} | "
            f"Overall:{ragas_score:.2f}"
        )
        return result

    def get_aggregate_stats(self) -> Dict:
        """Get aggregated RAGAS stats across all stored evaluations.

        Returns:
            Dict with mean scores per metric and overall trend
        """
        if not self._history:
            return self._default_scores(return_aggregate=True)

        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        aggregated = {}
        for metric in metric_names:
            vals = [e["scores"].get(metric, 0.5) for e in self._history]
            aggregated[metric] = {
                "mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "count": len(vals),
            }

        overall_vals = [e["ragas_score"] for e in self._history]
        aggregated["overall_ragas"] = round(sum(overall_vals) / len(overall_vals), 4)

        return aggregated

    def get_history(self, last_n: int = 50) -> List[Dict]:
        """Get recent RAGAS evaluation history.

        Args:
            last_n: Number of recent evaluations to return

        Returns:
            List of evaluation dicts
        """
        return self._history[-last_n:]

    def _default_scores(self, return_aggregate: bool = False) -> Dict:
        """Return default/empty score structure."""
        if return_aggregate:
            default = {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            return {
                "faithfulness": default,
                "answer_relevancy": default,
                "context_precision": default,
                "context_recall": default,
                "overall_ragas": 0.0,
            }
        return {
            "ragas_score": 0.0,
            "scores": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "details": {},
            "query": "",
            "answer_preview": "",
        }
