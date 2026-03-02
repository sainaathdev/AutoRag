"""Main Self-Improving RAG System."""

import uuid
from typing import List, Dict, Optional, Generator
from pathlib import Path

from utils.config_loader import get_config
from utils.logger import setup_logger
from ingestion import AdaptiveChunker, DocumentProcessor
from retrieval import VectorStore, HybridRetriever, CrossEncoderReranker
from agents import DeepSeekClient, QueryRewriterAgent, AnswerEvaluatorAgent, OptimizerAgent, RAGASEvaluator
from feedback import FeedbackStore, ConfidenceTracker


logger = setup_logger(__name__)


class SelfImprovingRAG:
    """Self-improving RAG system with adaptive learning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Self-Improving RAG System...")
        
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize components
        self._init_llm_client()
        self._init_vector_store()
        self._init_retriever()
        self._init_chunker()
        self._init_document_processor()
        self._init_agents()
        self._init_feedback_system()
        self._init_reranker()
        self._init_ragas_evaluator()
        
        # Query counter for optimization triggers
        self.query_count = 0
        
        logger.info("✓ Self-Improving RAG System initialized successfully")
    
    def _init_llm_client(self):
        """Initialize LLM client."""
        # Try Groq first, fallback to DeepSeek for backward compatibility
        llm_config = self.config.get_section("groq")
        if not llm_config or not llm_config.get("api_key"):
            llm_config = self.config.get_section("deepseek")
            provider = "DeepSeek"
        else:
            provider = "Groq"
        
        self.llm_client = DeepSeekClient(
            api_key=llm_config["api_key"],
            base_url=llm_config.get("base_url", "https://api.groq.com/openai/v1"),
            model=llm_config.get("model", "llama-3.3-70b-versatile"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2000)
        )
        logger.info(f"✓ LLM client initialized ({provider})")
    
    def _init_vector_store(self):
        """Initialize vector store."""
        vector_config = self.config.get_section("vector_db")
        embedding_config = self.config.get_section("embedding")
        
        self.vector_store = VectorStore(
            persist_directory=vector_config.get("persist_directory", "./data/vector_db"),
            collection_name=vector_config.get("collection_name", "rag_documents"),
            embedding_model=embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        logger.info("✓ Vector store initialized")
    
    def _init_retriever(self):
        """Initialize retriever."""
        retrieval_config = self.config.get_section("retrieval")
        
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            vector_weight=retrieval_config.get("vector_weight", 0.7),
            bm25_weight=retrieval_config.get("bm25_weight", 0.3)
        )
        self.default_top_k = retrieval_config.get("default_top_k", 5)
        self.use_hybrid = retrieval_config.get("hybrid_search_enabled", True)
        
        logger.info("✓ Retriever initialized")
    
    def _init_chunker(self):
        """Initialize adaptive chunker."""
        chunking_config = self.config.get_section("chunking")
        self.chunker = AdaptiveChunker(chunking_config)
        logger.info("✓ Adaptive chunker initialized")
    
    def _init_document_processor(self):
        """Initialize document processor."""
        self.doc_processor = DocumentProcessor(self.chunker)
        logger.info("✓ Document processor initialized")
    
    def _init_agents(self):
        """Initialize intelligent agents."""
        query_config = self.config.get_section("query_rewriting")
        eval_config = self.config.get_section("evaluation")
        opt_config = self.config.get_section("optimization")
        
        self.query_rewriter = QueryRewriterAgent(
            llm_client=self.llm_client,
            enabled=query_config.get("enabled", True)
        )
        
        self.answer_evaluator = AnswerEvaluatorAgent(
            llm_client=self.llm_client,
            enabled=eval_config.get("auto_evaluate", True)
        )
        
        self.optimizer = OptimizerAgent(
            llm_client=self.llm_client,
            enabled=opt_config.get("auto_optimization_enabled", True),
            auto_optimize=opt_config.get("auto_optimization_enabled", True)
        )
        
        logger.info("✓ Intelligent agents initialized")
    
    def _init_reranker(self):
        """Initialize cross-encoder reranker."""
        retrieval_config = self.config.get_section("retrieval")
        reranking_enabled = retrieval_config.get("reranking_enabled", True)
        
        reranker_model = retrieval_config.get(
            "reranker_model",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
            enabled=reranking_enabled
        )
        logger.info(f"✓ Cross-encoder reranker initialized (enabled={reranking_enabled})")
    
    def _init_ragas_evaluator(self):
        """Initialize RAGAS evaluator."""
        eval_config = self.config.get_section("evaluation")
        ragas_enabled = eval_config.get("ragas_enabled", True)
        
        self.ragas_evaluator = RAGASEvaluator(
            llm_client=self.llm_client,
            enabled=ragas_enabled
        )
        logger.info(f"✓ RAGAS evaluator initialized (enabled={ragas_enabled})")
    
    def _init_feedback_system(self):
        """Initialize feedback and tracking systems."""
        feedback_config = self.config.get_section("feedback")
        
        self.feedback_store = FeedbackStore(
            store_path=feedback_config.get("store_path", "./data/feedback")
        )
        
        self.confidence_tracker = ConfidenceTracker(
            window_size=feedback_config.get("performance_window", 100)
        )
        
        logger.info("✓ Feedback system initialized")
    
    def ingest_document(self, filepath: str, metadata: Optional[Dict] = None) -> int:
        """Ingest a document into the system.
        
        Args:
            filepath: Path to document
            metadata: Additional metadata
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Ingesting document: {filepath}")
        
        # Process document
        chunks = self.doc_processor.process_document(filepath, metadata)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Rebuild BM25 index
        if self.use_hybrid:
            self.retriever.rebuild_index()
        
        logger.info(f"✓ Ingested {len(chunks)} chunks from {filepath}")
        return len(chunks)
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        metadata: Optional[Dict] = None
    ) -> int:
        """Ingest all documents from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to process subdirectories
            metadata: Additional metadata
            
        Returns:
            Total number of chunks created
        """
        logger.info(f"Ingesting directory: {directory}")
        
        # Process all documents
        chunks = self.doc_processor.process_directory(directory, recursive, metadata)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Rebuild BM25 index
        if self.use_hybrid:
            self.retriever.rebuild_index()
        
        logger.info(f"✓ Ingested {len(chunks)} total chunks from {directory}")
        return len(chunks)
    
    def _generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer from context.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Generated answer
        """
        context_str = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(context)])
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Be concise and accurate
4. Cite which document(s) you used"""
        
        user_message = f"""Context:
{context_str}

Question: {query}

Answer:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.llm_client.chat(messages, temperature=0.3)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_metadata: bool = False
    ) -> Dict:
        """Query the RAG system.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_metadata: Whether to return detailed metadata
            
        Returns:
            Response dictionary with answer and metadata
        """
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query [{query_id}]: {query}")
        
        # Step 1: Query rewriting
        rewrite_result = self.query_rewriter.rewrite_query(query)
        final_query = rewrite_result.get("rewritten_query", query)
        
        if final_query != query:
            logger.info(f"Query rewritten: '{query}' -> '{final_query}'")
        
        # Step 2: Retrieval
        top_k = top_k or self.default_top_k
        retrieval_method = "hybrid" if self.use_hybrid else "vector"
        
        retrieved_chunks = self.retriever.search(
            final_query,
            top_k=top_k,
            use_hybrid=self.use_hybrid
        )
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks using {retrieval_method}")
        
        # Step 2b: Cross-encoder reranking
        retrieved_chunks = self.reranker.rerank(final_query, retrieved_chunks, top_k=top_k)
        reranked = any("rerank_score" in c for c in retrieved_chunks)
        if reranked:
            logger.info("✓ Chunks reranked by cross-encoder")
        
        # Step 3: Generate answer
        context = [chunk["text"] for chunk in retrieved_chunks]
        answer = self._generate_answer(query, context)
        
        # Step 4: Evaluate answer (existing confidence evaluator)
        evaluation = self.answer_evaluator.evaluate_answer(query, context, answer)
        confidence_score = evaluation.get("confidence_score", 0.5)
        
        logger.info(f"Answer generated with confidence: {confidence_score:.2f}")
        
        # Step 4b: RAGAS evaluation (async-friendly, non-blocking on failure)
        try:
            ragas_result = self.ragas_evaluator.full_evaluation(
                query=query, context=context, answer=answer
            )
        except Exception as e:
            logger.warning(f"RAGAS evaluation skipped: {e}")
            ragas_result = None
        
        # Step 5: Track performance
        self.confidence_tracker.add_score(confidence_score)
        self.optimizer.update_retrieval_stats(retrieval_method, confidence_score)
        
        # Update document stats
        for chunk in retrieved_chunks:
            doc_id = chunk.get("metadata", {}).get("document_id")
            if doc_id:
                is_failure = confidence_score < self.config.get("confidence.auto_improve_threshold", 0.6)
                self.optimizer.update_document_stats(doc_id, confidence_score, is_failure)
        
        # Step 6: Store feedback
        self.feedback_store.add_feedback(
            query_id=query_id,
            query=query,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            confidence_score=confidence_score,
            evaluation=evaluation,
            rewritten_query=final_query if final_query != query else None,
            retrieval_method=retrieval_method
        )
        
        # Step 7: Check if optimization needed
        self.query_count += 1
        if self.query_count % self.config.get("feedback.optimization_interval", 50) == 0:
            self._trigger_optimization()
        
        # Prepare response
        response = {
            "answer": answer,
            "confidence_score": confidence_score,
            "query_id": query_id
        }
        
        if return_metadata:
            response.update({
                "original_query": query,
                "rewritten_query": final_query,
                "retrieved_chunks": retrieved_chunks,
                "evaluation": evaluation,
                "retrieval_method": retrieval_method,
                "ragas": ragas_result,
                "reranked": reranked,
            })
        
        return response
    
    def stream_query(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Generator:
        """Stream query response token-by-token.
        
        Yields status update strings first, then token chunks.
        Each yielded item is a dict with 'type' key:
          - {'type': 'status', 'text': '...'}
          - {'type': 'token', 'text': '...'}
          - {'type': 'done', 'metadata': {...}}
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Yields:
            Streaming response dicts
        """
        query_id = str(uuid.uuid4())
        
        # Step 1: Query rewriting
        yield {"type": "status", "text": "🔍 Rewriting query..."}
        rewrite_result = self.query_rewriter.rewrite_query(query)
        final_query = rewrite_result.get("rewritten_query", query)
        
        # Step 2: Retrieval
        yield {"type": "status", "text": "📚 Retrieving relevant documents..."}
        top_k = top_k or self.default_top_k
        retrieval_method = "hybrid" if self.use_hybrid else "vector"
        
        retrieved_chunks = self.retriever.search(
            final_query, top_k=top_k, use_hybrid=self.use_hybrid
        )
        
        # Step 2b: Reranking
        yield {"type": "status", "text": "⚡ Reranking with cross-encoder..."}
        retrieved_chunks = self.reranker.rerank(final_query, retrieved_chunks, top_k=top_k)
        context = [chunk["text"] for chunk in retrieved_chunks]
        
        # Step 3: Stream answer generation
        yield {"type": "status", "text": "🤖 Generating answer..."}
        
        context_str = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(context)])
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Be concise and accurate
4. Cite which document(s) you used"""
        
        user_message = f"""Context:
{context_str}

Question: {query}

Answer:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        full_answer = ""
        try:
            stream = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=messages,
                temperature=0.3,
                max_tokens=self.llm_client.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_answer += delta
                    yield {"type": "token", "text": delta}
        except Exception as e:
            logger.error(f"Streaming failed, falling back to non-streaming: {e}")
            full_answer = self._generate_answer(query, context)
            yield {"type": "token", "text": full_answer}
        
        # Step 4: Evaluate
        evaluation = self.answer_evaluator.evaluate_answer(query, context, full_answer)
        confidence_score = evaluation.get("confidence_score", 0.5)
        
        # Track performance
        self.confidence_tracker.add_score(confidence_score)
        self.optimizer.update_retrieval_stats(retrieval_method, confidence_score)
        self.feedback_store.add_feedback(
            query_id=query_id,
            query=query,
            retrieved_chunks=retrieved_chunks,
            answer=full_answer,
            confidence_score=confidence_score,
            evaluation=evaluation,
            rewritten_query=final_query if final_query != query else None,
            retrieval_method=retrieval_method,
        )
        self.query_count += 1
        if self.query_count % self.config.get("feedback.optimization_interval", 50) == 0:
            self._trigger_optimization()
        
        # Step 5: RAGAS evaluation (runs after streaming completes)
        yield {"type": "status", "text": "📐 Running RAGAS evaluation..."}
        ragas_result = None
        try:
            ragas_result = self.ragas_evaluator.full_evaluation(
                query=query, context=context, answer=full_answer
            )
        except Exception as e:
            logger.warning(f"RAGAS evaluation skipped in stream: {e}")
        
        yield {
            "type": "done",
            "metadata": {
                "query_id": query_id,
                "confidence_score": confidence_score,
                "rewritten_query": final_query,
                "retrieval_method": retrieval_method,
                "num_chunks": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,
                "evaluation": evaluation,
                "ragas": ragas_result,
            }
        }
    
    def _trigger_optimization(self):
        """Trigger system optimization."""
        logger.info("Triggering system optimization...")
        
        # Check if optimization is needed
        if not self.confidence_tracker.needs_optimization():
            logger.info("System performing well, no optimization needed")
            return
        
        # Get problematic documents
        problematic_docs = self.optimizer.get_problematic_documents()
        
        if problematic_docs:
            logger.warning(f"Found {len(problematic_docs)} problematic documents")
            # In a full implementation, you would trigger re-chunking here
        
        # Get optimization report
        report = self.optimizer.get_optimization_report()
        logger.info(f"Optimization report: {report}")
    
    def get_statistics(self) -> Dict:
        """Get system statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "confidence": self.confidence_tracker.get_statistics(),
            "feedback": self.feedback_store.get_statistics(),
            "optimization": self.optimizer.get_optimization_report()
        }
    
    def reset(self):
        """Reset the system (clear all data)."""
        logger.warning("Resetting RAG system...")
        self.vector_store.reset_collection()
        if self.use_hybrid:
            self.retriever.rebuild_index()
        logger.warning("✓ System reset complete")
