"""Document chunking with adaptive strategies."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_size: int
    overlap: int
    chunking_strategy: str
    performance_score: float = 1.0
    failure_count: int = 0


class DocumentChunker:
    """Base document chunker with fixed-size chunking."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, document_id: str) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk document into fixed-size pieces.
        
        Args:
            text: Document text
            document_id: Unique document identifier
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk size
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_size=len(chunk_text),
                overlap=self.overlap,
                chunking_strategy="fixed_size"
            )
            
            chunks.append((chunk_text.strip(), metadata))
            
            start = end - self.overlap
            chunk_index += 1
        
        return chunks


class SemanticChunker:
    """Semantic chunking based on sentence embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 1024
    ):
        """Initialize semantic chunker.
        
        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Similarity threshold for grouping sentences
            max_chunk_size: Maximum chunk size
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def chunk(self, text: str, document_id: str) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk document based on semantic similarity.
        
        Args:
            text: Document text
            document_id: Unique document identifier
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Encode sentences
        embeddings = self.model.encode(sentences)
        
        # Group semantically similar sentences
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        chunk_index = 0
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_embedding, embeddings[i])
            current_size = sum(len(s) for s in current_chunk)
            
            # Add to current chunk if similar and not too large
            if similarity >= self.similarity_threshold and current_size < self.max_chunk_size:
                current_chunk.append(sentences[i])
                # Update chunk embedding (average)
                current_embedding = (current_embedding + embeddings[i]) / 2
            else:
                # Save current chunk and start new one
                chunk_text = ' '.join(current_chunk)
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    chunk_size=len(chunk_text),
                    overlap=0,
                    chunking_strategy="semantic"
                )
                chunks.append((chunk_text, metadata))
                
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
                chunk_index += 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_size=len(chunk_text),
                overlap=0,
                chunking_strategy="semantic"
            )
            chunks.append((chunk_text, metadata))
        
        return chunks


class AdaptiveChunker:
    """Adaptive chunker that adjusts strategy based on performance."""
    
    def __init__(self, config: Dict):
        """Initialize adaptive chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        self.default_chunker = DocumentChunker(
            chunk_size=config.get("default_chunk_size", 512),
            overlap=config.get("default_overlap", 50)
        )
        
        if config.get("semantic_chunking", False):
            self.semantic_chunker = SemanticChunker(
                max_chunk_size=config.get("max_chunk_size", 1024)
            )
        else:
            self.semantic_chunker = None
        
        # Track document performance
        self.document_performance: Dict[str, Dict] = {}
    
    def chunk(
        self,
        text: str,
        document_id: str,
        force_strategy: str = None
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk document with adaptive strategy selection.
        
        Args:
            text: Document text
            document_id: Unique document identifier
            force_strategy: Force specific strategy ('fixed', 'semantic')
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        # Determine strategy
        if force_strategy:
            strategy = force_strategy
        elif document_id in self.document_performance:
            perf = self.document_performance[document_id]
            # If performance is poor, try semantic chunking
            if perf.get("avg_confidence", 1.0) < 0.6 and self.semantic_chunker:
                strategy = "semantic"
            else:
                strategy = "fixed"
        else:
            strategy = "fixed"
        
        # Apply chunking
        if strategy == "semantic" and self.semantic_chunker:
            return self.semantic_chunker.chunk(text, document_id)
        else:
            return self.default_chunker.chunk(text, document_id)
    
    def rechunk_document(
        self,
        text: str,
        document_id: str,
        reduce_size: bool = True
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Re-chunk a document with adjusted parameters.
        
        Args:
            text: Document text
            document_id: Unique document identifier
            reduce_size: Whether to reduce chunk size
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        if reduce_size:
            # Try smaller chunks with more overlap
            new_chunk_size = max(
                self.config.get("min_chunk_size", 256),
                int(self.config.get("default_chunk_size", 512) * 0.7)
            )
            new_overlap = int(new_chunk_size * 0.2)
            
            chunker = DocumentChunker(chunk_size=new_chunk_size, overlap=new_overlap)
            return chunker.chunk(text, document_id)
        else:
            # Try semantic chunking
            if self.semantic_chunker:
                return self.semantic_chunker.chunk(text, document_id)
            else:
                return self.default_chunker.chunk(text, document_id)
    
    def update_performance(self, document_id: str, confidence_score: float):
        """Update document performance metrics.
        
        Args:
            document_id: Document identifier
            confidence_score: Confidence score for this query
        """
        if document_id not in self.document_performance:
            self.document_performance[document_id] = {
                "scores": [],
                "avg_confidence": 1.0,
                "query_count": 0
            }
        
        perf = self.document_performance[document_id]
        perf["scores"].append(confidence_score)
        perf["query_count"] += 1
        
        # Keep only recent scores
        if len(perf["scores"]) > 20:
            perf["scores"] = perf["scores"][-20:]
        
        perf["avg_confidence"] = np.mean(perf["scores"])
