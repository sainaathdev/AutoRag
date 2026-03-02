"""Vector database interface for document storage and retrieval."""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from utils.logger import setup_logger


logger = setup_logger(__name__)


class VectorStore:
    """Vector database wrapper for ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize vector store.
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized vector store: {collection_name}")
    
    def add_documents(self, chunks: List[Dict]) -> int:
        """Add document chunks to vector store.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Extract data
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Prepare metadata (remove text field)
        metadatas = []
        for chunk in chunks:
            metadata = {k: str(v) for k, v in chunk.items() if k != "text"}
            metadatas.append(metadata)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add to collection
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of search results with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks from a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        # Get all chunks for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            return len(results['ids'])
        
        return 0
    
    def update_chunk_metadata(self, chunk_id: str, metadata: Dict):
        """Update metadata for a specific chunk.
        
        Args:
            chunk_id: Chunk identifier
            metadata: New metadata
        """
        # Get existing chunk
        result = self.collection.get(ids=[chunk_id])
        
        if result['ids']:
            # Update metadata
            existing_metadata = result['metadatas'][0]
            existing_metadata.update({k: str(v) for k, v in metadata.items()})
            
            # Update in collection
            self.collection.update(
                ids=[chunk_id],
                metadatas=[existing_metadata]
            )
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory
        }
    
    def reset_collection(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning(f"Reset collection: {self.collection_name}")
