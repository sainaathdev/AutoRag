"""Unit tests for Self-Improving RAG System."""

import pytest
from pathlib import Path
import tempfile
import shutil

from ingestion.chunking import DocumentChunker, SemanticChunker, AdaptiveChunker
from retrieval.vector_store import VectorStore
from utils.config_loader import Config


class TestDocumentChunker:
    """Test document chunking."""
    
    def test_basic_chunking(self):
        """Test basic fixed-size chunking."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        
        text = "This is a test document. " * 20
        chunks = chunker.chunk(text, "test_doc")
        
        assert len(chunks) > 0
        assert all(len(chunk[0]) <= 120 for chunk in chunks)  # Allow some variance
        
    def test_chunk_metadata(self):
        """Test chunk metadata."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        
        text = "Test document content."
        chunks = chunker.chunk(text, "test_doc")
        
        assert len(chunks) > 0
        chunk_text, metadata = chunks[0]
        
        assert metadata.document_id == "test_doc"
        assert metadata.chunk_index == 0
        assert metadata.chunking_strategy == "fixed_size"


class TestAdaptiveChunker:
    """Test adaptive chunking."""
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection."""
        config = {
            "default_chunk_size": 512,
            "default_overlap": 50,
            "semantic_chunking": False
        }
        
        chunker = AdaptiveChunker(config)
        
        text = "Test document. " * 50
        chunks = chunker.chunk(text, "test_doc")
        
        assert len(chunks) > 0
        
    def test_performance_tracking(self):
        """Test performance tracking."""
        config = {
            "default_chunk_size": 512,
            "default_overlap": 50,
            "semantic_chunking": False
        }
        
        chunker = AdaptiveChunker(config)
        
        # Update performance
        chunker.update_performance("doc1", 0.8)
        chunker.update_performance("doc1", 0.7)
        chunker.update_performance("doc1", 0.9)
        
        assert "doc1" in chunker.document_performance
        assert chunker.document_performance["doc1"]["query_count"] == 3


class TestVectorStore:
    """Test vector store operations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_vector_store_initialization(self, temp_dir):
        """Test vector store initialization."""
        store = VectorStore(
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        
        assert store.collection_name == "test_collection"
        
    def test_add_and_search(self, temp_dir):
        """Test adding documents and searching."""
        store = VectorStore(
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        
        # Add test documents
        chunks = [
            {
                "text": "Machine learning is a subset of AI.",
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "chunk_index": 0,
                "chunk_size": 40
            },
            {
                "text": "Deep learning uses neural networks.",
                "chunk_id": "chunk_2",
                "document_id": "doc_1",
                "chunk_index": 1,
                "chunk_size": 35
            }
        ]
        
        count = store.add_documents(chunks)
        assert count == 2
        
        # Search
        results = store.search("What is machine learning?", top_k=1)
        assert len(results) > 0
        assert "machine learning" in results[0]["text"].lower()


class TestConfig:
    """Test configuration loader."""
    
    def test_config_loading(self):
        """Test loading configuration."""
        # This assumes config.yaml exists
        if Path("config.yaml").exists():
            config = Config("config.yaml")
            
            assert config.get("deepseek.model") is not None
            assert config.get("chunking.default_chunk_size") is not None
    
    def test_config_get_section(self):
        """Test getting configuration section."""
        if Path("config.yaml").exists():
            config = Config("config.yaml")
            
            chunking_config = config.get_section("chunking")
            assert isinstance(chunking_config, dict)
            assert "default_chunk_size" in chunking_config


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_document_processing_pipeline(self, temp_dir):
        """Test complete document processing pipeline."""
        from ingestion import AdaptiveChunker, DocumentProcessor
        
        # Create test document
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("This is a test document about machine learning. " * 10)
        
        # Process document
        config = {
            "default_chunk_size": 100,
            "default_overlap": 20,
            "semantic_chunking": False
        }
        
        chunker = AdaptiveChunker(config)
        processor = DocumentProcessor(chunker)
        
        chunks = processor.process_document(str(test_file))
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
