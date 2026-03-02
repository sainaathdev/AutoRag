"""Document ingestion and processing."""

import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from docx import Document as DocxDocument

from .chunking import AdaptiveChunker, ChunkMetadata
from utils.logger import setup_logger


logger = setup_logger(__name__)


class DocumentProcessor:
    """Process and ingest documents into the RAG system."""
    
    def __init__(self, chunker: AdaptiveChunker):
        """Initialize document processor.
        
        Args:
            chunker: Adaptive chunker instance
        """
        self.chunker = chunker
    
    def _generate_document_id(self, filepath: str) -> str:
        """Generate unique document ID from filepath.
        
        Args:
            filepath: Path to document
            
        Returns:
            Unique document ID
        """
        return hashlib.md5(filepath.encode()).hexdigest()
    
    def _extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF file.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = []
        with open(filepath, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        
        return '\n'.join(text)
    
    def _extract_text_from_docx(self, filepath: str) -> str:
        """Extract text from DOCX file.
        
        Args:
            filepath: Path to DOCX file
            
        Returns:
            Extracted text
        """
        doc = DocxDocument(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    def _extract_text_from_txt(self, filepath: str) -> str:
        """Extract text from TXT file.
        
        Args:
            filepath: Path to TXT file
            
        Returns:
            Extracted text
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def process_document(
        self,
        filepath: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Process a document and return chunks with metadata.
        
        Args:
            filepath: Path to document
            metadata: Additional metadata
            
        Returns:
            List of chunk dictionaries
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")
        
        # Extract text based on file type
        extension = path.suffix.lower()
        
        try:
            if extension == '.pdf':
                text = self._extract_text_from_pdf(filepath)
            elif extension == '.docx':
                text = self._extract_text_from_docx(filepath)
            elif extension == '.txt':
                text = self._extract_text_from_txt(filepath)
            else:
                raise ValueError(f"Unsupported file type: {extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            raise
        
        # Generate document ID
        document_id = self._generate_document_id(filepath)
        
        # Chunk document
        chunks = self.chunker.chunk(text, document_id)
        
        # Prepare chunk data
        chunk_data = []
        for chunk_text, chunk_metadata in chunks:
            data = {
                "text": chunk_text,
                "chunk_id": chunk_metadata.chunk_id,
                "document_id": document_id,
                "chunk_index": chunk_metadata.chunk_index,
                "chunk_size": chunk_metadata.chunk_size,
                "chunking_strategy": chunk_metadata.chunking_strategy,
                "source_file": str(path),
                "file_type": extension,
            }
            
            # Add custom metadata
            if metadata:
                data.update(metadata)
            
            chunk_data.append(data)
        
        logger.info(f"Processed {filepath}: {len(chunk_data)} chunks created")
        return chunk_data
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Process all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to process subdirectories
            metadata: Additional metadata
            
        Returns:
            List of all chunks from all documents
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_chunks = []
        pattern = "**/*" if recursive else "*"
        
        for filepath in dir_path.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in ['.pdf', '.docx', '.txt']:
                try:
                    chunks = self.process_document(str(filepath), metadata)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {filepath}: {e}")
        
        logger.info(f"Processed directory {directory}: {len(all_chunks)} total chunks")
        return all_chunks
