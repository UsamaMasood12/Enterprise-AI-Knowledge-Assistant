"""
Document Processing Module
Handles PDF, Word, Excel, Text, and Image documents
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import pdfplumber
from docx import Document
import pandas as pd
from loguru import logger
import hashlib
from datetime import datetime

class DocumentProcessor:
    """Process various document types into text chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def process_file(self, file_path: str) -> Dict:
        """
        Process a file and return structured data
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with processed document data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        file_ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size
        file_hash = self._get_file_hash(file_path)
        
        logger.info(f"Processing file: {file_path.name} ({file_ext}, {file_size} bytes)")
        
        # Extract text based on file type
        if file_ext == '.pdf':
            text = self._process_pdf(file_path)
        elif file_ext in ['.doc', '.docx']:
            text = self._process_word(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            text = self._process_excel(file_path)
        elif file_ext == '.csv':
            text = self._process_csv(file_path)
        elif file_ext in ['.txt', '.md', '.rst']:
            text = self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Create chunks
        chunks = self._create_chunks(text)
        
        logger.info(f"Extracted {len(text)} characters, created {len(chunks)} chunks")
        
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': file_ext,
            'file_size': file_size,
            'file_hash': file_hash,
            'text': text,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'processed_at': datetime.utcnow().isoformat(),
            'metadata': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }
    
    def _process_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        
        try:
            # Try pdfplumber first (better for complex PDFs)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"Both PDF readers failed: {e}")
                raise
        
        return text.strip()
    
    def _process_word(self, file_path: Path) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to process Word document: {e}")
            raise
    
    def _process_excel(self, file_path: Path) -> str:
        """Extract text from Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += df.to_string(index=False) + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to process Excel file: {e}")
            raise
    
    def _process_csv(self, file_path: Path) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False).strip()
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            raise
    
    def _process_text(self, file_path: Path) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
    
    def _create_chunks(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Simple character-based chunking
        text_length = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at sentence or word boundary
            if end < text_length:
                # Look for sentence end
                for separator in ['. ', '.\n', '! ', '?\n']:
                    last_sep = text[start:end].rfind(separator)
                    if last_sep != -1:
                        end = start + last_sep + len(separator)
                        break
                else:
                    # If no sentence end, look for word boundary
                    last_space = text[start:end].rfind(' ')
                    if last_space != -1:
                        end = start + last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'char_count': len(chunk_text)
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def batch_process(self, file_paths: List[str]) -> List[Dict]:
        """
        Process multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of processed document dictionaries
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    'file_name': Path(file_path).name,
                    'file_path': file_path,
                    'error': str(e),
                    'processed_at': datetime.utcnow().isoformat()
                })
        
        return results


# Example usage
if __name__ == "__main__":
    # Configure logging
    logger.add("document_processing.log", rotation="1 MB")
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Example: Process a file
    # result = processor.process_file("path/to/your/document.pdf")
    # print(f"Processed {result['file_name']}: {result['chunk_count']} chunks")
