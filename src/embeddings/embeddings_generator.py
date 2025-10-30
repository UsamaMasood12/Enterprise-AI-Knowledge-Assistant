"""
Embeddings Generator Module
Generate vector embeddings using OpenAI or Sentence Transformers (free)
"""

import os
from typing import List, Dict, Optional
import numpy as np
from loguru import logger
import json
from pathlib import Path
from datetime import datetime

# Try to import OpenAI
try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

# Try to import Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available")


class EmbeddingsGenerator:
    """Generate embeddings for text chunks using OpenAI or Sentence Transformers"""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        use_sentence_transformers: bool = False,
        sentence_transformer_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize embeddings generator
        
        Args:
            openai_api_key: OpenAI API key (optional if using sentence transformers)
            model: OpenAI model name or sentence transformer model
            use_sentence_transformers: Use free Sentence Transformers instead of OpenAI
            sentence_transformer_model: Sentence transformer model name
        """
        self.use_sentence_transformers = use_sentence_transformers
        
        if use_sentence_transformers:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
            
            self.model = sentence_transformer_model
            self.st_model = SentenceTransformer(sentence_transformer_model)
            logger.info(f"EmbeddingsGenerator initialized with Sentence Transformers: {sentence_transformer_model}")
            
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Run: pip install openai")
            
            self.model = model
            self.client = OpenAI(api_key=openai_api_key)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.info(f"EmbeddingsGenerator initialized with OpenAI: {model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            
            if self.use_sentence_transformers:
                # Use Sentence Transformers (FREE!)
                embedding = self.st_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
                
            else:
                # Use OpenAI
                # Check token count
                tokens = self.encoding.encode(text)
                if len(tokens) > 8000:  # OpenAI limit
                    logger.warning(f"Text too long ({len(tokens)} tokens), truncating...")
                    text = self.encoding.decode(tokens[:8000])
                
                # Generate embedding
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                
                embedding = response.data[0].embedding
                return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        if self.use_sentence_transformers:
            # Sentence Transformers can batch natively
            logger.info(f"Using Sentence Transformers batch encoding...")
            embeddings_array = self.st_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            embeddings = embeddings_array.tolist()
            
        else:
            # Process in batches for OpenAI
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                for text in batch:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)
        
        logger.success(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_document_chunks(
        self, 
        processed_doc: Dict,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate embeddings for all chunks in a processed document
        
        Args:
            processed_doc: Processed document dictionary from DocumentProcessor
            save_path: Optional path to save embeddings
            
        Returns:
            Dictionary with document info and embeddings
        """
        logger.info(f"Embedding document: {processed_doc['file_name']}")
        
        # Extract chunk texts
        chunk_texts = [chunk['text'] for chunk in processed_doc['chunks']]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(chunk_texts)
        
        # Create result
        result = {
            'file_name': processed_doc['file_name'],
            'file_path': processed_doc['file_path'],
            'file_hash': processed_doc['file_hash'],
            'chunk_count': len(chunk_texts),
            'embedding_model': self.model,
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'chunks_with_embeddings': []
        }
        
        # Combine chunks with embeddings
        for chunk, embedding in zip(processed_doc['chunks'], embeddings):
            result['chunks_with_embeddings'].append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'embedding': embedding,
                'start_char': chunk['start_char'],
                'end_char': chunk['end_char']
            })
        
        result['embedded_at'] = datetime.utcnow().isoformat()
        
        # Save if path provided
        if save_path:
            self._save_embeddings(result, save_path)
        
        logger.success(f"Embedded {result['chunk_count']} chunks, dimension: {result['embedding_dimension']}")
        return result
    
    def _save_embeddings(self, embeddings_data: Dict, save_path: str):
        """Save embeddings to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info(f"Saved embeddings to: {save_path}")
    
    def load_embeddings(self, file_path: str) -> Dict:
        """Load embeddings from file"""
        with open(file_path, 'r') as f:
            embeddings_data = json.load(f)
        
        logger.info(f"Loaded embeddings from: {file_path}")
        return embeddings_data
    
    def get_embedding_stats(self, embeddings_data: Dict) -> Dict:
        """Get statistics about embeddings"""
        embeddings = [chunk['embedding'] for chunk in embeddings_data['chunks_with_embeddings']]
        embeddings_array = np.array(embeddings)
        
        stats = {
            'count': len(embeddings),
            'dimension': embeddings_array.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
            'model': embeddings_data['embedding_model']
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    from src.utils.config import settings
    
    # Configure logging
    logger.add("embeddings.log", rotation="1 MB")
    
    print("\n=== Testing Embeddings Generator ===\n")
    
    # Option 1: Try OpenAI (if you have credits)
    try:
        print("Trying OpenAI...")
        generator = EmbeddingsGenerator(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        test_text = "This is a test sentence."
        embedding = generator.generate_embedding(test_text)
        print(f"✓ OpenAI working! Dimension: {len(embedding)}")
        
    except Exception as e:
        print(f"✗ OpenAI failed: {e}")
        print("\nTrying FREE Sentence Transformers...")
        
        # Option 2: Use FREE Sentence Transformers
        generator = EmbeddingsGenerator(
            use_sentence_transformers=True,
            sentence_transformer_model="all-MiniLM-L6-v2"
        )
        test_text = "This is a test sentence."
        embedding = generator.generate_embedding(test_text)
        print(f"✓ Sentence Transformers working! Dimension: {len(embedding)}")