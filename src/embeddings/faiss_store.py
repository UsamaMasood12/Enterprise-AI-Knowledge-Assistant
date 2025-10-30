"""
FAISS Vector Store Module
Store and search embeddings using FAISS
"""

import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import pickle
from loguru import logger
from datetime import datetime

class FAISSVectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []  # Store document metadata
        self.id_to_doc_map = {}  # Map FAISS ID to document info
        
        self._create_index()
        logger.info(f"FAISSVectorStore initialized: dimension={dimension}, type={index_type}")
    
    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == "flat":
            # Flat L2 index (exact search, best quality)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created Flat L2 index (exact search)")
            
        elif self.index_type == "ivf":
            # IVF index (faster, approximate search)
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"Created IVF index with {nlist} clusters")
            
        elif self.index_type == "hnsw":
            # HNSW index (very fast, approximate search)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            logger.info("Created HNSW index")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadata: List[Dict]
    ):
        """
        Add embeddings to the index
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries for each embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings_array)
            logger.success("Index trained")
        
        # Get starting ID
        start_id = self.index.ntotal
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, meta in enumerate(metadata):
            doc_id = start_id + i
            self.id_to_doc_map[doc_id] = meta
            self.documents.append(meta)
        
        logger.success(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
    
    def add_document(self, embeddings_data: Dict):
        """
        Add all chunks from an embedded document
        
        Args:
            embeddings_data: Output from EmbeddingsGenerator.embed_document_chunks()
        """
        logger.info(f"Adding document: {embeddings_data['file_name']}")
        
        embeddings = []
        metadata = []
        
        for chunk in embeddings_data['chunks_with_embeddings']:
            embeddings.append(chunk['embedding'])
            metadata.append({
                'file_name': embeddings_data['file_name'],
                'file_path': embeddings_data['file_path'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'start_char': chunk['start_char'],
                'end_char': chunk['end_char']
            })
        
        self.add_embeddings(embeddings, metadata)
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries with results
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Convert to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            # Get metadata
            metadata = self.id_to_doc_map.get(int(idx), {})
            
            results.append({
                'index': int(idx),
                'distance': float(dist),
                'similarity_score': float(1 / (1 + dist)),  # Convert distance to similarity
                'text': metadata.get('text', ''),
                'file_name': metadata.get('file_name', ''),
                'chunk_id': metadata.get('chunk_id', -1),
                'metadata': metadata
            })
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def search_with_threshold(
        self,
        query_embedding: List[float],
        k: int = 5,
        distance_threshold: float = 1.0
    ) -> List[Dict]:
        """
        Search with distance threshold
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            distance_threshold: Maximum distance threshold
            
        Returns:
            List of results below threshold
        """
        results = self.search(query_embedding, k)
        filtered = [r for r in results if r['distance'] <= distance_threshold]
        
        logger.info(f"Filtered to {len(filtered)} results below threshold {distance_threshold}")
        return filtered
    
    def save(self, directory: str):
        """
        Save index and metadata to disk
        
        Args:
            directory: Directory to save files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = directory / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to: {index_path}")
        
        # Save metadata
        metadata_path = directory / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'index_type': self.index_type,
                'id_to_doc_map': self.id_to_doc_map,
                'documents': self.documents,
                'saved_at': datetime.utcnow().isoformat()
            }, f)
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Save config
        config_path = directory / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'dimension': self.dimension,
                'index_type': self.index_type,
                'total_vectors': self.index.ntotal,
                'total_documents': len(set(d.get('file_name', '') for d in self.documents)),
                'saved_at': datetime.utcnow().isoformat()
            }, f, indent=2)
        
        logger.success(f"Vector store saved to: {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load index and metadata from disk
        
        Args:
            directory: Directory containing saved files
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        directory = Path(directory)
        
        # Load metadata first
        metadata_path = directory / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls(
            dimension=metadata['dimension'],
            index_type=metadata['index_type']
        )
        
        # Load FAISS index
        index_path = directory / "faiss_index.bin"
        store.index = faiss.read_index(str(index_path))
        
        # Restore metadata
        store.id_to_doc_map = metadata['id_to_doc_map']
        store.documents = metadata['documents']
        
        logger.success(f"Loaded vector store from: {directory}")
        logger.info(f"Total vectors: {store.index.ntotal}")
        
        return store
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        unique_files = set(d.get('file_name', '') for d in self.documents)
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_documents': len(unique_files),
            'unique_files': list(unique_files),
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
        
        return stats
    
    def clear(self):
        """Clear the index"""
        self._create_index()
        self.documents = []
        self.id_to_doc_map = {}
        logger.info("Vector store cleared")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logger.add("faiss_store.log", rotation="1 MB")
    
    # Example: Create store and add embeddings
    dimension = 1536  # OpenAI text-embedding-3-small dimension
    
    store = FAISSVectorStore(dimension=dimension, index_type="flat")
    
    # Example data
    example_embeddings = [
        [0.1] * dimension,  # Mock embedding
        [0.2] * dimension,
        [0.3] * dimension
    ]
    
    example_metadata = [
        {'text': 'First chunk', 'file_name': 'doc1.pdf', 'chunk_id': 0},
        {'text': 'Second chunk', 'file_name': 'doc1.pdf', 'chunk_id': 1},
        {'text': 'Third chunk', 'file_name': 'doc2.pdf', 'chunk_id': 0}
    ]
    
    store.add_embeddings(example_embeddings, example_metadata)
    
    # Search
    query = [0.15] * dimension
    results = store.search(query, k=2)
    
    print(f"\nSearch Results:")
    for r in results:
        print(f"  - {r['text'][:50]} (score: {r['similarity_score']:.3f})")
    
    # Stats
    stats = store.get_stats()
    print(f"\nStore Stats: {stats}")
