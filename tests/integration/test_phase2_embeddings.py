"""
Phase 2 Test Script - WITH FREE OPTION
Test embeddings generation and FAISS vector store
Uses FREE Sentence Transformers if OpenAI quota exceeded
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from pathlib import Path
from loguru import logger
from src.ingestion.document_processor import DocumentProcessor
from src.embeddings.embeddings_generator import EmbeddingsGenerator
from src.embeddings.faiss_store import FAISSVectorStore
from src.utils.config import settings
import sys
import json

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add("phase2_test.log", rotation="1 MB")

# Global flag for which embeddings to use
USE_FREE_EMBEDDINGS = False

def get_embeddings_generator():
    """Get embeddings generator - tries OpenAI first, falls back to free option"""
    global USE_FREE_EMBEDDINGS
    
    # Try OpenAI first
    if not USE_FREE_EMBEDDINGS and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "add_later":
        try:
            logger.info("Attempting to use OpenAI embeddings...")
            generator = EmbeddingsGenerator(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            # Test it works
            generator.generate_embedding("test")
            logger.success("✓ Using OpenAI embeddings")
            return generator
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")
            logger.info("Falling back to FREE Sentence Transformers...")
            USE_FREE_EMBEDDINGS = True
    
    # Use free Sentence Transformers
    logger.info("Using FREE Sentence Transformers (no API key needed)...")
    generator = EmbeddingsGenerator(
        use_sentence_transformers=True,
        sentence_transformer_model="all-MiniLM-L6-v2"
    )
    logger.success("✓ Using FREE Sentence Transformers (384-dim embeddings)")
    return generator

def test_embeddings_generation():
    """Test embeddings generation"""
    logger.info("=" * 60)
    logger.info("TESTING EMBEDDINGS GENERATION")
    logger.info("=" * 60)
    
    # Get generator (tries OpenAI, falls back to free)
    generator = get_embeddings_generator()
    
    # Test 1: Single embedding
    logger.info("\n📝 Test 1: Single text embedding")
    test_text = "This is a test sentence for generating embeddings."
    
    embedding = generator.generate_embedding(test_text)
    logger.success(f"✓ Generated embedding")
    logger.info(f"  - Dimension: {len(embedding)}")
    logger.info(f"  - First 5 values: {[f'{v:.4f}' for v in embedding[:5]]}")
    logger.info(f"  - Model: {generator.model}")
    
    # Test 2: Batch embeddings
    logger.info("\n📝 Test 2: Batch embeddings")
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neural networks.",
        "Deep learning uses multiple layers to learn representations."
    ]
    
    embeddings = generator.generate_embeddings_batch(test_texts, batch_size=2)
    logger.success(f"✓ Generated {len(embeddings)} embeddings")
    logger.info(f"  - Each dimension: {len(embeddings[0])}")
    
    return generator

def test_document_embedding():
    """Test embedding a processed document"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DOCUMENT EMBEDDING")
    logger.info("=" * 60)
    
    # Process a document first
    processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    # Use the sample file from Phase 1
    test_file = settings.RAW_DATA_DIR / "sample_test.txt"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info("Run Phase 1 test first: python test_phase1.py")
        return None
    
    logger.info(f"\n1. Processing document: {test_file.name}")
    processed_doc = processor.process_file(str(test_file))
    logger.success(f"✓ Processed into {processed_doc['chunk_count']} chunks")
    
    # Generate embeddings
    logger.info(f"\n2. Generating embeddings for {processed_doc['chunk_count']} chunks...")
    generator = get_embeddings_generator()
    
    embedded_doc = generator.embed_document_chunks(
        processed_doc,
        save_path=settings.EMBEDDINGS_DIR / f"{test_file.stem}_embeddings.json"
    )
    
    logger.success(f"✓ Generated embeddings")
    logger.info(f"  - Chunks embedded: {embedded_doc['chunk_count']}")
    logger.info(f"  - Embedding dimension: {embedded_doc['embedding_dimension']}")
    logger.info(f"  - Model used: {embedded_doc['embedding_model']}")
    
    # Get stats
    stats = generator.get_embedding_stats(embedded_doc)
    logger.info(f"\n📊 Embedding Statistics:")
    logger.info(f"  - Mean norm: {stats['mean_norm']:.4f}")
    logger.info(f"  - Std norm: {stats['std_norm']:.4f}")
    
    return embedded_doc

def test_faiss_store():
    """Test FAISS vector store"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FAISS VECTOR STORE")
    logger.info("=" * 60)
    
    # Load embedded document
    embeddings_file = settings.EMBEDDINGS_DIR / "sample_test_embeddings.json"
    
    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        logger.info("Run document embedding test first")
        return None
    
    logger.info(f"\n1. Loading embeddings from: {embeddings_file.name}")
    with open(embeddings_file, 'r') as f:
        embedded_doc = json.load(f)
    
    # Create FAISS store
    logger.info(f"\n2. Creating FAISS vector store...")
    dimension = embedded_doc['embedding_dimension']
    store = FAISSVectorStore(dimension=dimension, index_type="flat")
    logger.success(f"✓ Created vector store (dimension: {dimension})")
    
    # Add document to store
    logger.info(f"\n3. Adding document to vector store...")
    store.add_document(embedded_doc)
    logger.success(f"✓ Added {embedded_doc['chunk_count']} chunks")
    
    # Get stats
    stats = store.get_stats()
    logger.info(f"\n📊 Vector Store Statistics:")
    logger.info(f"  - Total vectors: {stats['total_vectors']}")
    logger.info(f"  - Dimension: {stats['dimension']}")
    logger.info(f"  - Index type: {stats['index_type']}")
    logger.info(f"  - Documents: {stats['total_documents']}")
    
    # Test search
    logger.info(f"\n4. Testing semantic search...")
    
    # Use first chunk as query
    query_embedding = embedded_doc['chunks_with_embeddings'][0]['embedding']
    query_text = embedded_doc['chunks_with_embeddings'][0]['text'][:100]
    
    logger.info(f"Query text: '{query_text}...'")
    
    results = store.search(query_embedding, k=3)
    
    logger.success(f"✓ Found {len(results)} results")
    logger.info("\n🔍 Top Results:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Similarity: {result['similarity_score']:.4f}")
        logger.info(f"    Distance: {result['distance']:.4f}")
        logger.info(f"    Text: {result['text'][:100]}...")
    
    # Save vector store
    logger.info(f"\n5. Saving vector store...")
    store_path = settings.EMBEDDINGS_DIR / "vector_store"
    store.save(str(store_path))
    logger.success(f"✓ Saved to: {store_path}")
    
    # Test loading
    logger.info(f"\n6. Testing load from disk...")
    loaded_store = FAISSVectorStore.load(str(store_path))
    logger.success(f"✓ Loaded vector store")
    logger.info(f"  - Total vectors: {loaded_store.index.ntotal}")
    
    return store

def test_semantic_search():
    """Test semantic search with different queries"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SEMANTIC SEARCH")
    logger.info("=" * 60)
    
    # Load vector store
    store_path = settings.EMBEDDINGS_DIR / "vector_store"
    
    if not store_path.exists():
        logger.error("Vector store not found. Run previous tests first.")
        return
    
    logger.info(f"\n1. Loading vector store...")
    store = FAISSVectorStore.load(str(store_path))
    logger.success(f"✓ Loaded store with {store.index.ntotal} vectors")
    
    # Get embeddings generator
    generator = get_embeddings_generator()
    
    # Test queries
    test_queries = [
        "What is data science?",
        "machine learning algorithms",
        "processing documents",
    ]
    
    logger.info(f"\n2. Running {len(test_queries)} test queries...")
    
    for i, query_text in enumerate(test_queries, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Query {i}: '{query_text}'")
        logger.info(f"{'='*50}")
        
        # Generate query embedding
        query_embedding = generator.generate_embedding(query_text)
        
        # Search
        results = store.search(query_embedding, k=3)
        
        logger.info(f"\n🔍 Top {len(results)} Results:")
        for j, result in enumerate(results, 1):
            logger.info(f"\n  {j}. Similarity: {result['similarity_score']:.4f}")
            logger.info(f"     Text: {result['text'][:150]}...")
    
    logger.success("\n✓ Semantic search test completed!")

def test_end_to_end():
    """Complete end-to-end test"""
    logger.info("\n" + "=" * 60)
    logger.info("END-TO-END TEST")
    logger.info("=" * 60)
    
    logger.info("\n📋 Running complete pipeline:")
    logger.info("  1. Document Processing")
    logger.info("  2. Embeddings Generation")
    logger.info("  3. FAISS Vector Store")
    logger.info("  4. Semantic Search")
    
    if USE_FREE_EMBEDDINGS:
        logger.info("\n💡 Using FREE Sentence Transformers (no API costs!)")
    
    try:
        # All tests
        generator = test_embeddings_generation()
        if generator is None:
            return
        
        embedded_doc = test_document_embedding()
        if embedded_doc is None:
            return
        
        store = test_faiss_store()
        if store is None:
            return
        
        test_semantic_search()
        
        logger.info("\n" + "🎉" * 30)
        logger.success("ALL PHASE 2 TESTS PASSED!")
        logger.info("🎉" * 30)
        
        logger.info("\n✅ Phase 2 Complete! You now have:")
        if USE_FREE_EMBEDDINGS:
            logger.info("  ✓ Embeddings generation (FREE Sentence Transformers)")
        else:
            logger.info("  ✓ Embeddings generation (OpenAI)")
        logger.info("  ✓ FAISS vector store")
        logger.info("  ✓ Semantic search capability")
        logger.info("  ✓ Save/load functionality")
        
        logger.info("\n📊 Files created:")
        logger.info(f"  - {settings.EMBEDDINGS_DIR / 'sample_test_embeddings.json'}")
        logger.info(f"  - {settings.EMBEDDINGS_DIR / 'vector_store/'}")
        
        logger.info("\n🚀 Ready for Phase 3: Advanced RAG Implementation!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run all tests"""
    logger.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     ENTERPRISE AI ASSISTANT - PHASE 2 TEST SUITE        ║
    ║                                                           ║
    ║  Testing: Embeddings Generation & FAISS Vector Store     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    test_end_to_end()

if __name__ == "__main__":
    main()