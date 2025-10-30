"""
Phase 3 Test Script
Test Advanced RAG Implementation
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pathlib import Path
from loguru import logger
from src.embeddings.embeddings_generator import EmbeddingsGenerator
from src.embeddings.faiss_store import FAISSVectorStore
from src.retrieval.rag_retriever import AdvancedRAGRetriever
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config import settings
import sys
import json

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add("phase3_test.log", rotation="1 MB")

def test_basic_retrieval():
    """Test basic RAG retrieval"""
    logger.info("=" * 60)
    logger.info("TESTING BASIC RAG RETRIEVAL")
    logger.info("=" * 60)
    
    # Load vector store
    store_path = settings.EMBEDDINGS_DIR / "vector_store"
    if not store_path.exists():
        logger.error("Vector store not found. Run Phase 2 first!")
        return None, None
    
    logger.info("\n1. Loading vector store...")
    vector_store = FAISSVectorStore.load(str(store_path))
    logger.success(f"✓ Loaded {vector_store.index.ntotal} vectors")
    
    # Initialize embeddings generator
    logger.info("\n2. Initializing embeddings generator...")
    embeddings_gen = EmbeddingsGenerator(
        openai_api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    logger.success("✓ Embeddings generator ready")
    
    # Initialize retriever
    logger.info("\n3. Initializing advanced retriever...")
    retriever = AdvancedRAGRetriever(
        vector_store=vector_store,
        embeddings_generator=embeddings_gen,
        llm_api_key=settings.OPENAI_API_KEY,
        llm_model="gpt-4o-mini"
    )
    logger.success("✓ Advanced retriever ready")
    
    # Test simple retrieval
    logger.info("\n4. Testing simple retrieval...")
    query = "What is machine learning?"
    
    query_embedding = embeddings_gen.generate_embedding(query)
    results = vector_store.search(query_embedding, k=3)
    
    logger.success(f"✓ Retrieved {len(results)} results")
    logger.info("\n📄 Top Results:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n  {i}. Similarity: {result['similarity_score']:.4f}")
        logger.info(f"     Text: {result['text'][:150]}...")
    
    return vector_store, retriever

def test_multi_query_retrieval(retriever):
    """Test multi-query retrieval"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MULTI-QUERY RETRIEVAL")
    logger.info("=" * 60)
    
    query = "How does the system process documents?"
    
    logger.info(f"\nOriginal Query: '{query}'")
    logger.info("\nGenerating alternative queries and retrieving...")
    
    results = retriever.multi_query_retrieval(
        query=query,
        num_queries=2,
        k_per_query=2
    )
    
    logger.success(f"\n✓ Multi-query retrieval returned {len(results)} unique results")
    
    logger.info("\n📄 Results by Source Query:")
    for i, result in enumerate(results[:5], 1):
        logger.info(f"\n  {i}. Source Query: {result['source_query']}")
        logger.info(f"     Similarity: {result['similarity_score']:.4f}")
        logger.info(f"     Text: {result['text'][:100]}...")

def test_self_rag(retriever):
    """Test self-RAG with relevance evaluation"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SELF-RAG")
    logger.info("=" * 60)
    
    # Test 1: Query that needs retrieval
    logger.info("\n📝 Test 1: Query needing retrieval")
    query1 = "What documents can the system handle?"
    
    results1, needs_retrieval1 = retriever.self_rag_retrieval(
        query=query1,
        k=5,
        relevance_threshold=0.5
    )
    
    logger.info(f"  Query: '{query1}'")
    logger.info(f"  Needs Retrieval: {needs_retrieval1}")
    logger.info(f"  Relevant Results: {len(results1)}")
    
    # Test 2: Query that might not need retrieval
    logger.info("\n📝 Test 2: General knowledge query")
    query2 = "What is 2+2?"
    
    results2, needs_retrieval2 = retriever.self_rag_retrieval(
        query=query2,
        k=5,
        relevance_threshold=0.5
    )
    
    logger.info(f"  Query: '{query2}'")
    logger.info(f"  Needs Retrieval: {needs_retrieval2}")
    logger.info(f"  Relevant Results: {len(results2)}")
    
    logger.success("\n✓ Self-RAG test completed")

def test_hybrid_retrieval(retriever):
    """Test hybrid semantic + keyword retrieval"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING HYBRID RETRIEVAL")
    logger.info("=" * 60)
    
    query = "document processing chunks"
    
    logger.info(f"\nQuery: '{query}'")
    logger.info("Combining semantic and keyword search...")
    
    results = retriever.hybrid_retrieval(
        query=query,
        k=5,
        semantic_weight=0.7
    )
    
    logger.success(f"\n✓ Hybrid retrieval returned {len(results)} results")
    
    logger.info("\n📊 Results with Combined Scores:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n  {i}. Hybrid Score: {result['hybrid_score']:.4f}")
        logger.info(f"     - Semantic: {result['semantic_score']:.4f}")
        logger.info(f"     - Keyword: {result['keyword_score']:.4f}")
        logger.info(f"     Text: {result['text'][:100]}...")

def test_contextual_compression(retriever):
    """Test contextual compression"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING CONTEXTUAL COMPRESSION")
    logger.info("=" * 60)
    
    query = "What file types are supported?"
    
    # Get initial results
    query_embedding = retriever.embeddings_generator.generate_embedding(query)
    results = retriever.vector_store.search(query_embedding, k=3)
    
    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Retrieved {len(results)} documents")
    logger.info("\nOriginal lengths:")
    for i, r in enumerate(results, 1):
        logger.info(f"  Doc {i}: {len(r['text'])} chars")
    
    # Apply compression
    logger.info("\nApplying contextual compression...")
    compressed = retriever.contextual_compression(query, results, max_tokens=200)
    
    logger.success("\n✓ Compression complete")
    logger.info("\nCompression Results:")
    for i, r in enumerate(compressed, 1):
        ratio = r['compression_ratio']
        logger.info(f"\n  Doc {i}:")
        logger.info(f"    Original: {len(r['original_text'])} chars")
        logger.info(f"    Compressed: {len(r['compressed_text'])} chars")
        logger.info(f"    Ratio: {ratio:.2%}")
        logger.info(f"    Text: {r['compressed_text'][:150]}...")

def test_rag_pipeline(retriever):
    """Test complete RAG pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMPLETE RAG PIPELINE")
    logger.info("=" * 60)
    
    # Initialize pipeline
    logger.info("\n1. Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_api_key=settings.OPENAI_API_KEY,
        llm_model="gpt-4o-mini",
        temperature=0.7
    )
    logger.success("✓ Pipeline ready")
    
    # Test questions
    test_questions = [
        "What types of documents can the system process?",
        "How are documents split into chunks?",
        "What is the purpose of embeddings?"
    ]
    
    logger.info(f"\n2. Testing with {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Question {i}: {question}")
        logger.info(f"{'='*50}")
        
        # Test with multi-query retrieval
        response = pipeline.query(
            question=question,
            retrieval_method="multi_query",
            k=3,
            include_sources=True
        )
        
        logger.info(f"\n💡 Answer:")
        logger.info(f"{response['answer']}\n")
        
        logger.info(f"📊 Metadata:")
        logger.info(f"  - Sources used: {response['num_sources']}")
        logger.info(f"  - Response time: {response['response_time']:.2f}s")
        logger.info(f"  - Method: {response['retrieval_method']}")
        
        if response.get('sources'):
            logger.info(f"\n📚 Top Sources:")
            for j, source in enumerate(response['sources'], 1):
                logger.info(f"\n  {j}. {source['file_name']} (similarity: {source['similarity_score']:.4f})")
                logger.info(f"     {source['text']}")
    
    logger.success("\n✓ RAG pipeline test completed")
    
    return pipeline

def test_chat_history(pipeline):
    """Test RAG with conversation history"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING CHAT HISTORY")
    logger.info("=" * 60)
    
    # Simulate conversation
    chat_history = [
        {"role": "user", "content": "What file types can the system handle?"},
        {"role": "assistant", "content": "The system can handle PDF, Word, Excel, CSV, and text files."}
    ]
    
    # Follow-up question (uses "it" referring to previous context)
    follow_up = "How does it process them?"
    
    logger.info("\n📝 Conversation History:")
    for msg in chat_history:
        logger.info(f"  {msg['role'].capitalize()}: {msg['content']}")
    
    logger.info(f"\n❓ Follow-up Question: {follow_up}")
    
    logger.info("\nContextualizing question...")
    response = pipeline.query_with_chat_history(
        question=follow_up,
        chat_history=chat_history,
        retrieval_method="multi_query",
        k=3
    )
    
    logger.info(f"\n🔄 Contextualized Question:")
    logger.info(f"  {response['contextualized_question']}")
    
    logger.info(f"\n💡 Answer:")
    logger.info(f"  {response['answer']}")
    
    logger.success("\n✓ Chat history test completed")

def test_end_to_end():
    """Complete Phase 3 end-to-end test"""
    logger.info("\n" + "=" * 60)
    logger.info("END-TO-END PHASE 3 TEST")
    logger.info("=" * 60)
    
    logger.info("\n📋 Testing Advanced RAG Features:")
    logger.info("  1. Basic Retrieval")
    logger.info("  2. Multi-Query Retrieval")
    logger.info("  3. Self-RAG")
    logger.info("  4. Hybrid Retrieval")
    logger.info("  5. Contextual Compression")
    logger.info("  6. Complete RAG Pipeline")
    logger.info("  7. Chat History")
    
    try:
        # Test 1: Basic Retrieval
        vector_store, retriever = test_basic_retrieval()
        if retriever is None:
            return
        
        # Test 2: Multi-Query
        test_multi_query_retrieval(retriever)
        
        # Test 3: Self-RAG
        test_self_rag(retriever)
        
        # Test 4: Hybrid
        test_hybrid_retrieval(retriever)
        
        # Test 5: Compression
        test_contextual_compression(retriever)
        
        # Test 6: Pipeline
        pipeline = test_rag_pipeline(retriever)
        
        # Test 7: Chat History
        test_chat_history(pipeline)
        
        logger.info("\n" + "🎉" * 30)
        logger.success("ALL PHASE 3 TESTS PASSED!")
        logger.info("🎉" * 30)
        
        logger.info("\n✅ Phase 3 Complete! You now have:")
        logger.info("  ✓ Multi-query retrieval")
        logger.info("  ✓ Self-RAG with relevance evaluation")
        logger.info("  ✓ Hybrid semantic + keyword search")
        logger.info("  ✓ Contextual compression")
        logger.info("  ✓ Complete RAG pipeline")
        logger.info("  ✓ Chat history support")
        
        logger.info("\n🚀 Ready for Phase 4: LLM Integration & Fine-tuning!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run all tests"""
    logger.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     ENTERPRISE AI ASSISTANT - PHASE 3 TEST SUITE        ║
    ║                                                           ║
    ║  Testing: Advanced RAG Implementation                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    test_end_to_end()

if __name__ == "__main__":
    main()
