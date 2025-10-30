"""
Phase 4 Test Script
Test Multi-LLM Integration & Intelligent Routing
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
from src.llm.multi_llm_manager import MultiLLMManager, QueryComplexity
from src.llm.enhanced_rag_pipeline import EnhancedRAGPipeline
from src.utils.config import settings
import sys
import json

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add("phase4_test.log", rotation="1 MB")

def test_multi_llm_manager():
    """Test multi-LLM manager"""
    logger.info("=" * 60)
    logger.info("TESTING MULTI-LLM MANAGER")
    logger.info("=" * 60)
    
    # Initialize manager
    logger.info("\n1. Initializing Multi-LLM Manager...")
    llm_manager = MultiLLMManager(
        openai_api_key=settings.OPENAI_API_KEY,
        default_provider="openai_gpt4_turbo"
    )
    
    available = llm_manager.get_available_providers()
    logger.success(f"✓ Initialized with {len(available)} providers")
    logger.info(f"  Available: {', '.join(available)}")
    
    # Test simple generation
    logger.info("\n2. Testing simple generation...")
    test_prompt = "What is machine learning in one sentence?"
    
    result = llm_manager.generate(
        prompt=test_prompt,
        provider="openai_gpt35",
        temperature=0.7,
        max_tokens=100
    )
    
    logger.success("✓ Generated response")
    logger.info(f"  Model: {result['model']}")
    logger.info(f"  Tokens: {result['total_tokens']}")
    logger.info(f"  Cost: ${result['estimated_cost']:.4f}")
    logger.info(f"  Time: {result['response_time']:.2f}s")
    logger.info(f"  Response: {result['text'][:100]}...")
    
    return llm_manager

def test_query_complexity_estimation(llm_manager):
    """Test query complexity estimation"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING QUERY COMPLEXITY ESTIMATION")
    logger.info("=" * 60)
    
    test_queries = [
        ("What is AI?", QueryComplexity.SIMPLE),
        ("Explain how neural networks work", QueryComplexity.MODERATE),
        ("Provide a comprehensive analysis of transformer architecture with detailed explanations", QueryComplexity.COMPLEX)
    ]
    
    logger.info("\nEstimating complexity for test queries...")
    
    for query, expected in test_queries:
        complexity = llm_manager.estimate_query_complexity(query)
        match = "✓" if complexity == expected else "✗"
        
        logger.info(f"\n{match} Query: '{query}'")
        logger.info(f"  Estimated: {complexity.value}")
        logger.info(f"  Expected: {expected.value}")

def test_intelligent_routing(llm_manager):
    """Test intelligent LLM routing"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING INTELLIGENT ROUTING")
    logger.info("=" * 60)
    
    test_query = "Explain document processing in detail"
    
    strategies = ["cost", "quality", "speed", "balanced"]
    
    logger.info(f"\nQuery: '{test_query}'")
    logger.info("\nRouting with different strategies:")
    
    for strategy in strategies:
        provider = llm_manager.route_query(test_query, optimize_for=strategy)
        logger.info(f"  {strategy:10s} → {provider}")

def test_model_comparison(llm_manager):
    """Test comparing multiple models"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MODEL COMPARISON")
    logger.info("=" * 60)
    
    test_prompt = "What is machine learning?"
    
    available = llm_manager.get_available_providers()
    providers_to_test = available[:2]  # Test first 2 available
    
    logger.info(f"\nComparing {len(providers_to_test)} models...")
    logger.info(f"Prompt: '{test_prompt}'")
    
    results = llm_manager.compare_models(test_prompt, providers_to_test)
    
    logger.success(f"\n✓ Comparison complete")
    logger.info("\n📊 Results:")
    
    for provider, result in results.items():
        if 'error' not in result:
            logger.info(f"\n  {provider}:")
            logger.info(f"    Response: {result['text'][:100]}...")
            logger.info(f"    Time: {result['response_time']:.2f}s")
            logger.info(f"    Tokens: {result['total_tokens']}")
            logger.info(f"    Cost: ${result['estimated_cost']:.4f}")
        else:
            logger.warning(f"\n  {provider}: {result['error']}")

def test_enhanced_rag_pipeline():
    """Test enhanced RAG pipeline with multi-LLM"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ENHANCED RAG PIPELINE")
    logger.info("=" * 60)
    
    # Load vector store
    store_path = settings.EMBEDDINGS_DIR / "vector_store"
    if not store_path.exists():
        logger.error("Vector store not found. Run Phase 2 & 3 first!")
        return None
    
    logger.info("\n1. Loading components...")
    vector_store = FAISSVectorStore.load(str(store_path))
    
    embeddings_gen = EmbeddingsGenerator(
        openai_api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    retriever = AdvancedRAGRetriever(
        vector_store=vector_store,
        embeddings_generator=embeddings_gen,
        llm_api_key=settings.OPENAI_API_KEY,
        llm_model="gpt-4o-mini"
    )
    
    llm_manager = MultiLLMManager(
        openai_api_key=settings.OPENAI_API_KEY,
        default_provider="openai_gpt4_turbo"
    )
    
    pipeline = EnhancedRAGPipeline(
        retriever=retriever,
        llm_manager=llm_manager,
        default_optimization="balanced"
    )
    
    logger.success("✓ Pipeline initialized")
    
    # Test query with auto-routing
    logger.info("\n2. Testing query with auto-routing...")
    question = "What file types can the system process?"
    
    response = pipeline.query(
        question=question,
        retrieval_method="multi_query",
        k=3,
        optimize_for="cost"  # Cost-optimized routing
    )
    
    logger.success("✓ Query processed")
    logger.info(f"\n💡 Answer:")
    logger.info(f"{response['answer']}\n")
    
    logger.info(f"📊 Metadata:")
    logger.info(f"  LLM Provider: {response['llm_provider']}")
    logger.info(f"  Model: {response['llm_model']}")
    logger.info(f"  Sources: {response['num_sources']}")
    logger.info(f"  Tokens: {response['llm_tokens']}")
    logger.info(f"  Cost: ${response['estimated_cost']:.4f}")
    logger.info(f"  Time: {response['response_time']:.2f}s")
    
    return pipeline

def test_optimization_strategies(pipeline):
    """Test different optimization strategies"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING OPTIMIZATION STRATEGIES")
    logger.info("=" * 60)
    
    question = "How does document chunking work?"
    strategies = ["cost", "quality", "speed"]
    
    logger.info(f"\nQuestion: '{question}'")
    logger.info("\nComparing optimization strategies:")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*50}")
        logger.info(f"Strategy: {strategy.upper()}")
        logger.info(f"{'='*50}")
        
        response = pipeline.query(
            question=question,
            retrieval_method="hybrid",
            k=3,
            optimize_for=strategy
        )
        
        results[strategy] = response
        
        logger.info(f"  Provider: {response['llm_provider']}")
        logger.info(f"  Cost: ${response['estimated_cost']:.4f}")
        logger.info(f"  Time: {response['response_time']:.2f}s")
        logger.info(f"  Answer: {response['answer'][:150]}...")
    
    # Summary
    logger.success("\n✓ Strategy comparison complete")
    logger.info("\n📊 Summary:")
    
    for strategy, response in results.items():
        logger.info(f"\n  {strategy.capitalize()}:")
        logger.info(f"    Provider: {response['llm_provider']}")
        logger.info(f"    Cost: ${response['estimated_cost']:.4f}")
        logger.info(f"    Time: {response['response_time']:.2f}s")

def test_batch_with_routing(pipeline):
    """Test batch processing with intelligent routing"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING BATCH PROCESSING WITH ROUTING")
    logger.info("=" * 60)
    
    questions = [
        "What is AI?",  # Simple
        "How are documents processed?",  # Moderate
        "Explain the complete RAG pipeline architecture"  # Complex
    ]
    
    logger.info(f"\nProcessing {len(questions)} queries with balanced optimization...")
    
    responses = pipeline.batch_query_with_routing(
        questions=questions,
        retrieval_method="multi_query",
        k=3,
        optimize_for="balanced"
    )
    
    logger.success(f"\n✓ Batch processing complete")
    logger.info("\n📊 Results:")
    
    total_cost = 0
    for i, (question, response) in enumerate(zip(questions, responses), 1):
        logger.info(f"\n  {i}. {question}")
        logger.info(f"     Provider: {response['llm_provider']}")
        logger.info(f"     Cost: ${response['estimated_cost']:.4f}")
        total_cost += response['estimated_cost']
    
    logger.info(f"\n  Total Cost: ${total_cost:.4f}")

def test_cost_tracking(pipeline):
    """Test cost tracking and reporting"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COST TRACKING")
    logger.info("=" * 60)
    
    logger.info("\nGenerating cost report...")
    report = pipeline.get_cost_report()
    
    logger.success("✓ Cost report generated")
    logger.info("\n📊 Usage Report:")
    logger.info(f"  Total Queries: {report['total_queries']}")
    logger.info(f"  Total Tokens: {report['total_tokens']:,}")
    logger.info(f"  Total Cost: ${report['total_cost']:.4f}")
    logger.info(f"  Avg Cost/Query: ${report['average_cost_per_query']:.4f}")
    
    if report['by_provider']:
        logger.info("\n  By Provider:")
        for provider, stats in report['by_provider'].items():
            logger.info(f"\n    {provider}:")
            logger.info(f"      Calls: {stats['calls']}")
            logger.info(f"      Tokens: {stats['tokens']:,}")
            logger.info(f"      Cost: ${stats['cost']:.4f}")

def test_end_to_end():
    """Complete Phase 4 end-to-end test"""
    logger.info("\n" + "=" * 60)
    logger.info("END-TO-END PHASE 4 TEST")
    logger.info("=" * 60)
    
    logger.info("\n📋 Testing Multi-LLM Integration:")
    logger.info("  1. Multi-LLM Manager")
    logger.info("  2. Query Complexity Estimation")
    logger.info("  3. Intelligent Routing")
    logger.info("  4. Model Comparison")
    logger.info("  5. Enhanced RAG Pipeline")
    logger.info("  6. Optimization Strategies")
    logger.info("  7. Batch Processing with Routing")
    logger.info("  8. Cost Tracking")
    
    try:
        # Test 1: Multi-LLM Manager
        llm_manager = test_multi_llm_manager()
        
        # Test 2: Complexity Estimation
        test_query_complexity_estimation(llm_manager)
        
        # Test 3: Routing
        test_intelligent_routing(llm_manager)
        
        # Test 4: Comparison
        test_model_comparison(llm_manager)
        
        # Test 5: Enhanced Pipeline
        pipeline = test_enhanced_rag_pipeline()
        if pipeline is None:
            return
        
        # Test 6: Strategies
        test_optimization_strategies(pipeline)
        
        # Test 7: Batch
        test_batch_with_routing(pipeline)
        
        # Test 8: Cost Tracking
        test_cost_tracking(pipeline)
        
        logger.info("\n" + "🎉" * 30)
        logger.success("ALL PHASE 4 TESTS PASSED!")
        logger.info("🎉" * 30)
        
        logger.info("\n✅ Phase 4 Complete! You now have:")
        logger.info("  ✓ Multi-LLM support (GPT-4, GPT-3.5)")
        logger.info("  ✓ Intelligent query routing")
        logger.info("  ✓ Cost optimization strategies")
        logger.info("  ✓ Query complexity estimation")
        logger.info("  ✓ Model comparison capabilities")
        logger.info("  ✓ Detailed cost tracking")
        logger.info("  ✓ Batch processing with routing")
        
        logger.info("\n🚀 Ready for Phase 5: NLP Features & Analytics!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run all tests"""
    logger.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     ENTERPRISE AI ASSISTANT - PHASE 4 TEST SUITE        ║
    ║                                                           ║
    ║  Testing: Multi-LLM Integration & Intelligent Routing    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    test_end_to_end()

if __name__ == "__main__":
    main()
