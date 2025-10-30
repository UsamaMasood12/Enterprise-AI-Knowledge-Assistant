"""
Phase 5 Test Script
Test NLP Features & Analytics
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pathlib import Path
from loguru import logger
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.analytics.analytics_tracker import AnalyticsTracker
from src.utils.config import settings
import sys

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add("phase5_test.log", rotation="1 MB")

def test_named_entity_recognition():
    """Test NER functionality"""
    logger.info("=" * 60)
    logger.info("TESTING NAMED ENTITY RECOGNITION")
    logger.info("=" * 60)
    
    # Initialize analyzer
    logger.info("\n1. Initializing NLP Analyzer...")
    analyzer = NLPAnalyzer(openai_api_key=settings.OPENAI_API_KEY)
    logger.success("✓ NLP Analyzer initialized")
    
    # Test text
    test_text = """
    John Smith works at Microsoft in Seattle. He joined the company in January 2020
    and has been working on Azure AI projects. The team recently launched a new product
    worth $10 million that will be presented at the Tech Summit in New York.
    """
    
    logger.info("\n2. Extracting entities...")
    logger.info(f"Text: {test_text.strip()}")
    
    # Test with LLM
    entities = analyzer.named_entity_recognition(test_text)
    
    logger.success(f"\n✓ Found entities:")
    for entity_type, items in entities.items():
        if items:
            logger.info(f"\n  {entity_type}:")
            for item in items:
                if isinstance(item, dict):
                    logger.info(f"    - {item['text']}")
                else:
                    logger.info(f"    - {item}")
    
    return analyzer

def test_sentiment_analysis(analyzer):
    """Test sentiment analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SENTIMENT ANALYSIS")
    logger.info("=" * 60)
    
    test_texts = [
        "This product is absolutely amazing! I love it!",
        "Terrible experience. Very disappointed with the service.",
        "The system works as expected. Nothing special."
    ]
    
    logger.info("\nAnalyzing sentiments...")
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n{i}. Text: \"{text}\"")
        
        sentiment = analyzer.sentiment_analysis(text)
        
        logger.info(f"   Sentiment: {sentiment['label']}")
        logger.info(f"   Score: {sentiment['score']:.2f}")
        if 'explanation' in sentiment:
            logger.info(f"   Explanation: {sentiment['explanation']}")
    
    logger.success("\n✓ Sentiment analysis complete")

def test_text_classification(analyzer):
    """Test text classification"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING TEXT CLASSIFICATION")
    logger.info("=" * 60)
    
    categories = ["Technology", "Finance", "Healthcare", "Education", "Entertainment"]
    
    test_texts = [
        "The new AI model uses transformer architecture for natural language processing.",
        "Stock market reached record highs with tech stocks leading the gains.",
        "Doctors recommend regular exercise and a balanced diet for good health."
    ]
    
    logger.info(f"\nCategories: {', '.join(categories)}")
    logger.info("\nClassifying texts...")
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n{i}. Text: \"{text}\"")
        
        result = analyzer.text_classification(text, categories)
        
        logger.info(f"   Category: {result['category']}")
        logger.info(f"   Confidence: {result['confidence']:.2f}")
        if 'reasoning' in result:
            logger.info(f"   Reasoning: {result['reasoning']}")
    
    logger.success("\n✓ Text classification complete")

def test_text_summarization(analyzer):
    """Test text summarization"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING TEXT SUMMARIZATION")
    logger.info("=" * 60)
    
    long_text = """
    Artificial Intelligence (AI) has revolutionized many industries in recent years.
    Machine learning algorithms can now process vast amounts of data and identify
    patterns that humans might miss. Deep learning, a subset of machine learning,
    uses neural networks with multiple layers to learn complex representations.
    
    Natural Language Processing (NLP) is one of the most impactful applications of AI.
    It enables computers to understand, interpret, and generate human language.
    Applications include chatbots, translation services, sentiment analysis, and
    text summarization. Recent advances in transformer models like GPT and BERT
    have dramatically improved NLP capabilities.
    
    The future of AI looks promising with continued research in areas like
    reinforcement learning, computer vision, and autonomous systems. However,
    ethical considerations around bias, privacy, and job displacement remain
    important challenges that need to be addressed.
    """
    
    logger.info("\n1. Original text:")
    logger.info(f"   Length: {len(long_text.split())} words")
    logger.info(f"   Preview: {long_text[:150]}...")
    
    logger.info("\n2. Generating summary...")
    result = analyzer.summarize_text(long_text, max_length=80)
    
    logger.success("\n✓ Summary generated:")
    logger.info(f"\n{result['summary']}\n")
    
    logger.info(f"Stats:")
    logger.info(f"  Original: {result['original_length']} words")
    logger.info(f"  Summary: {result['summary_length']} words")
    logger.info(f"  Compression: {result['compression_ratio']:.1%}")
    logger.info(f"  Method: {result['method']}")

def test_document_analysis(analyzer):
    """Test comprehensive document analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DOCUMENT ANALYSIS")
    logger.info("=" * 60)
    
    document = """
    Apple Inc. announced record quarterly earnings today, with CEO Tim Cook
    praising the team's innovation. The company's revenue reached $90 billion,
    driven by strong iPhone and Mac sales. Investors reacted positively, with
    stock prices rising 5% in after-hours trading. The Cupertino-based tech
    giant plans to expand its AI capabilities in 2024.
    """
    
    logger.info("\nAnalyzing document...")
    logger.info(f"Document: {document.strip()}\n")
    
    analysis = analyzer.analyze_document(document)
    
    logger.success("✓ Analysis complete\n")
    
    # Text Stats
    logger.info("📊 TEXT STATISTICS:")
    stats = analysis['text_stats']
    logger.info(f"  Words: {stats['word_count']}")
    logger.info(f"  Sentences: {stats['sentence_count']}")
    logger.info(f"  Avg word length: {stats['avg_word_length']:.1f}")
    
    # Entities
    logger.info("\n🏷️ ENTITIES:")
    for entity_type, items in analysis['entities'].items():
        if items:
            logger.info(f"  {entity_type}: {', '.join([str(i) if isinstance(i, str) else i['text'] for i in items[:3]])}")
    
    # Sentiment
    logger.info("\n😊 SENTIMENT:")
    sent = analysis['sentiment']
    logger.info(f"  {sent['label']} ({sent['score']:.2f})")
    
    # Summary
    logger.info("\n📝 SUMMARY:")
    logger.info(f"  {analysis['summary']['summary']}")

def test_analytics_tracker():
    """Test analytics tracking"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ANALYTICS TRACKER")
    logger.info("=" * 60)
    
    # Initialize tracker
    logger.info("\n1. Initializing Analytics Tracker...")
    tracker = AnalyticsTracker(save_dir="data/analytics")
    logger.success("✓ Tracker initialized")
    
    # Simulate some queries
    logger.info("\n2. Simulating queries...")
    
    mock_queries = [
        {
            'query': 'What is machine learning?',
            'response': {
                'answer': 'Machine learning is...',
                'retrieval_method': 'multi_query',
                'num_sources': 5,
                'response_time': 1.2,
                'llm_provider': 'openai_gpt35',
                'llm_tokens': 150,
                'estimated_cost': 0.0003
            }
        },
        {
            'query': 'How does RAG work?',
            'response': {
                'answer': 'RAG combines retrieval...',
                'retrieval_method': 'hybrid',
                'num_sources': 3,
                'response_time': 0.8,
                'llm_provider': 'openai_gpt4_turbo',
                'llm_tokens': 200,
                'estimated_cost': 0.002
            }
        },
        {
            'query': 'What file types are supported?',
            'response': {
                'answer': 'Supported types include...',
                'retrieval_method': 'simple',
                'num_sources': 2,
                'response_time': 0.5,
                'llm_provider': 'openai_gpt35',
                'llm_tokens': 100,
                'estimated_cost': 0.0002
            }
        }
    ]
    
    for mock in mock_queries:
        tracker.track_query(mock['query'], mock['response'], user_feedback=4.5)
    
    logger.success(f"✓ Tracked {len(mock_queries)} queries")
    
    # Get analytics
    logger.info("\n3. Generating analytics...")
    
    query_analytics = tracker.get_query_analytics(days=7)
    
    logger.success("\n✓ Query Analytics:")
    logger.info(f"  Total Queries: {query_analytics['total_queries']}")
    logger.info(f"  Avg Response Time: {query_analytics['avg_response_time']:.2f}s")
    logger.info(f"  Total Cost: ${query_analytics['total_cost']:.4f}")
    logger.info(f"  Avg Sources: {query_analytics['avg_sources_used']:.1f}")
    
    # Popular queries
    logger.info("\n4. Popular queries...")
    popular = tracker.get_popular_queries(top_n=3)
    
    for i, q in enumerate(popular, 1):
        logger.info(f"  {i}. \"{q['query']}\" ({q['count']} times)")
    
    return tracker

def test_ragas_metrics(tracker):
    """Test RAGAS evaluation metrics"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING RAGAS METRICS")
    logger.info("=" * 60)
    
    # Example RAG output
    query = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    contexts = [
        "Machine learning is a subset of AI that allows computers to learn without being explicitly programmed.",
        "Artificial intelligence encompasses various techniques including machine learning and deep learning.",
        "Data is the foundation of machine learning algorithms."
    ]
    ground_truth = "Machine learning is an AI technique that learns from data."
    
    logger.info(f"\nQuery: {query}")
    logger.info(f"Answer: {answer}")
    logger.info(f"Contexts: {len(contexts)} documents")
    
    logger.info("\nCalculating RAGAS metrics...")
    metrics = tracker.calculate_ragas_metrics(query, answer, contexts, ground_truth)
    
    logger.success("\n✓ RAGAS Metrics:")
    logger.info(f"  Context Relevance: {metrics['context_relevance']:.2f}")
    logger.info(f"  Answer Relevance: {metrics['answer_relevance']:.2f}")
    logger.info(f"  Faithfulness: {metrics['faithfulness']:.2f}")
    logger.info(f"  Answer Correctness: {metrics['answer_correctness']:.2f}")

def test_analytics_report(tracker):
    """Test analytics report generation"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ANALYTICS REPORT")
    logger.info("=" * 60)
    
    logger.info("\nGenerating comprehensive report...")
    
    report = tracker.generate_report()
    
    logger.success("\n✓ Report generated:\n")
    print(report)
    
    # Save analytics
    logger.info("\nSaving analytics to file...")
    filepath = tracker.save_analytics("phase5_analytics.json")
    logger.success(f"✓ Saved to: {filepath}")

def test_end_to_end():
    """Complete Phase 5 end-to-end test"""
    logger.info("\n" + "=" * 60)
    logger.info("END-TO-END PHASE 5 TEST")
    logger.info("=" * 60)
    
    logger.info("\n📋 Testing NLP Features & Analytics:")
    logger.info("  1. Named Entity Recognition")
    logger.info("  2. Sentiment Analysis")
    logger.info("  3. Text Classification")
    logger.info("  4. Text Summarization")
    logger.info("  5. Document Analysis")
    logger.info("  6. Analytics Tracking")
    logger.info("  7. RAGAS Metrics")
    logger.info("  8. Analytics Report")
    
    try:
        # Test 1: NER
        analyzer = test_named_entity_recognition()
        
        # Test 2: Sentiment
        test_sentiment_analysis(analyzer)
        
        # Test 3: Classification
        test_text_classification(analyzer)
        
        # Test 4: Summarization
        test_text_summarization(analyzer)
        
        # Test 5: Document Analysis
        test_document_analysis(analyzer)
        
        # Test 6: Analytics
        tracker = test_analytics_tracker()
        
        # Test 7: RAGAS
        test_ragas_metrics(tracker)
        
        # Test 8: Report
        test_analytics_report(tracker)
        
        logger.info("\n" + "🎉" * 30)
        logger.success("ALL PHASE 5 TESTS PASSED!")
        logger.info("🎉" * 30)
        
        logger.info("\n✅ Phase 5 Complete! You now have:")
        logger.info("  ✓ Named Entity Recognition (NER)")
        logger.info("  ✓ Sentiment Analysis")
        logger.info("  ✓ Text Classification")
        logger.info("  ✓ Text Summarization")
        logger.info("  ✓ Document Analysis")
        logger.info("  ✓ Analytics Tracking")
        logger.info("  ✓ RAGAS Evaluation")
        logger.info("  ✓ Performance Reports")
        
        logger.info("\n🚀 Ready for Phase 6: AWS Deployment & API!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run all tests"""
    logger.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     ENTERPRISE AI ASSISTANT - PHASE 5 TEST SUITE        ║
    ║                                                           ║
    ║  Testing: NLP Features & Analytics                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    test_end_to_end()

if __name__ == "__main__":
    main()