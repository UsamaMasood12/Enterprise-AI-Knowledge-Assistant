"""
API Endpoints Test Suite
Comprehensive testing of all Enterprise AI Assistant API endpoints
"""

import requests
import json
from loguru import logger
import sys
from typing import Dict, Any
import time

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# API Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30


class APITester:
    """Test all API endpoints"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
    
    def test_endpoint(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=TIMEOUT, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=TIMEOUT, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def log_result(self, test_name: str, result: Dict[str, Any], expected_status: int = 200):
        """Log test result"""
        if result.get("success") and result.get("status_code") == expected_status:
            logger.success(f"✓ {test_name}")
            logger.info(f"  Status: {result['status_code']} | Time: {result.get('response_time', 0):.2f}s")
            self.passed += 1
            return True
        else:
            logger.error(f"✗ {test_name}")
            if result.get("error"):
                logger.error(f"  Error: {result['error']}")
            else:
                logger.error(f"  Status: {result.get('status_code')} (expected {expected_status})")
            self.failed += 1
            return False
    
    def test_health(self):
        """Test health endpoint"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: Health Check")
        logger.info("=" * 60)
        
        result = self.test_endpoint("GET", "/health")
        
        if self.log_result("Health Check", result):
            data = result['data']
            logger.info(f"  Version: {data.get('version')}")
            logger.info(f"  Status: {data.get('status')}")
            logger.info(f"  Components: {data.get('components')}")
    
    def test_root(self):
        """Test root endpoint"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Root Endpoint")
        logger.info("=" * 60)
        
        result = self.test_endpoint("GET", "/")
        self.log_result("Root Endpoint", result)
    
    def test_query_simple(self):
        """Test simple query endpoint"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Query Endpoint (Simple)")
        logger.info("=" * 60)
        
        payload = {
            "question": "What is machine learning?",
            "retrieval_method": "simple",
            "k": 3,
            "optimize_for": "balanced",
            "include_sources": True
        }
        
        logger.info(f"Question: {payload['question']}")
        
        result = self.test_endpoint("POST", "/query", json=payload)
        
        if self.log_result("Simple Query", result):
            data = result['data']
            logger.info(f"  Answer: {data.get('answer', '')[:100]}...")
            logger.info(f"  LLM Provider: {data.get('llm_provider')}")
            logger.info(f"  Sources: {data.get('num_sources')}")
            logger.info(f"  Cost: ${data.get('estimated_cost', 0):.4f}")
    
    def test_nlp_sentiment(self):
        """Test sentiment analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: NLP - Sentiment Analysis")
        logger.info("=" * 60)
        
        test_texts = [
            "This is absolutely amazing! I love it!",
            "Terrible experience. Very disappointed.",
            "It works as expected."
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n  Test {i}: \"{text}\"")
            
            result = self.test_endpoint("POST", "/nlp/sentiment", json={"text": text})
            
            if result.get("success"):
                sentiment = result['data'].get('sentiment', {})
                logger.info(f"    Sentiment: {sentiment.get('label')} ({sentiment.get('score', 0):.2f})")
                self.passed += 1
            else:
                logger.error(f"    Failed: {result.get('error')}")
                self.failed += 1
    
    def test_nlp_ner(self):
        """Test named entity recognition"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: NLP - Named Entity Recognition")
        logger.info("=" * 60)
        
        text = "John Smith works at Microsoft in Seattle and earns $100,000 per year."
        logger.info(f"Text: {text}")
        
        result = self.test_endpoint("POST", "/nlp/ner", json={"text": text})
        
        if self.log_result("Named Entity Recognition", result):
            entities = result['data'].get('entities', {})
            for entity_type, items in entities.items():
                if items:
                    logger.info(f"  {entity_type}: {items}")
    
    def test_nlp_classification(self):
        """Test text classification"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: NLP - Text Classification")
        logger.info("=" * 60)
        
        text = "The new AI model uses transformer architecture for natural language processing."
        categories = ["Technology", "Finance", "Healthcare", "Education"]
        
        logger.info(f"Text: {text}")
        logger.info(f"Categories: {categories}")
        
        result = self.test_endpoint(
            "POST", 
            "/nlp/classify",
            json={"text": text, "categories": categories}
        )
        
        if self.log_result("Text Classification", result):
            classification = result['data'].get('classification', {})
            logger.info(f"  Category: {classification.get('category')}")
            logger.info(f"  Confidence: {classification.get('confidence', 0):.2f}")
    
    def test_nlp_summarization(self):
        """Test text summarization"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: NLP - Text Summarization")
        logger.info("=" * 60)
        
        long_text = """
        Artificial Intelligence has revolutionized many industries in recent years.
        Machine learning algorithms can now process vast amounts of data and identify
        patterns that humans might miss. Deep learning uses neural networks with
        multiple layers to learn complex representations. Natural Language Processing
        enables computers to understand and generate human language. Applications
        include chatbots, translation services, and sentiment analysis.
        """
        
        logger.info(f"Original length: {len(long_text.split())} words")
        
        result = self.test_endpoint(
            "POST",
            "/nlp/summarize",
            json={"text": long_text, "max_length": 50}
        )
        
        if self.log_result("Text Summarization", result):
            summary_data = result['data'].get('summary', {})
            logger.info(f"  Summary: {summary_data.get('summary', '')[:150]}...")
            logger.info(f"  Compression: {summary_data.get('compression_ratio', 0):.1%}")
    
    def test_analytics_summary(self):
        """Test analytics summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 8: Analytics Summary")
        logger.info("=" * 60)
        
        result = self.test_endpoint("GET", "/analytics/summary")
        
        if self.log_result("Analytics Summary", result):
            data = result['data']
            query_analytics = data.get('query_analytics', {})
            if 'total_queries' in query_analytics:
                logger.info(f"  Total Queries: {query_analytics.get('total_queries')}")
                logger.info(f"  Avg Response Time: {query_analytics.get('avg_response_time', 0):.2f}s")
            else:
                logger.info("  No queries tracked yet")
    
    def test_analytics_cost(self):
        """Test cost report"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 9: Cost Report")
        logger.info("=" * 60)
        
        result = self.test_endpoint("GET", "/analytics/cost?days=7")
        
        if self.log_result("Cost Report", result):
            data = result['data']
            if 'total_cost' in data:
                logger.info(f"  Total Cost: ${data.get('total_cost', 0):.4f}")
                logger.info(f"  API Calls: {data.get('total_api_calls', 0)}")
            else:
                logger.info("  No cost data available")
    
    def test_query_multi_query(self):
        """Test multi-query retrieval"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 10: Query with Multi-Query Retrieval")
        logger.info("=" * 60)
        
        payload = {
            "question": "How does the RAG system work?",
            "retrieval_method": "multi_query",
            "k": 5,
            "optimize_for": "quality"
        }
        
        logger.info(f"Question: {payload['question']}")
        
        result = self.test_endpoint("POST", "/query", json=payload)
        
        if self.log_result("Multi-Query Retrieval", result):
            data = result['data']
            logger.info(f"  Method: {payload['retrieval_method']}")
            logger.info(f"  Optimization: {payload['optimize_for']}")
    
    def run_all_tests(self):
        """Run all API tests"""
        logger.info("=" * 60)
        logger.info("ENTERPRISE AI ASSISTANT - API TEST SUITE")
        logger.info("=" * 60)
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Timeout: {TIMEOUT}s")
        
        start_time = time.time()
        
        # Run all tests
        self.test_health()
        self.test_root()
        self.test_query_simple()
        self.test_nlp_sentiment()
        self.test_nlp_ner()
        self.test_nlp_classification()
        self.test_nlp_summarization()
        self.test_analytics_summary()
        self.test_analytics_cost()
        self.test_query_multi_query()
        
        # Summary
        total_time = time.time() - start_time
        total_tests = self.passed + self.failed
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.success(f"Passed: {self.passed}")
        if self.failed > 0:
            logger.error(f"Failed: {self.failed}")
        else:
            logger.success(f"Failed: {self.failed}")
        logger.info(f"Success Rate: {(self.passed/total_tests*100):.1f}%")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info("=" * 60)
        
        if self.failed == 0:
            logger.success("\n🎉 ALL TESTS PASSED! 🎉\n")
        else:
            logger.warning(f"\n⚠️  {self.failed} test(s) failed. Check logs above.\n")
        
        return self.failed == 0


def main():
    """Main test runner"""
    logger.info("Starting API tests...")
    logger.info("Make sure the API is running at http://localhost:8000\n")
    
    # Wait a moment for user
    time.sleep(1)
    
    tester = APITester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()