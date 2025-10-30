"""
Analytics & Metrics Tracker
Track performance metrics, evaluate RAG quality, and generate analytics
"""

from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path


class AnalyticsTracker:
    """Track and analyze RAG system performance"""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize analytics tracker
        
        Args:
            save_dir: Directory to save analytics data
        """
        self.save_dir = Path(save_dir) if save_dir else Path("data/analytics")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.query_history = []
        self.retrieval_metrics = []
        self.llm_metrics = []
        self.cost_metrics = []
        
        logger.info(f"AnalyticsTracker initialized (save_dir: {self.save_dir})")
    
    def track_query(
        self,
        query: str,
        response: Dict,
        user_feedback: Optional[float] = None
    ):
        """
        Track a query and its response
        
        Args:
            query: User query
            response: Response dictionary from RAG pipeline
            user_feedback: Optional user feedback score (0-5)
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'query_length': len(query.split()),
            'retrieval_method': response.get('retrieval_method'),
            'num_sources': response.get('num_sources', 0),
            'response_time': response.get('response_time', 0),
            'llm_provider': response.get('llm_provider'),
            'llm_tokens': response.get('llm_tokens', 0),
            'estimated_cost': response.get('estimated_cost', 0),
            'answer_length': len(response.get('answer', '').split()),
            'user_feedback': user_feedback
        }
        
        self.query_history.append(record)
        logger.info(f"Tracked query: {len(self.query_history)} total")
    
    def track_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict],
        relevant_doc_ids: Optional[List[int]] = None
    ):
        """
        Track retrieval metrics
        
        Args:
            query: Query string
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: Ground truth relevant document IDs (for evaluation)
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'num_retrieved': len(retrieved_docs),
            'avg_similarity': sum(d.get('similarity_score', 0) for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
            'top_similarity': max((d.get('similarity_score', 0) for d in retrieved_docs), default=0)
        }
        
        # Calculate precision/recall if ground truth provided
        if relevant_doc_ids:
            retrieved_ids = [d.get('chunk_id') for d in retrieved_docs]
            
            true_positives = len(set(retrieved_ids) & set(relevant_doc_ids))
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
            recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            record['precision'] = precision
            record['recall'] = recall
            record['f1_score'] = f1
        
        self.retrieval_metrics.append(record)
    
    def track_llm_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        response_time: float
    ):
        """Track LLM API usage"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'provider': provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': cost,
            'response_time': response_time
        }
        
        self.llm_metrics.append(record)
        self.cost_metrics.append(cost)
    
    def get_query_analytics(self, days: int = 7) -> Dict:
        """
        Get query analytics for specified time period
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics
        """
        logger.info(f"Generating query analytics for last {days} days...")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_queries = [
            q for q in self.query_history
            if datetime.fromisoformat(q['timestamp']) > cutoff_date
        ]
        
        if not recent_queries:
            return {'error': 'No queries in time period'}
        
        analytics = {
            'period_days': days,
            'total_queries': len(recent_queries),
            'avg_response_time': sum(q['response_time'] for q in recent_queries) / len(recent_queries),
            'total_cost': sum(q['estimated_cost'] for q in recent_queries),
            'avg_cost_per_query': sum(q['estimated_cost'] for q in recent_queries) / len(recent_queries),
            'avg_sources_used': sum(q['num_sources'] for q in recent_queries) / len(recent_queries),
            'avg_query_length': sum(q['query_length'] for q in recent_queries) / len(recent_queries),
            'avg_answer_length': sum(q['answer_length'] for q in recent_queries) / len(recent_queries)
        }
        
        # Provider breakdown
        provider_stats = defaultdict(lambda: {'count': 0, 'cost': 0})
        for q in recent_queries:
            provider = q.get('llm_provider', 'unknown')
            provider_stats[provider]['count'] += 1
            provider_stats[provider]['cost'] += q['estimated_cost']
        
        analytics['by_provider'] = dict(provider_stats)
        
        # User feedback (if available)
        feedback_queries = [q for q in recent_queries if q.get('user_feedback')]
        if feedback_queries:
            analytics['avg_user_feedback'] = sum(q['user_feedback'] for q in feedback_queries) / len(feedback_queries)
            analytics['feedback_count'] = len(feedback_queries)
        
        logger.success("Query analytics generated")
        return analytics
    
    def get_retrieval_analytics(self) -> Dict:
        """Get retrieval performance analytics"""
        if not self.retrieval_metrics:
            return {'error': 'No retrieval data'}
        
        analytics = {
            'total_retrievals': len(self.retrieval_metrics),
            'avg_docs_retrieved': sum(r['num_retrieved'] for r in self.retrieval_metrics) / len(self.retrieval_metrics),
            'avg_similarity_score': sum(r['avg_similarity'] for r in self.retrieval_metrics) / len(self.retrieval_metrics)
        }
        
        # If we have precision/recall data
        metrics_with_eval = [r for r in self.retrieval_metrics if 'precision' in r]
        if metrics_with_eval:
            analytics['avg_precision'] = sum(r['precision'] for r in metrics_with_eval) / len(metrics_with_eval)
            analytics['avg_recall'] = sum(r['recall'] for r in metrics_with_eval) / len(metrics_with_eval)
            analytics['avg_f1_score'] = sum(r['f1_score'] for r in metrics_with_eval) / len(metrics_with_eval)
        
        return analytics
    
    def get_cost_report(self, days: int = 30) -> Dict:
        """
        Get detailed cost report
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Cost report dictionary
        """
        logger.info(f"Generating cost report for last {days} days...")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_llm = [
            m for m in self.llm_metrics
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        if not recent_llm:
            return {'error': 'No LLM usage in time period'}
        
        report = {
            'period_days': days,
            'total_api_calls': len(recent_llm),
            'total_cost': sum(m['cost'] for m in recent_llm),
            'total_tokens': sum(m['total_tokens'] for m in recent_llm),
            'avg_cost_per_call': sum(m['cost'] for m in recent_llm) / len(recent_llm),
            'avg_tokens_per_call': sum(m['total_tokens'] for m in recent_llm) / len(recent_llm)
        }
        
        # By provider
        provider_costs = defaultdict(lambda: {'calls': 0, 'cost': 0, 'tokens': 0})
        for m in recent_llm:
            provider = m['provider']
            provider_costs[provider]['calls'] += 1
            provider_costs[provider]['cost'] += m['cost']
            provider_costs[provider]['tokens'] += m['total_tokens']
        
        report['by_provider'] = dict(provider_costs)
        
        # By model
        model_costs = defaultdict(lambda: {'calls': 0, 'cost': 0})
        for m in recent_llm:
            model = m['model']
            model_costs[model]['calls'] += 1
            model_costs[model]['cost'] += m['cost']
        
        report['by_model'] = dict(model_costs)
        
        # Daily breakdown
        daily_costs = defaultdict(float)
        for m in recent_llm:
            date = datetime.fromisoformat(m['timestamp']).date().isoformat()
            daily_costs[date] += m['cost']
        
        report['daily_costs'] = dict(sorted(daily_costs.items()))
        
        logger.success("Cost report generated")
        return report
    
    def get_popular_queries(self, top_n: int = 10) -> List[Dict]:
        """Get most common queries"""
        query_counts = defaultdict(int)
        query_info = {}
        
        for q in self.query_history:
            query_text = q['query'].lower()
            query_counts[query_text] += 1
            query_info[query_text] = q
        
        popular = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [
            {
                'query': query,
                'count': count,
                'avg_response_time': query_info[query]['response_time']
            }
            for query, count in popular
        ]
    
    def calculate_ragas_metrics(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Calculate RAGAS-style metrics for RAG evaluation
        
        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with RAGAS metrics
        """
        logger.info("Calculating RAGAS metrics...")
        
        metrics = {}
        
        # Context Relevance: How relevant are retrieved contexts to the query
        metrics['context_relevance'] = self._calculate_context_relevance(query, contexts)
        
        # Answer Relevance: How relevant is the answer to the query
        metrics['answer_relevance'] = self._calculate_answer_relevance(query, answer)
        
        # Faithfulness: How faithful is the answer to the contexts
        metrics['faithfulness'] = self._calculate_faithfulness(answer, contexts)
        
        # Ground truth metrics (if provided)
        if ground_truth:
            metrics['answer_correctness'] = self._calculate_correctness(answer, ground_truth)
        
        logger.success("RAGAS metrics calculated")
        return metrics
    
    def _calculate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Calculate context relevance score (simplified)"""
        query_words = set(query.lower().split())
        
        relevance_scores = []
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(query_words & context_words)
            score = overlap / len(query_words) if query_words else 0
            relevance_scores.append(score)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    
    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Calculate answer relevance score (simplified)"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(query_words & answer_words)
        return overlap / len(query_words) if query_words else 0
    
    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Calculate faithfulness score (simplified)"""
        answer_words = set(answer.lower().split())
        
        # Check how many answer words appear in contexts
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        supported = len(answer_words & context_words)
        return supported / len(answer_words) if answer_words else 0
    
    def _calculate_correctness(self, answer: str, ground_truth: str) -> float:
        """Calculate answer correctness (simplified)"""
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        overlap = len(answer_words & truth_words)
        union = len(answer_words | truth_words)
        
        return overlap / union if union else 0
    
    def save_analytics(self, filename: str = "analytics_report.json"):
        """Save all analytics to file"""
        filepath = self.save_dir / filename
        
        data = {
            'generated_at': datetime.utcnow().isoformat(),
            'query_analytics': self.get_query_analytics(days=30),
            'retrieval_analytics': self.get_retrieval_analytics(),
            'cost_report': self.get_cost_report(days=30),
            'popular_queries': self.get_popular_queries(top_n=10),
            'total_queries': len(self.query_history),
            'total_cost': sum(self.cost_metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Analytics saved to: {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """Generate human-readable analytics report"""
        report = []
        report.append("=" * 60)
        report.append("ANALYTICS REPORT")
        report.append("=" * 60)
        
        # Query Analytics
        query_analytics = self.get_query_analytics(days=30)
        if 'error' not in query_analytics:
            report.append("\n📊 QUERY ANALYTICS (Last 30 Days)")
            report.append(f"  Total Queries: {query_analytics['total_queries']}")
            report.append(f"  Avg Response Time: {query_analytics['avg_response_time']:.2f}s")
            report.append(f"  Total Cost: ${query_analytics['total_cost']:.2f}")
            report.append(f"  Avg Cost/Query: ${query_analytics['avg_cost_per_query']:.4f}")
            report.append(f"  Avg Sources Used: {query_analytics['avg_sources_used']:.1f}")
        
        # Cost Report
        cost_report = self.get_cost_report(days=30)
        if 'error' not in cost_report:
            report.append("\n💰 COST BREAKDOWN")
            report.append(f"  Total API Calls: {cost_report['total_api_calls']}")
            report.append(f"  Total Cost: ${cost_report['total_cost']:.2f}")
            report.append(f"  Total Tokens: {cost_report['total_tokens']:,}")
            
            report.append("\n  By Provider:")
            for provider, stats in cost_report['by_provider'].items():
                report.append(f"    {provider}: ${stats['cost']:.2f} ({stats['calls']} calls)")
        
        # Popular Queries
        popular = self.get_popular_queries(top_n=5)
        if popular:
            report.append("\n🔥 TOP 5 QUERIES")
            for i, q in enumerate(popular, 1):
                report.append(f"  {i}. \"{q['query'][:50]}...\" ({q['count']} times)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    logger.add("analytics.log", rotation="1 MB")
    
    print("Analytics Tracker initialized!")
    print("\nFeatures:")
    print("  - Query tracking")
    print("  - Retrieval metrics")
    print("  - Cost tracking")
    print("  - RAGAS evaluation")
    print("  - Analytics reports")