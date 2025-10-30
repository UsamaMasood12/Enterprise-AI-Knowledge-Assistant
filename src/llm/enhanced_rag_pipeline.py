"""
Enhanced RAG Pipeline with Multi-LLM Support
"""

from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime
from src.llm.multi_llm_manager import MultiLLMManager, QueryComplexity
from src.retrieval.rag_retriever import AdvancedRAGRetriever
import json

class EnhancedRAGPipeline:
    """RAG pipeline with multi-LLM support and intelligent routing"""
    
    def __init__(
        self,
        retriever: AdvancedRAGRetriever,
        llm_manager: MultiLLMManager,
        default_optimization: str = "balanced"
    ):
        """
        Initialize enhanced RAG pipeline
        
        Args:
            retriever: Advanced RAG retriever
            llm_manager: Multi-LLM manager
            default_optimization: Default routing strategy ('cost', 'quality', 'speed', 'balanced')
        """
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.default_optimization = default_optimization
        
        logger.info(f"EnhancedRAGPipeline initialized with {default_optimization} optimization")
    
    def query(
        self,
        question: str,
        retrieval_method: str = "multi_query",
        k: int = 5,
        optimize_for: Optional[str] = None,
        provider: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Process query with intelligent LLM routing
        
        Args:
            question: User question
            retrieval_method: Retrieval method ('simple', 'multi_query', 'self_rag', 'hybrid')
            k: Number of documents to retrieve
            optimize_for: Optimization strategy (None uses default)
            provider: Specific LLM provider (None for auto-routing)
            include_sources: Include source documents
            
        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Processing query: '{question[:50]}...'")
        start_time = datetime.utcnow()
        
        # Step 1: Retrieve relevant documents
        if retrieval_method == "multi_query":
            results = self.retriever.multi_query_retrieval(question, num_queries=2, k_per_query=k)
        elif retrieval_method == "self_rag":
            results, needs_retrieval = self.retriever.self_rag_retrieval(question, k=k)
            if not needs_retrieval:
                return self._generate_direct_answer(question, start_time, provider or "auto")
        elif retrieval_method == "hybrid":
            results = self.retriever.hybrid_retrieval(question, k=k)
        else:  # simple
            query_embedding = self.retriever.embeddings_generator.generate_embedding(question)
            results = self.retriever.vector_store.search(query_embedding, k=k)
        
        logger.info(f"Retrieved {len(results)} documents")
        
        # Step 2: Compress results
        compressed_results = self.retriever.contextual_compression(question, results, max_tokens=300)
        
        # Step 3: Route to appropriate LLM
        if provider is None:
            optimization = optimize_for or self.default_optimization
            provider = self.llm_manager.route_query(question, optimize_for=optimization)
            logger.info(f"Auto-routed to: {provider}")
        
        # Step 4: Generate answer
        answer_data = self._generate_answer_with_llm(question, compressed_results, provider)
        
        # Step 5: Prepare response
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()
        
        response = {
            'question': question,
            'answer': answer_data['text'],
            'retrieval_method': retrieval_method,
            'llm_provider': provider,
            'llm_model': answer_data.get('model', 'unknown'),
            'num_sources': len(results),
            'response_time': response_time,
            'llm_tokens': answer_data.get('total_tokens', 0),
            'estimated_cost': answer_data.get('estimated_cost', 0),
            'timestamp': end_time.isoformat()
        }
        
        if include_sources:
            response['sources'] = [
                {
                    'text': r.get('compressed_text', r['text'])[:200] + "...",
                    'file_name': r.get('file_name', 'unknown'),
                    'similarity_score': r.get('similarity_score', 0),
                    'chunk_id': r.get('chunk_id', -1)
                }
                for r in results[:3]
            ]
        
        logger.success(f"Generated answer in {response_time:.2f}s (${response['estimated_cost']:.4f})")
        return response
    
    def batch_query_with_routing(
        self,
        questions: List[str],
        retrieval_method: str = "multi_query",
        k: int = 5,
        optimize_for: str = "balanced"
    ) -> List[Dict]:
        """
        Process multiple queries with intelligent routing for each
        
        Args:
            questions: List of questions
            retrieval_method: Retrieval method
            k: Number of documents
            optimize_for: Optimization strategy
            
        Returns:
            List of responses
        """
        logger.info(f"Batch processing {len(questions)} queries with {optimize_for} optimization")
        
        responses = []
        total_cost = 0
        
        for i, question in enumerate(questions, 1):
            logger.info(f"\nQuery {i}/{len(questions)}")
            
            response = self.query(
                question=question,
                retrieval_method=retrieval_method,
                k=k,
                optimize_for=optimize_for
            )
            
            responses.append(response)
            total_cost += response['estimated_cost']
            
            logger.info(f"  Provider: {response['llm_provider']}")
            logger.info(f"  Cost: ${response['estimated_cost']:.4f}")
        
        logger.success(f"Batch complete! Total cost: ${total_cost:.4f}")
        return responses
    
    def compare_llm_responses(
        self,
        question: str,
        providers: List[str],
        retrieval_method: str = "multi_query",
        k: int = 5
    ) -> Dict[str, Dict]:
        """
        Compare responses from different LLMs for the same query
        
        Args:
            question: Question to test
            providers: List of LLM providers to compare
            retrieval_method: Retrieval method
            k: Number of documents
            
        Returns:
            Dictionary of responses by provider
        """
        logger.info(f"Comparing {len(providers)} LLMs for query: '{question[:50]}...'")
        
        # Retrieve once (same for all LLMs)
        if retrieval_method == "multi_query":
            results = self.retriever.multi_query_retrieval(question, num_queries=2, k_per_query=k)
        else:
            query_embedding = self.retriever.embeddings_generator.generate_embedding(question)
            results = self.retriever.vector_store.search(query_embedding, k=k)
        
        compressed_results = self.retriever.contextual_compression(question, results, max_tokens=300)
        
        # Generate with each LLM
        comparisons = {}
        
        for provider in providers:
            try:
                start_time = datetime.utcnow()
                answer_data = self._generate_answer_with_llm(question, compressed_results, provider)
                end_time = datetime.utcnow()
                
                comparisons[provider] = {
                    'answer': answer_data['text'],
                    'model': answer_data.get('model', 'unknown'),
                    'tokens': answer_data.get('total_tokens', 0),
                    'cost': answer_data.get('estimated_cost', 0),
                    'response_time': (end_time - start_time).total_seconds()
                }
                
                logger.info(f"✓ {provider}: {comparisons[provider]['response_time']:.2f}s, ${comparisons[provider]['cost']:.4f}")
                
            except Exception as e:
                logger.error(f"✗ {provider} failed: {e}")
                comparisons[provider] = {'error': str(e)}
        
        return comparisons
    
    def _generate_answer_with_llm(
        self,
        question: str,
        results: List[Dict],
        provider: str
    ) -> Dict:
        """Generate answer using specified LLM"""
        
        # Build context
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('compressed_text', result['text'])
            source = result.get('file_name', 'unknown')
            context_parts.append(f"[Source {i} - {source}]\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
        
        system_message = "You are a helpful assistant that answers questions based on provided context. Always cite which source you're using."
        
        # Generate with LLM
        result = self.llm_manager.generate(
            prompt=prompt,
            provider=provider,
            system_message=system_message,
            temperature=0.7,
            max_tokens=500
        )
        
        return result
    
    def _generate_direct_answer(
        self,
        question: str,
        start_time: datetime,
        provider: str
    ) -> Dict:
        """Generate direct answer without retrieval"""
        logger.info("Generating direct answer (no retrieval needed)")
        
        try:
            result = self.llm_manager.generate(
                prompt=question,
                provider=provider,
                temperature=0.7,
                max_tokens=500
            )
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                'question': question,
                'answer': result['text'],
                'retrieval_method': 'direct',
                'llm_provider': provider,
                'llm_model': result.get('model', 'unknown'),
                'num_sources': 0,
                'response_time': response_time,
                'llm_tokens': result.get('total_tokens', 0),
                'estimated_cost': result.get('estimated_cost', 0),
                'timestamp': end_time.isoformat(),
                'note': 'No retrieval needed - answered from general knowledge'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate direct answer: {e}")
            return {
                'question': question,
                'answer': "I apologize, but I encountered an error.",
                'error': str(e)
            }
    
    def get_cost_report(self) -> Dict:
        """Get detailed cost report"""
        stats = self.llm_manager.get_usage_stats()
        
        report = {
            'total_queries': stats['total_calls'],
            'total_tokens': stats['total_tokens'],
            'total_cost': stats['estimated_cost'],
            'by_provider': stats['by_provider'],
            'average_cost_per_query': stats['estimated_cost'] / stats['total_calls'] if stats['total_calls'] > 0 else 0
        }
        
        return report
    
    def save_response(self, response: Dict, filepath: str):
        """Save response to file"""
        with open(filepath, 'w') as f:
            json.dump(response, f, indent=2)
        logger.info(f"Saved response to: {filepath}")


# Example usage
if __name__ == "__main__":
    logger.add("enhanced_rag.log", rotation="1 MB")
    
    print("Enhanced RAG Pipeline with Multi-LLM Support")
    print("\nFeatures:")
    print("  - Intelligent LLM routing")
    print("  - Cost optimization")
    print("  - Multi-provider support")
    print("  - Model comparison")
    print("  - Detailed cost tracking")
