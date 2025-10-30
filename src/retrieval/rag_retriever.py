"""
Advanced RAG Retriever
Implements multi-query, self-RAG, and hybrid retrieval strategies
"""

from typing import List, Dict, Optional, Tuple
from loguru import logger
import numpy as np
from openai import OpenAI
from src.embeddings.embeddings_generator import EmbeddingsGenerator
from src.embeddings.faiss_store import FAISSVectorStore

class AdvancedRAGRetriever:
    """Advanced retrieval strategies for RAG"""
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embeddings_generator: EmbeddingsGenerator,
        llm_api_key: str,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize advanced RAG retriever
        
        Args:
            vector_store: FAISS vector store
            embeddings_generator: Embeddings generator
            llm_api_key: OpenAI API key for LLM
            llm_model: LLM model name
        """
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.llm_client = OpenAI(api_key=llm_api_key)
        self.llm_model = llm_model
        
        logger.info(f"AdvancedRAGRetriever initialized with model: {llm_model}")
    
    def multi_query_retrieval(
        self,
        query: str,
        num_queries: int = 3,
        k_per_query: int = 3
    ) -> List[Dict]:
        """
        Multi-query retrieval: Generate multiple search queries and combine results
        
        Args:
            query: Original user query
            num_queries: Number of alternative queries to generate
            k_per_query: Number of results per query
            
        Returns:
            Combined and deduplicated results
        """
        logger.info(f"Multi-query retrieval for: '{query}'")
        
        # Generate alternative queries using LLM
        alternative_queries = self._generate_alternative_queries(query, num_queries)
        logger.info(f"Generated {len(alternative_queries)} alternative queries")
        
        # Retrieve for each query
        all_results = []
        seen_texts = set()
        
        for i, alt_query in enumerate([query] + alternative_queries, 1):
            logger.info(f"  Query {i}: {alt_query}")
            
            # Generate embedding
            query_embedding = self.embeddings_generator.generate_embedding(alt_query)
            
            # Search
            results = self.vector_store.search(query_embedding, k=k_per_query)
            
            # Deduplicate by text
            for result in results:
                text_key = result['text'][:100]  # Use first 100 chars as key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    result['source_query'] = alt_query
                    all_results.append(result)
        
        logger.success(f"Multi-query retrieval: {len(all_results)} unique results")
        return all_results
    
    def self_rag_retrieval(
        self,
        query: str,
        k: int = 5,
        relevance_threshold: float = 0.5
    ) -> Tuple[List[Dict], bool]:
        """
        Self-RAG: LLM evaluates if retrieval is needed and assesses relevance
        
        Args:
            query: User query
            k: Number of results to retrieve
            relevance_threshold: Minimum relevance score
            
        Returns:
            Tuple of (filtered results, needs_retrieval flag)
        """
        logger.info(f"Self-RAG retrieval for: '{query}'")
        
        # Step 1: Decide if retrieval is needed
        needs_retrieval = self._decide_retrieval_need(query)
        
        if not needs_retrieval:
            logger.info("LLM determined retrieval not needed for this query")
            return [], False
        
        # Step 2: Retrieve documents
        query_embedding = self.embeddings_generator.generate_embedding(query)
        results = self.vector_store.search(query_embedding, k=k)
        
        # Step 3: Evaluate relevance of each result
        filtered_results = []
        for result in results:
            relevance_score = self._evaluate_relevance(query, result['text'])
            result['llm_relevance_score'] = relevance_score
            
            if relevance_score >= relevance_threshold:
                filtered_results.append(result)
                logger.info(f"  ✓ Relevant (score: {relevance_score:.2f})")
            else:
                logger.info(f"  ✗ Not relevant (score: {relevance_score:.2f})")
        
        logger.success(f"Self-RAG: {len(filtered_results)}/{len(results)} results passed relevance check")
        return filtered_results, True
    
    def hybrid_retrieval(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Hybrid retrieval: Combine semantic search with keyword matching
        
        Args:
            query: User query
            k: Number of results
            semantic_weight: Weight for semantic scores (0-1)
            
        Returns:
            Combined and re-ranked results
        """
        logger.info(f"Hybrid retrieval for: '{query}'")
        
        # Semantic search
        query_embedding = self.embeddings_generator.generate_embedding(query)
        semantic_results = self.vector_store.search(query_embedding, k=k)
        
        # Keyword matching (simple BM25-like scoring)
        keyword_scores = self._keyword_search(query, semantic_results)
        
        # Combine scores
        for i, result in enumerate(semantic_results):
            semantic_score = result['similarity_score']
            keyword_score = keyword_scores[i]
            
            # Hybrid score
            hybrid_score = (
                semantic_weight * semantic_score + 
                (1 - semantic_weight) * keyword_score
            )
            
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            result['hybrid_score'] = hybrid_score
        
        # Re-rank by hybrid score
        semantic_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        logger.success(f"Hybrid retrieval: re-ranked {len(semantic_results)} results")
        return semantic_results
    
    def contextual_compression(
        self,
        query: str,
        results: List[Dict],
        max_tokens: int = 500
    ) -> List[Dict]:
        """
        Contextual compression: Extract only relevant portions of retrieved documents
        
        Args:
            query: User query
            results: Retrieved results
            max_tokens: Maximum tokens per compressed result
            
        Returns:
            Results with compressed text
        """
        logger.info(f"Applying contextual compression...")
        
        compressed_results = []
        
        for result in results:
            # Use LLM to extract relevant portions
            compressed_text = self._compress_text(query, result['text'], max_tokens)
            
            compressed_result = result.copy()
            compressed_result['original_text'] = result['text']
            compressed_result['compressed_text'] = compressed_text
            compressed_result['compression_ratio'] = len(compressed_text) / len(result['text'])
            
            compressed_results.append(compressed_result)
        
        logger.success(f"Compressed {len(results)} results")
        return compressed_results
    
    def _generate_alternative_queries(self, query: str, num_queries: int) -> List[str]:
        """Generate alternative phrasings of the query"""
        prompt = f"""Given the following question, generate {num_queries} alternative phrasings or related questions that could help retrieve relevant information.

Original question: {query}

Generate {num_queries} alternative questions (one per line):"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            alternatives = response.choices[0].message.content.strip().split('\n')
            # Clean up (remove numbers, bullets, etc.)
            alternatives = [q.strip().lstrip('0123456789.-) ') for q in alternatives if q.strip()]
            
            return alternatives[:num_queries]
            
        except Exception as e:
            logger.error(f"Failed to generate alternative queries: {e}")
            return []
    
    def _decide_retrieval_need(self, query: str) -> bool:
        """Decide if retrieval is needed for this query"""
        prompt = f"""Determine if external information retrieval is needed to answer this question.

Question: {query}

Answer with just 'YES' if retrieval is needed, or 'NO' if the question can be answered with general knowledge.

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return 'YES' in answer
            
        except Exception as e:
            logger.error(f"Failed to decide retrieval need: {e}")
            return True  # Default to retrieval
    
    def _evaluate_relevance(self, query: str, document: str) -> float:
        """Evaluate relevance of a document to the query (0-1 score)"""
        prompt = f"""Rate the relevance of this document to the question on a scale of 0.0 to 1.0.

Question: {query}

Document: {document[:500]}...

Relevance score (0.0-1.0):"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating document relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to 0-1
            
        except Exception as e:
            logger.error(f"Failed to evaluate relevance: {e}")
            return 0.5  # Default score
    
    def _keyword_search(self, query: str, results: List[Dict]) -> List[float]:
        """Simple keyword matching scores"""
        query_words = set(query.lower().split())
        scores = []
        
        for result in results:
            text_words = set(result['text'].lower().split())
            # Jaccard similarity
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            score = intersection / union if union > 0 else 0.0
            scores.append(score)
        
        return scores
    
    def _compress_text(self, query: str, text: str, max_tokens: int) -> str:
        """Extract relevant portions of text"""
        prompt = f"""Extract the most relevant information from this text that answers the question. Keep it concise (under {max_tokens} tokens).

Question: {query}

Text: {text}

Relevant extract:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts relevant information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to compress text: {e}")
            return text[:max_tokens * 4]  # Rough fallback


# Example usage
if __name__ == "__main__":
    logger.add("rag_retriever.log", rotation="1 MB")
    
    print("Advanced RAG Retriever initialized!")
    print("Features:")
    print("  - Multi-query retrieval")
    print("  - Self-RAG with relevance evaluation")
    print("  - Hybrid search (semantic + keyword)")
    print("  - Contextual compression")
