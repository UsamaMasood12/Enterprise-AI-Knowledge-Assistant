"""
RAG Pipeline
Complete RAG system integrating retrieval and generation
"""

from typing import List, Dict, Optional
from loguru import logger
from openai import OpenAI
from src.retrieval.rag_retriever import AdvancedRAGRetriever
from datetime import datetime
import json

class RAGPipeline:
    """End-to-end RAG pipeline for question answering"""
    
    def __init__(
        self,
        retriever: AdvancedRAGRetriever,
        llm_api_key: str,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Advanced RAG retriever
            llm_api_key: OpenAI API key
            llm_model: LLM model for generation
            temperature: Generation temperature
        """
        self.retriever = retriever
        self.llm_client = OpenAI(api_key=llm_api_key)
        self.llm_model = llm_model
        self.temperature = temperature
        
        logger.info(f"RAGPipeline initialized with model: {llm_model}")
    
    def query(
        self,
        question: str,
        retrieval_method: str = "multi_query",
        k: int = 5,
        include_sources: bool = True
    ) -> Dict:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User question
            retrieval_method: Method to use ('simple', 'multi_query', 'self_rag', 'hybrid')
            k: Number of documents to retrieve
            include_sources: Include source documents in response
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: '{question}' with method: {retrieval_method}")
        start_time = datetime.utcnow()
        
        # Step 1: Retrieve relevant documents
        if retrieval_method == "multi_query":
            results = self.retriever.multi_query_retrieval(question, num_queries=2, k_per_query=k)
        elif retrieval_method == "self_rag":
            results, needs_retrieval = self.retriever.self_rag_retrieval(question, k=k)
            if not needs_retrieval:
                return self._generate_direct_answer(question, start_time)
        elif retrieval_method == "hybrid":
            results = self.retriever.hybrid_retrieval(question, k=k)
        else:  # simple
            query_embedding = self.retriever.embeddings_generator.generate_embedding(question)
            results = self.retriever.vector_store.search(query_embedding, k=k)
        
        logger.info(f"Retrieved {len(results)} documents")
        
        # Step 2: Compress results (optional)
        compressed_results = self.retriever.contextual_compression(question, results, max_tokens=300)
        
        # Step 3: Generate answer using LLM
        answer = self._generate_answer(question, compressed_results)
        
        # Step 4: Prepare response
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()
        
        response = {
            'question': question,
            'answer': answer,
            'retrieval_method': retrieval_method,
            'num_sources': len(results),
            'response_time': response_time,
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
                for r in results[:3]  # Top 3 sources
            ]
        
        logger.success(f"Generated answer in {response_time:.2f}s")
        return response
    
    def batch_query(
        self,
        questions: List[str],
        retrieval_method: str = "multi_query",
        k: int = 5
    ) -> List[Dict]:
        """
        Process multiple queries
        
        Args:
            questions: List of questions
            retrieval_method: Retrieval method to use
            k: Number of documents per query
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing {len(questions)} queries in batch")
        
        responses = []
        for i, question in enumerate(questions, 1):
            logger.info(f"\nQuery {i}/{len(questions)}")
            response = self.query(question, retrieval_method=retrieval_method, k=k)
            responses.append(response)
        
        logger.success(f"Completed batch processing of {len(questions)} queries")
        return responses
    
    def query_with_chat_history(
        self,
        question: str,
        chat_history: List[Dict],
        retrieval_method: str = "multi_query",
        k: int = 5
    ) -> Dict:
        """
        Query with conversation history for context
        
        Args:
            question: Current question
            chat_history: Previous chat messages [{"role": "user/assistant", "content": "..."}]
            retrieval_method: Retrieval method
            k: Number of documents
            
        Returns:
            Response dictionary
        """
        logger.info(f"Query with {len(chat_history)} history messages")
        
        # Reformulate question with context
        contextualized_question = self._contextualize_question(question, chat_history)
        
        # Process query
        response = self.query(contextualized_question, retrieval_method, k)
        response['original_question'] = question
        response['contextualized_question'] = contextualized_question
        
        return response
    
    def _generate_answer(self, question: str, results: List[Dict]) -> str:
        """Generate answer using LLM with retrieved context"""
        
        # Build context from results
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
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always cite which source you're using."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "I apologize, but I encountered an error generating the answer."
    
    def _generate_direct_answer(self, question: str, start_time: datetime) -> Dict:
        """Generate answer without retrieval"""
        logger.info("Generating direct answer (no retrieval needed)")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                'question': question,
                'answer': answer,
                'retrieval_method': 'direct',
                'num_sources': 0,
                'response_time': response_time,
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
    
    def _contextualize_question(self, question: str, chat_history: List[Dict]) -> str:
        """Reformulate question with conversation context"""
        
        if not chat_history:
            return question
        
        # Build history string
        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history[-3:]  # Last 3 exchanges
        ])
        
        prompt = f"""Given the conversation history, reformulate the following question to be standalone.

Conversation History:
{history_text}

Current Question: {question}

Standalone Question:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You reformulate questions to be standalone based on conversation context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            
            reformulated = response.choices[0].message.content.strip()
            return reformulated
            
        except Exception as e:
            logger.error(f"Failed to contextualize question: {e}")
            return question
    
    def save_response(self, response: Dict, filepath: str):
        """Save response to file"""
        with open(filepath, 'w') as f:
            json.dump(response, f, indent=2)
        logger.info(f"Saved response to: {filepath}")


# Example usage
if __name__ == "__main__":
    logger.add("rag_pipeline.log", rotation="1 MB")
    
    print("RAG Pipeline initialized!")
    print("\nFeatures:")
    print("  - Multiple retrieval strategies")
    print("  - Contextual compression")
    print("  - Source citation")
    print("  - Chat history support")
    print("  - Batch processing")
