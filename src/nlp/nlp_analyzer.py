"""
NLP Analyzer (OpenAI-Only Version)
Named Entity Recognition, Sentiment Analysis, Classification, and Summarization
"""

from typing import Dict, List, Optional
from loguru import logger
import re
from collections import Counter
from openai import OpenAI


class NLPAnalyzer:
    """Advanced NLP analysis using OpenAI (no local dependencies)"""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize NLP analyzer
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        logger.info("✓ NLP Analyzer initialized with OpenAI")
    
    def named_entity_recognition(self, text: str) -> Dict:
        """
        Extract named entities from text using OpenAI
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entities by type
        """
        logger.info("Performing Named Entity Recognition...")
        
        prompt = f"""Extract named entities from this text. Categorize them as: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, or EVENT.

Text: {text}

Return as JSON format with arrays for each category. Only include categories that have entities.
Example:
{{
    "PERSON": ["John Smith", "Jane Doe"],
    "ORGANIZATION": ["Microsoft", "Apple"],
    "LOCATION": ["Seattle", "New York"]
}}

JSON:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at named entity recognition. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            import json
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
            else:
                entities = json.loads(content)
            
            logger.success(f"Found {sum(len(v) for v in entities.values())} entities")
            return entities
            
        except Exception as e:
            logger.error(f"NER failed: {e}")
            return self._ner_simple(text)
    
    def _ner_simple(self, text: str) -> Dict:
        """Simple rule-based NER (fallback)"""
        entities = {}
        
        # Capitalized words (potential names/places)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if capitalized:
            entities['CAPITALIZED'] = list(set(capitalized))
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', text)
        if dates:
            entities['DATE'] = dates
        
        # Money
        money = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        if money:
            entities['MONEY'] = money
        
        logger.info(f"Simple NER: Found {sum(len(v) for v in entities.values())} entities")
        return entities
    
    def sentiment_analysis(self, text: str) -> Dict:
        """
        Analyze sentiment of text using OpenAI
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment label and score
        """
        logger.info("Performing sentiment analysis...")
        
        prompt = f"""Analyze the sentiment of this text. Respond with:
- Label: POSITIVE, NEGATIVE, or NEUTRAL
- Score: 0.0 to 1.0 (confidence)
- Brief explanation (one sentence)

Text: {text[:500]}

Format:
Label: [label]
Score: [score]
Explanation: [explanation]"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            label_match = re.search(r'Label:\s*(\w+)', content, re.IGNORECASE)
            score_match = re.search(r'Score:\s*([\d.]+)', content)
            explanation_match = re.search(r'Explanation:\s*(.+)', content, re.DOTALL)
            
            sentiment = {
                'label': label_match.group(1).upper() if label_match else 'NEUTRAL',
                'score': float(score_match.group(1)) if score_match else 0.5,
                'explanation': explanation_match.group(1).strip() if explanation_match else '',
                'method': 'llm'
            }
            
            logger.success(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._sentiment_simple(text)
    
    def _sentiment_simple(self, text: str) -> Dict:
        """Simple lexicon-based sentiment (fallback)"""
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'love', 'best', 'perfect', 'happy', 'brilliant', 'outstanding'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
                         'disappointing', 'sad', 'angry', 'useless', 'fail'}
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        if pos_count > neg_count:
            label = 'POSITIVE'
            score = min(0.5 + (pos_count - neg_count) * 0.1, 1.0)
        elif neg_count > pos_count:
            label = 'NEGATIVE'
            score = min(0.5 + (neg_count - pos_count) * 0.1, 1.0)
        else:
            label = 'NEUTRAL'
            score = 0.5
        
        logger.info(f"Simple Sentiment: {label} ({score:.2f})")
        return {'label': label, 'score': score, 'method': 'simple'}
    
    def text_classification(self, text: str, categories: List[str]) -> Dict:
        """
        Classify text into predefined categories using OpenAI
        
        Args:
            text: Input text
            categories: List of possible categories
            
        Returns:
            Dictionary with predicted category and confidence
        """
        logger.info(f"Classifying text into {len(categories)} categories...")
        
        categories_str = ", ".join(categories)
        
        prompt = f"""Classify this text into one of these categories: {categories_str}

Text: {text[:500]}

Respond with:
Category: [category name from the list above]
Confidence: [0.0-1.0]
Reasoning: [brief explanation in one sentence]"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at text classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            category_match = re.search(r'Category:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', content)
            reasoning_match = re.search(r'Reasoning:\s*(.+)', content, re.DOTALL)
            
            classification = {
                'category': category_match.group(1).strip() if category_match else categories[0],
                'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
                'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
                'all_categories': categories
            }
            
            logger.success(f"Classified as: {classification['category']} ({classification['confidence']:.2f})")
            return classification
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._classify_simple(text, categories)
    
    def _classify_simple(self, text: str, categories: List[str]) -> Dict:
        """Simple keyword-based classification"""
        text_lower = text.lower()
        scores = {}
        
        for category in categories:
            score = text_lower.count(category.lower())
            scores[category] = score
        
        if max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            confidence = min(scores[best_category] / 10, 1.0)
        else:
            best_category = categories[0]
            confidence = 0.1
        
        logger.info(f"Simple classification: {best_category} ({confidence:.2f})")
        return {
            'category': best_category,
            'confidence': confidence,
            'method': 'simple'
        }
    
    def summarize_text(self, text: str, max_length: int = 150) -> Dict:
        """
        Summarize text using OpenAI
        
        Args:
            text: Input text
            max_length: Maximum summary length (words)
            
        Returns:
            Dictionary with summary and metadata
        """
        logger.info(f"Summarizing text ({len(text)} chars)...")
        
        prompt = f"""Summarize this text in approximately {max_length} words. Be concise and capture the main points.

Text: {text}

Summary:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at text summarization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=max_length * 2
            )
            
            summary = response.choices[0].message.content.strip()
            
            result = {
                'summary': summary,
                'original_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary) / len(text),
                'method': 'llm'
            }
            
            logger.success(f"Generated summary ({result['summary_length']} words, {result['compression_ratio']:.1%} compression)")
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._summarize_extractive(text, max_length)
    
    def _summarize_extractive(self, text: str, max_length: int) -> Dict:
        """Simple extractive summarization"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Take first few sentences
        summary_sentences = []
        word_count = 0
        
        for sent in sentences:
            sent_words = len(sent.split())
            if word_count + sent_words <= max_length:
                summary_sentences.append(sent)
                word_count += sent_words
            else:
                break
        
        summary = '. '.join(summary_sentences) + '.'
        
        return {
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()),
            'compression_ratio': len(summary) / len(text),
            'method': 'extractive'
        }
    
    def analyze_document(self, text: str) -> Dict:
        """
        Comprehensive document analysis
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with all NLP analyses
        """
        logger.info("Performing comprehensive document analysis...")
        
        analysis = {
            'text_stats': self._get_text_stats(text),
            'entities': self.named_entity_recognition(text),
            'sentiment': self.sentiment_analysis(text),
            'summary': self.summarize_text(text, max_length=100)
        }
        
        logger.success("Document analysis complete")
        return analysis
    
    def _get_text_stats(self, text: str) -> Dict:
        """Get basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0
        }


# Example usage
if __name__ == "__main__":
    logger.add("nlp_analyzer.log", rotation="1 MB")
    
    print("NLP Analyzer initialized!")
    print("\nFeatures:")
    print("  - Named Entity Recognition (OpenAI)")
    print("  - Sentiment Analysis (OpenAI)")
    print("  - Text Classification (OpenAI)")
    print("  - Text Summarization (OpenAI)")
    print("  - Document Analysis")
