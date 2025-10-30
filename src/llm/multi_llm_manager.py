"""
Multi-LLM Manager
Support for multiple LLM providers with intelligent routing
"""

from typing import Dict, List, Optional, Any
from loguru import logger
from enum import Enum
import time

# Import LLM clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic Claude not available")


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT4_TURBO = "openai_gpt4_turbo"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    ANTHROPIC_CLAUDE_INSTANT = "anthropic_claude_instant"
    LOCAL_LLAMA = "local_llama"  # Placeholder for local models


class QueryComplexity(Enum):
    """Query complexity levels for routing"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class MultiLLMManager:
    """Manage multiple LLM providers with intelligent routing"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_provider: str = "openai_gpt4_turbo"
    ):
        """
        Initialize multi-LLM manager
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            default_provider: Default LLM provider to use
        """
        self.clients = {}
        self.model_configs = {}
        self.usage_stats = {
            'total_calls': 0,
            'by_provider': {},
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
        
        # Initialize OpenAI
        if openai_api_key and OPENAI_AVAILABLE:
            self.clients['openai'] = OpenAI(api_key=openai_api_key)
            self._setup_openai_models()
            logger.info("✓ OpenAI initialized")
        
        # Initialize Anthropic Claude
        if anthropic_api_key and ANTHROPIC_AVAILABLE:
            self.clients['anthropic'] = Anthropic(api_key=anthropic_api_key)
            self._setup_anthropic_models()
            logger.info("✓ Anthropic Claude initialized")
        
        self.default_provider = default_provider
        logger.info(f"MultiLLMManager initialized with default: {default_provider}")
    
    def _setup_openai_models(self):
        """Configure OpenAI models"""
        self.model_configs.update({
            LLMProvider.OPENAI_GPT4: {
                'model_name': 'gpt-4',
                'max_tokens': 8192,
                'cost_per_1k_input': 0.03,
                'cost_per_1k_output': 0.06,
                'speed': 'slow',
                'quality': 'highest'
            },
            LLMProvider.OPENAI_GPT4_TURBO: {
                'model_name': 'gpt-4-turbo-preview',
                'max_tokens': 4096,
                'cost_per_1k_input': 0.01,
                'cost_per_1k_output': 0.03,
                'speed': 'medium',
                'quality': 'high'
            },
            LLMProvider.OPENAI_GPT35: {
                'model_name': 'gpt-3.5-turbo',
                'max_tokens': 4096,
                'cost_per_1k_input': 0.0005,
                'cost_per_1k_output': 0.0015,
                'speed': 'fast',
                'quality': 'good'
            }
        })
    
    def _setup_anthropic_models(self):
        """Configure Anthropic models"""
        self.model_configs.update({
            LLMProvider.ANTHROPIC_CLAUDE: {
                'model_name': 'claude-3-opus-20240229',
                'max_tokens': 4096,
                'cost_per_1k_input': 0.015,
                'cost_per_1k_output': 0.075,
                'speed': 'medium',
                'quality': 'highest'
            },
            LLMProvider.ANTHROPIC_CLAUDE_INSTANT: {
                'model_name': 'claude-3-sonnet-20240229',
                'max_tokens': 4096,
                'cost_per_1k_input': 0.003,
                'cost_per_1k_output': 0.015,
                'speed': 'fast',
                'quality': 'high'
            }
        })
    
    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion from any LLM provider
        
        Args:
            prompt: User prompt
            provider: LLM provider (auto-select if None)
            system_message: System message/instructions
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Auto-select provider if not specified
        if provider is None:
            provider = self.default_provider
        
        logger.info(f"Generating with provider: {provider}")
        
        # Route to appropriate provider
        provider_enum = LLMProvider(provider)
        
        if provider_enum in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT4_TURBO, LLMProvider.OPENAI_GPT35]:
            result = self._generate_openai(prompt, provider_enum, system_message, temperature, max_tokens)
        elif provider_enum in [LLMProvider.ANTHROPIC_CLAUDE, LLMProvider.ANTHROPIC_CLAUDE_INSTANT]:
            result = self._generate_anthropic(prompt, provider_enum, system_message, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Add metadata
        end_time = time.time()
        result['response_time'] = end_time - start_time
        result['provider'] = provider
        
        # Update stats
        self._update_stats(provider, result)
        
        return result
    
    def _generate_openai(
        self,
        prompt: str,
        provider: LLMProvider,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate with OpenAI"""
        if 'openai' not in self.clients:
            raise ValueError("OpenAI client not initialized")
        
        config = self.model_configs[provider]
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.clients['openai'].chat.completions.create(
                model=config['model_name'],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                'text': response.choices[0].message.content,
                'model': config['model_name'],
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'estimated_cost': self._calculate_cost(
                    provider,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def _generate_anthropic(
        self,
        prompt: str,
        provider: LLMProvider,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate with Anthropic Claude"""
        if 'anthropic' not in self.clients:
            raise ValueError("Anthropic client not initialized")
        
        config = self.model_configs[provider]
        
        try:
            response = self.clients['anthropic'].messages.create(
                model=config['model_name'],
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message or "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Anthropic returns usage in a different format
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            return {
                'text': response.content[0].text,
                'model': config['model_name'],
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'estimated_cost': self._calculate_cost(provider, input_tokens, output_tokens)
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def _calculate_cost(self, provider: LLMProvider, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost"""
        config = self.model_configs[provider]
        
        input_cost = (input_tokens / 1000) * config['cost_per_1k_input']
        output_cost = (output_tokens / 1000) * config['cost_per_1k_output']
        
        return input_cost + output_cost
    
    def _update_stats(self, provider: str, result: Dict):
        """Update usage statistics"""
        self.usage_stats['total_calls'] += 1
        self.usage_stats['total_tokens'] += result.get('total_tokens', 0)
        self.usage_stats['estimated_cost'] += result.get('estimated_cost', 0)
        
        if provider not in self.usage_stats['by_provider']:
            self.usage_stats['by_provider'][provider] = {
                'calls': 0,
                'tokens': 0,
                'cost': 0
            }
        
        self.usage_stats['by_provider'][provider]['calls'] += 1
        self.usage_stats['by_provider'][provider]['tokens'] += result.get('total_tokens', 0)
        self.usage_stats['by_provider'][provider]['cost'] += result.get('estimated_cost', 0)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        available = []
        
        if 'openai' in self.clients:
            available.extend([
                LLMProvider.OPENAI_GPT4.value,
                LLMProvider.OPENAI_GPT4_TURBO.value,
                LLMProvider.OPENAI_GPT35.value
            ])
        
        if 'anthropic' in self.clients:
            available.extend([
                LLMProvider.ANTHROPIC_CLAUDE.value,
                LLMProvider.ANTHROPIC_CLAUDE_INSTANT.value
            ])
        
        return available
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return self.usage_stats.copy()
    
    def estimate_query_complexity(self, query: str) -> QueryComplexity:
        """
        Estimate query complexity for routing
        
        Args:
            query: User query
            
        Returns:
            QueryComplexity enum
        """
        # Simple heuristics (can be improved with ML model)
        word_count = len(query.split())
        
        # Check for complexity indicators
        complex_indicators = [
            'explain in detail', 'analyze', 'compare', 'evaluate',
            'comprehensive', 'step by step', 'detailed'
        ]
        
        simple_indicators = [
            'what is', 'define', 'who is', 'when', 'where',
            'list', 'name'
        ]
        
        query_lower = query.lower()
        
        # Check for indicators
        has_complex = any(ind in query_lower for ind in complex_indicators)
        has_simple = any(ind in query_lower for ind in simple_indicators)
        
        if has_complex or word_count > 50:
            return QueryComplexity.COMPLEX
        elif has_simple and word_count < 20:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE
    
    def route_query(
        self,
        query: str,
        optimize_for: str = "balanced"
    ) -> str:
        """
        Intelligently route query to best LLM
        
        Args:
            query: User query
            optimize_for: 'cost', 'quality', 'speed', or 'balanced'
            
        Returns:
            Selected provider string
        """
        complexity = self.estimate_query_complexity(query)
        available = self.get_available_providers()
        
        logger.info(f"Query complexity: {complexity.value}, optimizing for: {optimize_for}")
        
        # Routing logic
        if optimize_for == "cost":
            # Cheapest model that can handle the complexity
            if complexity == QueryComplexity.SIMPLE:
                return LLMProvider.OPENAI_GPT35.value if LLMProvider.OPENAI_GPT35.value in available else available[0]
            elif complexity == QueryComplexity.MODERATE:
                return LLMProvider.OPENAI_GPT4_TURBO.value if LLMProvider.OPENAI_GPT4_TURBO.value in available else available[0]
            else:
                return LLMProvider.OPENAI_GPT4.value if LLMProvider.OPENAI_GPT4.value in available else available[0]
        
        elif optimize_for == "quality":
            # Best quality regardless of cost
            if LLMProvider.OPENAI_GPT4.value in available:
                return LLMProvider.OPENAI_GPT4.value
            elif LLMProvider.ANTHROPIC_CLAUDE.value in available:
                return LLMProvider.ANTHROPIC_CLAUDE.value
            else:
                return available[0]
        
        elif optimize_for == "speed":
            # Fastest model
            if LLMProvider.OPENAI_GPT35.value in available:
                return LLMProvider.OPENAI_GPT35.value
            elif LLMProvider.ANTHROPIC_CLAUDE_INSTANT.value in available:
                return LLMProvider.ANTHROPIC_CLAUDE_INSTANT.value
            else:
                return available[0]
        
        else:  # balanced
            # Balance cost, quality, and speed
            if complexity == QueryComplexity.SIMPLE:
                return LLMProvider.OPENAI_GPT35.value if LLMProvider.OPENAI_GPT35.value in available else available[0]
            elif complexity == QueryComplexity.MODERATE:
                return LLMProvider.OPENAI_GPT4_TURBO.value if LLMProvider.OPENAI_GPT4_TURBO.value in available else available[0]
            else:
                return LLMProvider.OPENAI_GPT4.value if LLMProvider.OPENAI_GPT4.value in available else available[0]
    
    def compare_models(self, prompt: str, providers: List[str]) -> Dict[str, Dict]:
        """
        Compare responses from multiple models
        
        Args:
            prompt: Test prompt
            providers: List of provider strings to compare
            
        Returns:
            Dictionary of results by provider
        """
        logger.info(f"Comparing {len(providers)} models...")
        
        results = {}
        
        for provider in providers:
            try:
                result = self.generate(prompt, provider=provider)
                results[provider] = result
                logger.info(f"✓ {provider}: {result['response_time']:.2f}s, ${result['estimated_cost']:.4f}")
            except Exception as e:
                logger.error(f"✗ {provider} failed: {e}")
                results[provider] = {'error': str(e)}
        
        return results


# Example usage
if __name__ == "__main__":
    logger.add("multi_llm.log", rotation="1 MB")
    
    print("Multi-LLM Manager")
    print("\nSupported Providers:")
    print("  - OpenAI GPT-4")
    print("  - OpenAI GPT-4 Turbo")
    print("  - OpenAI GPT-3.5 Turbo")
    print("  - Anthropic Claude 3 Opus")
    print("  - Anthropic Claude 3 Sonnet")
