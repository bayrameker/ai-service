"""
LLM Provider Manager - Unified interface for all LLM providers
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from enum import Enum
import random

from app.llm_providers.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamResponse, 
    ModelInfo, ModelType
)
from app.llm_providers.openai_provider import OpenAIProvider
from app.llm_providers.anthropic_provider import AnthropicProvider
from app.llm_providers.ollama_provider import OllamaProvider
from app.core.config import settings

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"


class LLMProviderManager:
    """Manages all LLM providers and provides unified interface"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.model_to_provider: Dict[str, str] = {}
        self.provider_stats: Dict[str, Dict[str, Any]] = {}
        self.load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        self._round_robin_index = 0
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all available LLM providers"""
        logger.info("Initializing LLM Provider Manager...")
        
        # Initialize OpenAI provider
        if settings.OPENAI_API_KEY:
            try:
                openai_provider = OpenAIProvider()
                await openai_provider.initialize()
                self.providers["openai"] = openai_provider
                self.provider_stats["openai"] = {
                    "requests": 0,
                    "errors": 0,
                    "total_latency": 0,
                    "avg_latency": 0
                }
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Anthropic provider
        if settings.ANTHROPIC_API_KEY:
            try:
                anthropic_provider = AnthropicProvider()
                await anthropic_provider.initialize()
                self.providers["anthropic"] = anthropic_provider
                self.provider_stats["anthropic"] = {
                    "requests": 0,
                    "errors": 0,
                    "total_latency": 0,
                    "avg_latency": 0
                }
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic provider: {e}")
        
        # Initialize Ollama provider
        try:
            ollama_provider = OllamaProvider()
            await ollama_provider.initialize()
            self.providers["ollama"] = ollama_provider
            self.provider_stats["ollama"] = {
                "requests": 0,
                "errors": 0,
                "total_latency": 0,
                "avg_latency": 0
            }
            logger.info("Ollama provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")
        
        # Build model to provider mapping
        await self._build_model_mapping()
        
        self._initialized = True
        logger.info(f"LLM Provider Manager initialized with {len(self.providers)} providers")
    
    async def _build_model_mapping(self) -> None:
        """Build mapping from model names to providers"""
        self.model_to_provider.clear()
        
        for provider_name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                for model in models:
                    self.model_to_provider[model.name] = provider_name
            except Exception as e:
                logger.error(f"Failed to get models from {provider_name}: {e}")
    
    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Get provider by name"""
        return self.providers.get(provider_name)
    
    def get_provider_for_model(self, model_name: str) -> Optional[BaseLLMProvider]:
        """Get provider that supports the given model"""
        provider_name = self.model_to_provider.get(model_name)
        if provider_name:
            return self.providers.get(provider_name)
        return None
    
    async def generate(self, request: LLMRequest, provider_name: Optional[str] = None) -> LLMResponse:
        """Generate text completion using specified or auto-selected provider"""
        if not self._initialized:
            await self.initialize()
        
        # Select provider
        if provider_name:
            provider = self.get_provider(provider_name)
            if not provider:
                raise ValueError(f"Provider '{provider_name}' not available")
        else:
            provider = self.get_provider_for_model(request.model)
            if not provider:
                raise ValueError(f"No provider available for model '{request.model}'")
        
        # Update stats
        provider_name = provider.name
        self.provider_stats[provider_name]["requests"] += 1
        
        try:
            response = await provider.generate(request)
            
            # Update latency stats
            if response.latency_ms:
                stats = self.provider_stats[provider_name]
                stats["total_latency"] += response.latency_ms
                stats["avg_latency"] = stats["total_latency"] / stats["requests"]
            
            return response
            
        except Exception as e:
            self.provider_stats[provider_name]["errors"] += 1
            logger.error(f"Generation failed with {provider_name}: {e}")
            raise
    
    async def generate_stream(
        self, 
        request: LLMRequest, 
        provider_name: Optional[str] = None
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Generate streaming text completion"""
        if not self._initialized:
            await self.initialize()
        
        # Select provider
        if provider_name:
            provider = self.get_provider(provider_name)
            if not provider:
                raise ValueError(f"Provider '{provider_name}' not available")
        else:
            provider = self.get_provider_for_model(request.model)
            if not provider:
                raise ValueError(f"No provider available for model '{request.model}'")
        
        # Update stats
        provider_name = provider.name
        self.provider_stats[provider_name]["requests"] += 1
        
        try:
            async for chunk in provider.generate_stream(request):
                yield chunk
                
        except Exception as e:
            self.provider_stats[provider_name]["errors"] += 1
            logger.error(f"Streaming generation failed with {provider_name}: {e}")
            raise
    
    async def get_embedding(
        self, 
        text: str, 
        model: Optional[str] = None, 
        provider_name: Optional[str] = None
    ) -> List[float]:
        """Get text embedding"""
        if not self._initialized:
            await self.initialize()
        
        # Select provider for embeddings
        if provider_name:
            provider = self.get_provider(provider_name)
        elif model:
            provider = self.get_provider_for_model(model)
        else:
            # Default to OpenAI for embeddings
            provider = self.get_provider("openai")
        
        if not provider:
            raise ValueError("No provider available for embeddings")
        
        try:
            return await provider.get_embedding(text, model)
        except NotImplementedError:
            # Try fallback to OpenAI
            openai_provider = self.get_provider("openai")
            if openai_provider and openai_provider != provider:
                return await openai_provider.get_embedding(text, model)
            raise
    
    async def list_all_models(self) -> List[ModelInfo]:
        """List all available models from all providers"""
        if not self._initialized:
            await self.initialize()
        
        all_models = []
        for provider in self.providers.values():
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception as e:
                logger.error(f"Failed to get models from {provider.name}: {e}")
        
        return all_models
    
    async def list_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """List models by type"""
        all_models = await self.list_all_models()
        return [model for model in all_models if model.type == model_type]
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_status = {}
        
        for provider_name, provider in self.providers.items():
            try:
                health_status[provider_name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
                health_status[provider_name] = False
        
        return health_status
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get provider statistics"""
        return self.provider_stats.copy()
    
    async def select_best_provider(
        self, 
        model_type: ModelType = ModelType.CHAT,
        strategy: Optional[LoadBalancingStrategy] = None
    ) -> Optional[str]:
        """Select best provider based on strategy"""
        strategy = strategy or self.load_balancing_strategy
        
        available_providers = []
        for provider_name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                if any(model.type == model_type for model in models):
                    available_providers.append(provider_name)
            except Exception:
                continue
        
        if not available_providers:
            return None
        
        if strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_providers)
        
        elif strategy == LoadBalancingStrategy.ROUND_ROBIN:
            provider = available_providers[self._round_robin_index % len(available_providers)]
            self._round_robin_index += 1
            return provider
        
        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select provider with least requests
            min_requests = float('inf')
            best_provider = None
            for provider_name in available_providers:
                requests = self.provider_stats[provider_name]["requests"]
                if requests < min_requests:
                    min_requests = requests
                    best_provider = provider_name
            return best_provider
        
        elif strategy == LoadBalancingStrategy.LATENCY_OPTIMIZED:
            # Select provider with lowest average latency
            min_latency = float('inf')
            best_provider = None
            for provider_name in available_providers:
                avg_latency = self.provider_stats[provider_name]["avg_latency"]
                if avg_latency > 0 and avg_latency < min_latency:
                    min_latency = avg_latency
                    best_provider = provider_name
            return best_provider or available_providers[0]
        
        elif strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            # Select cheapest provider (Ollama first, then others)
            if "ollama" in available_providers:
                return "ollama"
            return available_providers[0]
        
        return available_providers[0]
    
    async def shutdown(self) -> None:
        """Shutdown all providers"""
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except Exception as e:
                    logger.error(f"Error closing provider {provider.name}: {e}")
        
        logger.info("LLM Provider Manager shutdown complete")
