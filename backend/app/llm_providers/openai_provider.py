"""
OpenAI LLM Provider Implementation
"""

import asyncio
import time
from typing import List, Optional, AsyncGenerator, Dict
import openai
from openai import AsyncOpenAI
import logging

from app.llm_providers.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamResponse, 
    ModelInfo, ModelType
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__("openai", api_key or settings.OPENAI_API_KEY, **kwargs)
        self.client: Optional[AsyncOpenAI] = None
        
        # OpenAI model configurations
        self._model_configs = {
            "gpt-4": {
                "name": "gpt-4",
                "type": ModelType.CHAT,
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.03,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": False
            },
            "gpt-4-turbo": {
                "name": "gpt-4-turbo",
                "type": ModelType.CHAT,
                "max_tokens": 128000,
                "cost_per_1k_tokens": 0.01,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": True
            },
            "gpt-4o": {
                "name": "gpt-4o",
                "type": ModelType.CHAT,
                "max_tokens": 128000,
                "cost_per_1k_tokens": 0.005,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": True
            },
            "gpt-3.5-turbo": {
                "name": "gpt-3.5-turbo",
                "type": ModelType.CHAT,
                "max_tokens": 16385,
                "cost_per_1k_tokens": 0.001,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": False
            },
            "text-embedding-ada-002": {
                "name": "text-embedding-ada-002",
                "type": ModelType.EMBEDDING,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.0001,
                "supports_streaming": False,
                "supports_functions": False,
                "supports_vision": False
            },
            "text-embedding-3-small": {
                "name": "text-embedding-3-small",
                "type": ModelType.EMBEDDING,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00002,
                "supports_streaming": False,
                "supports_functions": False,
                "supports_vision": False
            },
            "text-embedding-3-large": {
                "name": "text-embedding-3-large",
                "type": ModelType.EMBEDDING,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00013,
                "supports_streaming": False,
                "supports_functions": False,
                "supports_vision": False
            }
        }
    
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Initialize model info
        for model_name, config in self._model_configs.items():
            self._models[model_name] = ModelInfo(
                name=model_name,
                provider=self.name,
                **config
            )
        
        self._initialized = True
        logger.info(f"OpenAI provider initialized with {len(self._models)} models")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text completion using OpenAI"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            
            if request.messages:
                messages.extend(request.messages)
            else:
                messages.append({"role": "user", "content": request.prompt})
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                stream=False
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason,
                provider=self.name,
                request_id=response.id,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamResponse, None]:
        """Generate streaming text completion"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare messages
            messages = []
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            
            if request.messages:
                messages.extend(request.messages)
            else:
                messages.append({"role": "user", "content": request.prompt})
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                stream=True
            )
            
            content = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    content += delta
                    
                    yield LLMStreamResponse(
                        content=content,
                        delta=delta,
                        model=chunk.model,
                        provider=self.name,
                        is_complete=False,
                        request_id=chunk.id
                    )
                
                if chunk.choices[0].finish_reason:
                    yield LLMStreamResponse(
                        content=content,
                        delta="",
                        model=chunk.model,
                        provider=self.name,
                        is_complete=True,
                        request_id=chunk.id
                    )
                    break
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get text embedding from OpenAI"""
        if not self._initialized:
            await self.initialize()
        
        model = model or "text-embedding-3-small"
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models"""
        if not self._initialized:
            await self.initialize()
        
        return list(self._models.values())
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Simple API call to check health
            await self.client.models.list()
            return True
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
