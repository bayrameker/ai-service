"""
Anthropic Claude LLM Provider Implementation
"""

import asyncio
import time
from typing import List, Optional, AsyncGenerator, Dict
import anthropic
from anthropic import AsyncAnthropic
import logging

from app.llm_providers.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamResponse, 
    ModelInfo, ModelType
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__("anthropic", api_key or settings.ANTHROPIC_API_KEY, **kwargs)
        self.client: Optional[AsyncAnthropic] = None
        
        # Anthropic model configurations
        self._model_configs = {
            "claude-3-opus-20240229": {
                "name": "claude-3-opus-20240229",
                "type": ModelType.CHAT,
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.015,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": True
            },
            "claude-3-sonnet-20240229": {
                "name": "claude-3-sonnet-20240229",
                "type": ModelType.CHAT,
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.003,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": True
            },
            "claude-3-5-sonnet-20241022": {
                "name": "claude-3-5-sonnet-20241022",
                "type": ModelType.CHAT,
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.003,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": True
            },
            "claude-3-haiku-20240307": {
                "name": "claude-3-haiku-20240307",
                "type": ModelType.CHAT,
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.00025,
                "supports_streaming": True,
                "supports_functions": True,
                "supports_vision": True
            },
            "claude-2.1": {
                "name": "claude-2.1",
                "type": ModelType.CHAT,
                "max_tokens": 200000,
                "cost_per_1k_tokens": 0.008,
                "supports_streaming": True,
                "supports_functions": False,
                "supports_vision": False
            }
        }
    
    async def initialize(self) -> None:
        """Initialize Anthropic client"""
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        
        # Initialize model info
        for model_name, config in self._model_configs.items():
            self._models[model_name] = ModelInfo(
                name=model_name,
                provider=self.name,
                **config
            )
        
        self._initialized = True
        logger.info(f"Anthropic provider initialized with {len(self._models)} models")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text completion using Anthropic Claude"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare messages for Claude format
            messages = []
            system_message = request.system_message
            
            if request.messages:
                messages = request.messages.copy()
            else:
                messages = [{"role": "user", "content": request.prompt}]
            
            # Claude API call parameters
            kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 1000,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop_sequences": request.stop,
                "stream": False
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            # Make API call
            response = await self.client.messages.create(**kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract content from response
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                finish_reason=response.stop_reason or "stop",
                provider=self.name,
                request_id=response.id,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamResponse, None]:
        """Generate streaming text completion"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare messages for Claude format
            messages = []
            system_message = request.system_message
            
            if request.messages:
                messages = request.messages.copy()
            else:
                messages = [{"role": "user", "content": request.prompt}]
            
            # Claude API call parameters
            kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 1000,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop_sequences": request.stop,
                "stream": True
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            # Make streaming API call
            stream = await self.client.messages.create(**kwargs)
            
            content = ""
            request_id = None
            model_name = request.model
            
            async for chunk in stream:
                if chunk.type == "message_start":
                    request_id = chunk.message.id
                    model_name = chunk.message.model
                elif chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        delta = chunk.delta.text
                        content += delta
                        
                        yield LLMStreamResponse(
                            content=content,
                            delta=delta,
                            model=model_name,
                            provider=self.name,
                            is_complete=False,
                            request_id=request_id
                        )
                elif chunk.type == "message_delta":
                    if chunk.delta.stop_reason:
                        yield LLMStreamResponse(
                            content=content,
                            delta="",
                            model=model_name,
                            provider=self.name,
                            is_complete=True,
                            request_id=request_id
                        )
                        break
                        
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get text embedding - Anthropic doesn't provide embeddings, raise NotImplementedError"""
        raise NotImplementedError("Anthropic does not provide embedding models")
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Anthropic models"""
        if not self._initialized:
            await self.initialize()
        
        return list(self._models.values())
    
    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Simple API call to check health
            test_response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
            
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
