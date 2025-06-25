"""
Ollama Local LLM Provider Implementation
"""

import asyncio
import time
import json
from typing import List, Optional, AsyncGenerator, Dict
import httpx
import logging

from app.llm_providers.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamResponse, 
    ModelInfo, ModelType
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama Local LLM Provider"""
    
    def __init__(self, base_url: Optional[str] = None, **kwargs):
        super().__init__("ollama", None, **kwargs)
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.client: Optional[httpx.AsyncClient] = None
        
        # Common local models that might be available
        self._default_models = {
            "llama3.3": {
                "name": "llama3.3",
                "type": ModelType.CHAT,
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0,  # Local models are free
                "supports_streaming": True,
                "supports_functions": False,
                "supports_vision": False
            },
            "llama3.1": {
                "name": "llama3.1",
                "type": ModelType.CHAT,
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0,
                "supports_streaming": True,
                "supports_functions": False,
                "supports_vision": False
            },
            "deepseek-r1": {
                "name": "deepseek-r1",
                "type": ModelType.CHAT,
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0,
                "supports_streaming": True,
                "supports_functions": False,
                "supports_vision": False
            },
            "codellama": {
                "name": "codellama",
                "type": ModelType.CHAT,
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0,
                "supports_streaming": True,
                "supports_functions": False,
                "supports_vision": False
            },
            "mistral": {
                "name": "mistral",
                "type": ModelType.CHAT,
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0,
                "supports_streaming": True,
                "supports_functions": False,
                "supports_vision": False
            },
            "nomic-embed-text": {
                "name": "nomic-embed-text",
                "type": ModelType.EMBEDDING,
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.0,
                "supports_streaming": False,
                "supports_functions": False,
                "supports_vision": False
            }
        }
    
    async def initialize(self) -> None:
        """Initialize Ollama client"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0)
        )
        
        try:
            # Get available models from Ollama
            available_models = await self._get_available_models()
            
            # Initialize model info for available models
            for model_name in available_models:
                if model_name in self._default_models:
                    config = self._default_models[model_name]
                else:
                    # Default config for unknown models
                    config = {
                        "name": model_name,
                        "type": ModelType.CHAT,
                        "max_tokens": 4096,
                        "cost_per_1k_tokens": 0.0,
                        "supports_streaming": True,
                        "supports_functions": False,
                        "supports_vision": False
                    }
                
                self._models[model_name] = ModelInfo(
                    provider=self.name,
                    **config
                )
            
            self._initialized = True
            logger.info(f"Ollama provider initialized with {len(self._models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")
            # Initialize with default models even if Ollama is not available
            for model_name, config in self._default_models.items():
                self._models[model_name] = ModelInfo(
                    provider=self.name,
                    **config
                )
            self._initialized = True
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to get Ollama models: {e}")
            return []
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text completion using Ollama"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare prompt for Ollama
            if request.messages:
                # Convert messages to a single prompt
                prompt_parts = []
                for msg in request.messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                prompt = "\n".join(prompt_parts) + "\nAssistant:"
            else:
                prompt = request.prompt
            
            # Add system message if provided
            if request.system_message and not request.messages:
                prompt = f"System: {request.system_message}\nUser: {prompt}\nAssistant:"
            
            # Prepare request data
            data = {
                "model": request.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                }
            }
            
            if request.stop:
                data["options"]["stop"] = request.stop
            
            # Make API call
            response = await self.client.post("/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=result.get("response", ""),
                model=request.model,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                finish_reason="stop",
                provider=self.name,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamResponse, None]:
        """Generate streaming text completion"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare prompt for Ollama
            if request.messages:
                # Convert messages to a single prompt
                prompt_parts = []
                for msg in request.messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                prompt = "\n".join(prompt_parts) + "\nAssistant:"
            else:
                prompt = request.prompt
            
            # Add system message if provided
            if request.system_message and not request.messages:
                prompt = f"System: {request.system_message}\nUser: {prompt}\nAssistant:"
            
            # Prepare request data
            data = {
                "model": request.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                }
            }
            
            if request.stop:
                data["options"]["stop"] = request.stop
            
            # Make streaming API call
            content = ""
            async with self.client.stream("POST", "/api/generate", json=data) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk_data = json.loads(line)
                            delta = chunk_data.get("response", "")
                            
                            if delta:
                                content += delta
                                yield LLMStreamResponse(
                                    content=content,
                                    delta=delta,
                                    model=request.model,
                                    provider=self.name,
                                    is_complete=False
                                )
                            
                            if chunk_data.get("done", False):
                                yield LLMStreamResponse(
                                    content=content,
                                    delta="",
                                    model=request.model,
                                    provider=self.name,
                                    is_complete=True
                                )
                                break
                                
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get text embedding from Ollama"""
        if not self._initialized:
            await self.initialize()
        
        model = model or "nomic-embed-text"
        
        try:
            data = {
                "model": model,
                "prompt": text
            }
            
            response = await self.client.post("/api/embeddings", json=data)
            response.raise_for_status()
            result = response.json()
            
            return result.get("embedding", [])
            
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise
    
    async def list_models(self) -> List[ModelInfo]:
        """List available Ollama models"""
        if not self._initialized:
            await self.initialize()
        
        return list(self._models.values())
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            if not self.client:
                await self.initialize()
            
            response = await self.client.get("/api/tags")
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
