"""
Base LLM Provider interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from pydantic import BaseModel
from enum import Enum
import asyncio


class ModelType(str, Enum):
    """LLM Model types"""
    TEXT = "text"
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"


class LLMRequest(BaseModel):
    """LLM request model"""
    prompt: str
    model: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    system_message: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None


class LLMResponse(BaseModel):
    """LLM response model"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    provider: str
    request_id: Optional[str] = None
    latency_ms: Optional[int] = None


class LLMStreamResponse(BaseModel):
    """LLM streaming response model"""
    content: str
    delta: str
    model: str
    provider: str
    is_complete: bool = False
    request_id: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    provider: str
    type: ModelType
    max_tokens: int
    cost_per_1k_tokens: Optional[float] = None
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_vision: bool = False


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, name: str, api_key: Optional[str] = None, **kwargs):
        self.name = name
        self.api_key = api_key
        self.config = kwargs
        self._models: Dict[str, ModelInfo] = {}
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamResponse, None]:
        """Generate streaming text completion"""
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get text embedding"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models"""
        pass
    
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self._models.get(model_name)
    
    def supports_model(self, model_name: str) -> bool:
        """Check if provider supports a model"""
        return model_name in self._models
    
    async def estimate_cost(self, request: LLMRequest) -> Optional[float]:
        """Estimate cost for a request"""
        model_info = self.get_model_info(request.model)
        if not model_info or not model_info.cost_per_1k_tokens:
            return None
        
        # Rough estimation based on prompt length
        estimated_tokens = len(request.prompt.split()) * 1.3  # Rough token estimation
        if request.max_tokens:
            estimated_tokens += request.max_tokens
        
        return (estimated_tokens / 1000) * model_info.cost_per_1k_tokens
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
