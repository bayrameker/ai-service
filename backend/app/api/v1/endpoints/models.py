"""
Models API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
import logging

from app.llm_providers.base import ModelInfo, ModelType
from app.llm_providers.manager import LLMProviderManager

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_llm_manager(request: Request) -> LLMProviderManager:
    """Get LLM manager from app state"""
    return request.app.state.llm_manager


@router.get("/", response_model=List[ModelInfo])
async def list_all_models(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """List all available models from all providers"""
    try:
        models = await llm_manager.list_all_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-type/{model_type}", response_model=List[ModelInfo])
async def list_models_by_type(
    model_type: ModelType,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """List models by type (chat, embedding, etc.)"""
    try:
        models = await llm_manager.list_models_by_type(model_type)
        return models
    except Exception as e:
        logger.error(f"Failed to list models by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-provider/{provider_name}", response_model=List[ModelInfo])
async def list_models_by_provider(
    provider_name: str,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """List models by provider"""
    try:
        provider = llm_manager.get_provider(provider_name)
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        
        models = await provider.list_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models by provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}")
async def get_model_info(
    model_name: str,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Get information about a specific model"""
    try:
        provider = llm_manager.get_provider_for_model(model_name)
        if not provider:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        model_info = provider.get_model_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return {
            "model": model_info,
            "provider": provider.name,
            "available": True
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers/summary")
async def get_providers_summary(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Get summary of all providers and their models"""
    try:
        summary = {}
        
        for provider_name, provider in llm_manager.providers.items():
            try:
                models = await provider.list_models()
                health = await provider.health_check()
                
                summary[provider_name] = {
                    "name": provider_name,
                    "healthy": health,
                    "model_count": len(models),
                    "models_by_type": {
                        "chat": len([m for m in models if m.type == ModelType.CHAT]),
                        "embedding": len([m for m in models if m.type == ModelType.EMBEDDING]),
                        "image": len([m for m in models if m.type == ModelType.IMAGE]),
                        "audio": len([m for m in models if m.type == ModelType.AUDIO])
                    },
                    "features": {
                        "streaming": any(m.supports_streaming for m in models),
                        "functions": any(m.supports_functions for m in models),
                        "vision": any(m.supports_vision for m in models)
                    }
                }
            except Exception as e:
                logger.error(f"Failed to get summary for {provider_name}: {e}")
                summary[provider_name] = {
                    "name": provider_name,
                    "healthy": False,
                    "error": str(e)
                }
        
        return {
            "providers": summary,
            "total_providers": len(llm_manager.providers),
            "healthy_providers": sum(1 for p in summary.values() if p.get("healthy", False))
        }
    except Exception as e:
        logger.error(f"Failed to get providers summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
