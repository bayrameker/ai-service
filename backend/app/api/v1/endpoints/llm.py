"""
LLM API endpoints with load balancing and model switching
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json
import asyncio
import logging

from app.llm_providers.base import LLMRequest, LLMResponse, ModelType
from app.llm_providers.manager import LLMProviderManager, LoadBalancingStrategy
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_llm_manager(request: Request) -> LLMProviderManager:
    """Get LLM manager from app state"""
    return request.app.state.llm_manager


@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    llm_request: LLMRequest,
    provider: Optional[str] = None,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Generate text completion with automatic provider selection and load balancing"""
    try:
        response = await llm_manager.generate(llm_request, provider_name=provider)
        return response
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_text_stream(
    llm_request: LLMRequest,
    provider: Optional[str] = None,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Generate streaming text completion"""
    try:
        async def stream_generator():
            async for chunk in llm_manager.generate_stream(llm_request, provider_name=provider):
                yield f"data: {json.dumps(chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding")
async def get_embedding(
    text: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Get text embedding"""
    try:
        embedding = await llm_manager.get_embedding(text, model, provider_name=provider)
        return {
            "embedding": embedding,
            "model": model,
            "provider": provider,
            "dimensions": len(embedding)
        }
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch-provider")
async def switch_provider(
    model_type: ModelType = ModelType.CHAT,
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Switch to best provider based on strategy"""
    try:
        best_provider = await llm_manager.select_best_provider(model_type, strategy)
        if not best_provider:
            raise HTTPException(status_code=404, detail="No suitable provider found")
        
        return {
            "selected_provider": best_provider,
            "strategy": strategy,
            "model_type": model_type
        }
    except Exception as e:
        logger.error(f"Provider switching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers/stats")
async def get_provider_stats(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Get provider statistics for load balancing insights"""
    try:
        stats = llm_manager.get_stats()
        health_status = await llm_manager.health_check()
        
        return {
            "stats": stats,
            "health": health_status,
            "total_providers": len(llm_manager.providers),
            "active_providers": sum(1 for status in health_status.values() if status)
        }
    except Exception as e:
        logger.error(f"Failed to get provider stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/providers/health-check")
async def check_providers_health(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Check health of all providers"""
    try:
        health_status = await llm_manager.health_check()
        return {
            "health_status": health_status,
            "healthy_providers": [name for name, status in health_status.items() if status],
            "unhealthy_providers": [name for name, status in health_status.items() if not status]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cost-estimate")
async def estimate_cost(
    llm_request: LLMRequest,
    provider: Optional[str] = None,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Estimate cost for a request"""
    try:
        # Get provider
        if provider:
            provider_obj = llm_manager.get_provider(provider)
        else:
            provider_obj = llm_manager.get_provider_for_model(llm_request.model)
        
        if not provider_obj:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        estimated_cost = await provider_obj.estimate_cost(llm_request)
        
        return {
            "estimated_cost": estimated_cost,
            "currency": "USD",
            "model": llm_request.model,
            "provider": provider_obj.name,
            "note": "This is an estimate and actual costs may vary"
        }
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate")
async def batch_generate(
    requests: List[LLMRequest],
    provider: Optional[str] = None,
    max_concurrent: int = 5,
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Generate multiple completions concurrently"""
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(req: LLMRequest):
            async with semaphore:
                return await llm_manager.generate(req, provider_name=provider)
        
        # Execute all requests concurrently
        tasks = [generate_single(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(response)
                })
            else:
                results.append({
                    "index": i,
                    "success": True,
                    "response": response.dict()
                })
        
        return {
            "results": results,
            "total_requests": len(requests),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
