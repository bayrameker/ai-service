"""
Health check API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
import logging
import asyncio
from datetime import datetime

from app.llm_providers.manager import LLMProviderManager
from app.core.database import get_redis

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_llm_manager(request: Request) -> LLMProviderManager:
    """Get LLM manager from app state"""
    return request.app.state.llm_manager


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Service"
    }


@router.get("/detailed")
async def detailed_health_check(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Detailed health check including all services"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    overall_healthy = True
    
    # Check Redis
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        health_status["services"]["redis"] = {
            "status": "healthy",
            "message": "Connected"
        }
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        overall_healthy = False
    
    # Check LLM Providers
    try:
        provider_health = await llm_manager.health_check()
        health_status["services"]["llm_providers"] = {
            "status": "healthy" if any(provider_health.values()) else "unhealthy",
            "providers": provider_health,
            "healthy_count": sum(1 for status in provider_health.values() if status),
            "total_count": len(provider_health)
        }
        
        if not any(provider_health.values()):
            overall_healthy = False
            
    except Exception as e:
        health_status["services"]["llm_providers"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        overall_healthy = False
    
    # Update overall status
    health_status["status"] = "healthy" if overall_healthy else "unhealthy"
    
    if not overall_healthy:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@router.get("/providers")
async def providers_health(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Check health of LLM providers only"""
    try:
        provider_health = await llm_manager.health_check()
        return {
            "providers": provider_health,
            "healthy_providers": [name for name, status in provider_health.items() if status],
            "unhealthy_providers": [name for name, status in provider_health.items() if not status],
            "summary": {
                "total": len(provider_health),
                "healthy": sum(1 for status in provider_health.values() if status),
                "unhealthy": sum(1 for status in provider_health.values() if not status)
            }
        }
    except Exception as e:
        logger.error(f"Provider health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readiness")
async def readiness_check(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Readiness check for Kubernetes"""
    try:
        # Check if at least one LLM provider is available
        provider_health = await llm_manager.health_check()
        
        if not any(provider_health.values()):
            raise HTTPException(
                status_code=503, 
                detail="No LLM providers available"
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/liveness")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
