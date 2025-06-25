"""
Main API Router for v1
"""

from fastapi import APIRouter
from app.api.v1.endpoints import llm, models, health, agents, memory, integrations, collaboration, plugins, docs

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(integrations.router, prefix="/integrations", tags=["integrations"])
api_router.include_router(collaboration.router, prefix="/collaboration", tags=["collaboration"])
api_router.include_router(plugins.router, prefix="/plugins", tags=["plugins"])
api_router.include_router(docs.router, prefix="/docs", tags=["documentation"])
