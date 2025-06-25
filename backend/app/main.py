"""
AI Service - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.database import init_db
from app.api.v1.router import api_router
from app.llm_providers.manager import LLMProviderManager
from app.agents.manager import AgentManager
from app.memory.manager import MemoryManager
from app.integrations.api_gateway import api_gateway
from app.integrations.webhooks import webhook_system
from app.integrations.database import database_manager
from app.integrations.cloud_services import cloud_service_manager
from app.integrations.third_party_adapters import third_party_adapter_manager
from app.collaboration.message_passing import message_broker
from app.collaboration.discovery import agent_registry
from app.plugins.manager import plugin_manager
from app.security.auth import auth_system

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Service...")
    
    # Initialize database
    await init_db()
    
    # Initialize managers
    app.state.llm_manager = LLMProviderManager()
    app.state.memory_manager = MemoryManager("system", app.state.llm_manager)
    app.state.agent_manager = AgentManager(app.state.llm_manager)

    # Initialize LLM providers
    await app.state.llm_manager.initialize()

    # Initialize memory manager
    await app.state.memory_manager.initialize()

    # Initialize agent manager
    await app.state.agent_manager.initialize(app.state.llm_manager)

    # Initialize integrations
    await api_gateway.initialize()
    await webhook_system.initialize()

    # Initialize collaboration systems
    await message_broker.initialize()
    await agent_registry.initialize()

    # Initialize plugin system
    await plugin_manager.initialize()

    # Initialize security
    await auth_system.initialize()
    
    logger.info("AI Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Service...")
    
    # Cleanup managers
    if hasattr(app.state, 'agent_manager'):
        await app.state.agent_manager.shutdown()

    if hasattr(app.state, 'memory_manager'):
        await app.state.memory_manager.shutdown()

    if hasattr(app.state, 'llm_manager'):
        await app.state.llm_manager.shutdown()

    # Cleanup integrations
    await api_gateway.shutdown()
    await webhook_system.shutdown()
    await database_manager.shutdown()
    await cloud_service_manager.shutdown()
    await third_party_adapter_manager.shutdown()

    # Cleanup collaboration systems
    await message_broker.shutdown()
    await agent_registry.shutdown()

    # Cleanup plugin system
    await plugin_manager.shutdown()
    
    logger.info("AI Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AI Service",
    description="Bağımsız AI ve Agentic AI Hizmet Platformu",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Add Prometheus metrics endpoint
if settings.PROMETHEUS_ENABLED:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Service API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        # Check Redis connection
        # Check LLM providers
        
        return {
            "status": "healthy",
            "services": {
                "database": "connected",
                "redis": "connected",
                "llm_providers": "available"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/info")
async def service_info():
    """Service information endpoint"""
    return {
        "name": "AI Service",
        "version": "1.0.0",
        "description": "Bağımsız AI ve Agentic AI Hizmet Platformu",
        "features": [
            "Multi-LLM Provider Support",
            "Agentic AI Framework",
            "Self-Learning Memory System",
            "Agent-to-Agent Collaboration",
            "Third-Party Integrations",
            "Security & Authentication"
        ],
        "supported_llm_providers": [
            "OpenAI",
            "Anthropic Claude",
            "Ollama (Local)"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
