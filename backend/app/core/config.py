"""
Configuration settings for AI Service
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str = "postgresql://ai_user:ai_password@localhost:5432/ai_service"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Security
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    ENCRYPTION_KEY: str = "your-encryption-key-change-in-production"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Agent Configuration
    MAX_AGENTS: int = 100
    AGENT_TIMEOUT: int = 300
    MEMORY_RETENTION_DAYS: int = 30
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    GRAFANA_ENABLED: bool = True
    
    # Vector Database
    VECTOR_DB_TYPE: str = "faiss"
    VECTOR_DB_PATH: str = "./data/vector_store"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # A2A Protocol
    A2A_ENABLED: bool = True
    A2A_PORT: int = 8001
    A2A_DISCOVERY_ENABLED: bool = True
    
    # Third Party Integrations
    WEBHOOK_SECRET: str = "your-webhook-secret"
    API_GATEWAY_ENABLED: bool = True
    
    # Performance
    WORKER_PROCESSES: int = 4
    MAX_CONNECTIONS: int = 1000
    CACHE_TTL: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
