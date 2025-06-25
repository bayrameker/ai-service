"""
Test configuration and fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
import tempfile
import os

from fastapi.testclient import TestClient
from app.main import app
from app.llm_providers.manager import LLMProviderManager
from app.agents.manager import AgentManager
from app.memory.manager import MemoryManager
from app.integrations.api_gateway import api_gateway
from app.integrations.webhooks import webhook_system
from app.collaboration.message_passing import message_broker
from app.collaboration.discovery import agent_registry
from app.plugins.manager import plugin_manager
from app.security.auth import auth_system


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> TestClient:
    """Create test client"""
    return TestClient(app)


@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.keys.return_value = []
    return mock_redis


@pytest.fixture
async def llm_manager(mock_redis) -> AsyncGenerator[LLMProviderManager, None]:
    """Create LLM manager for testing"""
    manager = LLMProviderManager()
    
    # Mock the providers to avoid actual API calls
    manager.providers = {
        "openai": AsyncMock(),
        "anthropic": AsyncMock(),
        "ollama": AsyncMock()
    }
    
    # Mock generate methods
    for provider in manager.providers.values():
        provider.generate.return_value = {
            "content": "Test response",
            "model": "test-model",
            "usage": {"tokens": 100}
        }
        provider.generate_stream.return_value = AsyncMock()
        provider.is_available.return_value = True
    
    yield manager


@pytest.fixture
async def memory_manager(llm_manager, mock_redis) -> AsyncGenerator[MemoryManager, None]:
    """Create memory manager for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set temporary vector DB path
        os.environ["VECTOR_DB_PATH"] = temp_dir
        
        manager = MemoryManager("test_agent", llm_manager)
        
        # Mock vector stores to avoid file operations
        manager.episodic_memory.vector_store = AsyncMock()
        manager.semantic_memory.vector_store = AsyncMock()
        manager.rag_system.vector_store = AsyncMock()
        
        # Mock Redis
        manager.redis_client = mock_redis
        manager.episodic_memory.redis_client = mock_redis
        manager.semantic_memory.redis_client = mock_redis
        manager.working_memory.redis_client = mock_redis
        
        yield manager


@pytest.fixture
async def agent_manager(llm_manager) -> AsyncGenerator[AgentManager, None]:
    """Create agent manager for testing"""
    manager = AgentManager(llm_manager)
    
    # Mock task queue
    manager.task_queue = AsyncMock()
    manager.workflow_engine = AsyncMock()
    
    yield manager


@pytest.fixture
async def test_api_gateway():
    """Create API gateway for testing"""
    gateway = api_gateway
    gateway.http_client = AsyncMock()
    yield gateway


@pytest.fixture
async def test_webhook_system(mock_redis):
    """Create webhook system for testing"""
    system = webhook_system
    system.redis_client = mock_redis
    yield system


@pytest.fixture
async def test_message_broker(mock_redis):
    """Create message broker for testing"""
    broker = message_broker
    broker.redis_client = mock_redis
    yield broker


@pytest.fixture
async def test_agent_registry(mock_redis):
    """Create agent registry for testing"""
    registry = agent_registry
    registry.redis_client = mock_redis
    yield registry


@pytest.fixture
async def test_plugin_manager():
    """Create plugin manager for testing"""
    manager = plugin_manager
    # Clear any existing plugins
    manager.plugins.clear()
    yield manager


@pytest.fixture
async def test_auth_system(mock_redis):
    """Create auth system for testing"""
    auth = auth_system
    auth.redis_client = mock_redis
    # Clear existing users for clean tests
    auth.users.clear()
    auth.sessions.clear()
    auth.api_keys.clear()
    yield auth


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing"""
    return {
        "name": "Test Agent",
        "description": "A test agent",
        "role": "general_assistant",
        "llm_model": "gpt-4",
        "capabilities": ["text_generation", "analysis"],
        "max_iterations": 5,
        "timeout": 300
    }


@pytest.fixture
def sample_task():
    """Sample task for testing"""
    return {
        "title": "Test Task",
        "description": "A test task",
        "priority": "normal",
        "requirements": ["text_generation"],
        "context": {"test": True}
    }


@pytest.fixture
def sample_experience():
    """Sample experience for testing"""
    return {
        "type": "task_execution",
        "title": "Test Experience",
        "description": "A test experience",
        "context": {"task_id": "test-123"},
        "outcome": {"success": True, "result": "Test completed"},
        "success": True,
        "lessons_learned": ["Always test thoroughly"]
    }


@pytest.fixture
def sample_knowledge():
    """Sample knowledge for testing"""
    return {
        "type": "fact",
        "subject": "Testing",
        "content": "Testing is important for software quality",
        "confidence": "high",
        "source": "test_suite",
        "verified": True
    }


@pytest.fixture
def sample_message():
    """Sample message for testing"""
    return {
        "sender_id": "agent-1",
        "receiver_id": "agent-2",
        "type": "direct",
        "priority": "normal",
        "subject": "test_message",
        "content": {"message": "Hello, this is a test"}
    }


@pytest.fixture
def sample_webhook_config():
    """Sample webhook configuration for testing"""
    return {
        "name": "Test Webhook",
        "description": "A test webhook",
        "url": "https://example.com/webhook",
        "events": ["task.completed"],
        "secret": "test-secret"
    }


@pytest.fixture
def sample_api_endpoint():
    """Sample API endpoint configuration for testing"""
    return {
        "name": "Test API",
        "description": "A test API endpoint",
        "base_url": "https://api.example.com",
        "path": "/test",
        "method": "GET",
        "auth_type": "api_key",
        "auth_config": {"api_key": "test-key"}
    }


@pytest.fixture
def sample_user():
    """Sample user for testing"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "role": "user"
    }


# Async test helpers
@pytest.fixture
def async_test():
    """Decorator for async tests"""
    def decorator(func):
        return pytest.mark.asyncio(func)
    return decorator


# Mock external services
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = AsyncMock(
        choices=[
            AsyncMock(
                message=AsyncMock(content="Test response"),
                finish_reason="stop"
            )
        ],
        usage=AsyncMock(total_tokens=100)
    )
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    mock_client = AsyncMock()
    mock_client.messages.create.return_value = AsyncMock(
        content=[AsyncMock(text="Test response")],
        usage=AsyncMock(input_tokens=50, output_tokens=50)
    )
    return mock_client


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client"""
    mock_client = AsyncMock()
    mock_client.chat.return_value = {
        "message": {"content": "Test response"},
        "done": True
    }
    return mock_client


# Database fixtures
@pytest.fixture
async def mock_vector_store():
    """Mock vector store"""
    mock_store = AsyncMock()
    mock_store.add_texts.return_value = ["doc-1", "doc-2"]
    mock_store.similarity_search.return_value = [
        AsyncMock(page_content="Test content", metadata={"id": "doc-1"})
    ]
    mock_store.similarity_search_with_score.return_value = [
        (AsyncMock(page_content="Test content", metadata={"id": "doc-1"}), 0.9)
    ]
    return mock_store


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests"""
    return {
        "max_response_time": 5.0,  # seconds
        "max_memory_usage": 500,   # MB
        "concurrent_requests": 10,
        "test_duration": 30        # seconds
    }


# Integration test fixtures
@pytest.fixture
async def integration_test_setup():
    """Setup for integration tests"""
    # This would set up a test database, Redis instance, etc.
    # For now, we'll use mocks
    return {
        "database_url": "sqlite:///test.db",
        "redis_url": "redis://localhost:6379/1",
        "test_mode": True
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup code here if needed
    pass
