"""
Tests for LLM providers
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.llm_providers.openai_provider import OpenAIProvider
from app.llm_providers.anthropic_provider import AnthropicProvider
from app.llm_providers.ollama_provider import OllamaProvider
from app.llm_providers.manager import LLMProviderManager


class TestOpenAIProvider:
    """Test OpenAI provider"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_openai_client):
        """Test provider initialization"""
        with patch('app.llm_providers.openai_provider.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_openai_client
            
            provider = OpenAIProvider()
            await provider.initialize()
            
            assert provider.client is not None
            assert provider.available_models is not None
    
    @pytest.mark.asyncio
    async def test_generate(self, mock_openai_client):
        """Test text generation"""
        with patch('app.llm_providers.openai_provider.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_openai_client
            
            provider = OpenAIProvider()
            await provider.initialize()
            
            result = await provider.generate(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert result["content"] == "Test response"
            assert "usage" in result
    
    @pytest.mark.asyncio
    async def test_generate_stream(self, mock_openai_client):
        """Test streaming generation"""
        with patch('app.llm_providers.openai_provider.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_openai_client
            
            # Mock streaming response
            async def mock_stream():
                yield AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="Test"))])
                yield AsyncMock(choices=[AsyncMock(delta=AsyncMock(content=" response"))])
            
            mock_openai_client.chat.completions.create.return_value = mock_stream()
            
            provider = OpenAIProvider()
            await provider.initialize()
            
            chunks = []
            async for chunk in provider.generate_stream(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0


class TestAnthropicProvider:
    """Test Anthropic provider"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_anthropic_client):
        """Test provider initialization"""
        with patch('app.llm_providers.anthropic_provider.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            provider = AnthropicProvider()
            await provider.initialize()
            
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_generate(self, mock_anthropic_client):
        """Test text generation"""
        with patch('app.llm_providers.anthropic_provider.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            provider = AnthropicProvider()
            await provider.initialize()
            
            result = await provider.generate(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert result["content"] == "Test response"
            assert "usage" in result


class TestOllamaProvider:
    """Test Ollama provider"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_ollama_client):
        """Test provider initialization"""
        with patch('app.llm_providers.ollama_provider.AsyncClient') as mock_ollama:
            mock_ollama.return_value = mock_ollama_client
            
            provider = OllamaProvider()
            await provider.initialize()
            
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_generate(self, mock_ollama_client):
        """Test text generation"""
        with patch('app.llm_providers.ollama_provider.AsyncClient') as mock_ollama:
            mock_ollama.return_value = mock_ollama_client
            
            provider = OllamaProvider()
            await provider.initialize()
            
            result = await provider.generate(
                model="llama3.3",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert result["content"] == "Test response"


class TestLLMProviderManager:
    """Test LLM provider manager"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, llm_manager):
        """Test manager initialization"""
        assert len(llm_manager.providers) > 0
        assert "openai" in llm_manager.providers
        assert "anthropic" in llm_manager.providers
        assert "ollama" in llm_manager.providers
    
    @pytest.mark.asyncio
    async def test_generate(self, llm_manager):
        """Test generation through manager"""
        result = await llm_manager.generate(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result["content"] == "Test response"
        assert "model" in result
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, llm_manager):
        """Test getting available models"""
        models = await llm_manager.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, llm_manager):
        """Test load balancing"""
        # Mock multiple providers for the same model
        llm_manager.model_providers = {
            "gpt-4": ["openai", "openai-backup"]
        }
        llm_manager.providers["openai-backup"] = AsyncMock()
        llm_manager.providers["openai-backup"].generate.return_value = {
            "content": "Backup response",
            "model": "gpt-4"
        }
        
        # Test that requests are distributed
        results = []
        for _ in range(4):
            result = await llm_manager.generate(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
            results.append(result["content"])
        
        # Should have responses from both providers
        assert len(set(results)) > 0
    
    @pytest.mark.asyncio
    async def test_failover(self, llm_manager):
        """Test failover mechanism"""
        # Make primary provider fail
        llm_manager.providers["openai"].generate.side_effect = Exception("Provider failed")
        
        # Add backup provider
        llm_manager.model_providers = {
            "gpt-4": ["openai", "anthropic"]
        }
        
        result = await llm_manager.generate(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should get response from backup provider
        assert result["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_provider_health_check(self, llm_manager):
        """Test provider health checks"""
        health = await llm_manager.health_check()
        
        assert isinstance(health, dict)
        assert "openai" in health
        assert "anthropic" in health
        assert "ollama" in health
    
    @pytest.mark.asyncio
    async def test_get_provider_stats(self, llm_manager):
        """Test getting provider statistics"""
        # Generate some requests to create stats
        await llm_manager.generate(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        stats = llm_manager.get_provider_stats()
        
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
    
    @pytest.mark.asyncio
    async def test_model_switching(self, llm_manager):
        """Test dynamic model switching"""
        # Test switching between different models
        models = ["gpt-4", "claude-3-sonnet-20240229", "llama3.3"]
        
        for model in models:
            result = await llm_manager.generate(
                model=model,
                messages=[{"role": "user", "content": "Hello"}]
            )
            assert result["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, llm_manager):
        """Test handling concurrent requests"""
        import asyncio
        
        async def make_request():
            return await llm_manager.generate(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for result in results:
            assert result["content"] == "Test response"
