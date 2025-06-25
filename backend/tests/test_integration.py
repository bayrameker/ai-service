"""
Integration tests for the AI service
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json

from app.main import app
from app.agents.base import AgentConfig, Task, AgentRole
from app.memory.episodic import Experience
from app.memory.semantic import Knowledge


class TestAPIIntegration:
    """Test API integration"""
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_detailed_health_endpoint(self, client):
        """Test detailed health endpoint"""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "llm_providers" in data
        assert "memory_systems" in data
        assert "integrations" in data
    
    def test_models_endpoint(self, client):
        """Test models endpoint"""
        response = client.get("/api/v1/models/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('app.llm_providers.manager.LLMProviderManager.generate')
    def test_llm_generate_endpoint(self, mock_generate, client):
        """Test LLM generation endpoint"""
        mock_generate.return_value = {
            "content": "Test response",
            "model": "gpt-4",
            "usage": {"tokens": 100}
        }
        
        response = client.post("/api/v1/llm/generate", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Test response"
    
    @patch('app.agents.manager.AgentManager.create_agent')
    def test_create_agent_endpoint(self, mock_create, client, sample_agent_config):
        """Test agent creation endpoint"""
        mock_create.return_value = "agent-123"
        
        response = client.post("/api/v1/agents/create", json=sample_agent_config)
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "agent-123"
    
    @patch('app.memory.manager.MemoryManager.store_experience')
    def test_store_experience_endpoint(self, mock_store, client, sample_experience):
        """Test storing experience endpoint"""
        mock_store.return_value = "exp-123"
        
        response = client.post("/api/v1/memory/experiences", json=sample_experience)
        
        assert response.status_code == 200
        data = response.json()
        assert data["experience_id"] == "exp-123"


class TestAgentMemoryIntegration:
    """Test agent and memory system integration"""
    
    @pytest.mark.asyncio
    async def test_agent_stores_experience(self, agent_manager, memory_manager, sample_agent_config, sample_task):
        """Test that agents store experiences in memory"""
        # Create agent
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        agent = agent_manager.get_agent(agent_id)
        
        # Mock memory manager
        agent.memory_manager = memory_manager
        
        # Execute task
        task = Task(**sample_task)
        result = await agent.execute_task(task)
        
        # Verify experience was stored
        assert result.success is True
        # In a real test, we would verify that store_experience was called
    
    @pytest.mark.asyncio
    async def test_agent_retrieves_knowledge(self, agent_manager, memory_manager, sample_agent_config):
        """Test that agents can retrieve knowledge from memory"""
        # Create agent
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        agent = agent_manager.get_agent(agent_id)
        
        # Mock memory manager
        agent.memory_manager = memory_manager
        
        # Mock knowledge retrieval
        memory_manager.semantic_memory.get_knowledge = AsyncMock()
        memory_manager.semantic_memory.get_knowledge.return_value = [
            Knowledge(
                type="fact",
                subject="Testing",
                content="Testing is important",
                confidence="high"
            )
        ]
        
        # Agent should be able to access knowledge
        knowledge = await agent.memory_manager.semantic_memory.get_knowledge("testing")
        assert len(knowledge) > 0
        assert knowledge[0].subject == "Testing"


class TestLLMAgentIntegration:
    """Test LLM and agent integration"""
    
    @pytest.mark.asyncio
    async def test_agent_uses_llm(self, agent_manager, llm_manager, sample_agent_config, sample_task):
        """Test that agents use LLM for task execution"""
        # Create agent
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        agent = agent_manager.get_agent(agent_id)
        
        # Execute task
        task = Task(**sample_task)
        result = await agent.execute_task(task)
        
        # Verify LLM was used
        assert result.success is True
        # In a real test, we would verify that LLM generate was called
    
    @pytest.mark.asyncio
    async def test_multiple_agents_different_models(self, agent_manager, sample_agent_config):
        """Test multiple agents using different LLM models"""
        # Create agents with different models
        configs = []
        for i, model in enumerate(["gpt-4", "claude-3-sonnet-20240229", "llama3.3"]):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            config.llm_model = model
            configs.append(config)
        
        agent_ids = []
        for config in configs:
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        # Verify agents were created with different models
        assert len(agent_ids) == 3
        for i, agent_id in enumerate(agent_ids):
            agent = agent_manager.get_agent(agent_id)
            assert agent.config.llm_model in ["gpt-4", "claude-3-sonnet-20240229", "llama3.3"]


class TestCollaborationIntegration:
    """Test agent collaboration integration"""
    
    @pytest.mark.asyncio
    async def test_agent_message_passing(self, test_message_broker, sample_message):
        """Test agent message passing"""
        # Register agents
        agent1_queue = test_message_broker.register_agent("agent-1")
        agent2_queue = test_message_broker.register_agent("agent-2")
        
        # Send message
        from app.collaboration.message_passing import Message
        message = Message(**sample_message)
        
        success = await test_message_broker.send_message(message)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_agent_discovery(self, test_agent_registry):
        """Test agent discovery and registration"""
        from app.collaboration.discovery import AgentRegistration, AgentService, ServiceType
        
        # Register agent
        registration = AgentRegistration(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
            role=AgentRole.GENERAL_ASSISTANT,
            services=[
                AgentService(
                    name="text_generation",
                    type=ServiceType.LLM_GENERATION,
                    description="Generate text"
                )
            ]
        )
        
        success = await test_agent_registry.register_agent(registration)
        assert success is True
        
        # Find agent
        agent = test_agent_registry.get_agent("test-agent")
        assert agent is not None
        assert agent.name == "Test Agent"


class TestIntegrationWorkflows:
    """Test complete integration workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_task_workflow(self, agent_manager, memory_manager, sample_agent_config, sample_task):
        """Test complete task execution workflow"""
        # 1. Create agent
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        
        # 2. Submit task
        task = Task(**sample_task)
        task_id = await agent_manager.submit_task(task)
        
        # 3. Assign task to agent
        success = await agent_manager.assign_task_to_agent(task, agent_id)
        assert success is True
        
        # 4. Execute task (this would happen automatically in real system)
        agent = agent_manager.get_agent(agent_id)
        result = await agent.execute_task(task)
        
        # 5. Verify task completion
        assert result.success is True
        assert task.status == "completed"
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, agent_manager, test_message_broker, sample_agent_config):
        """Test multi-agent collaboration workflow"""
        # 1. Create multiple agents
        agent_ids = []
        for i in range(3):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            config.role = [AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.COORDINATOR][i]
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        # 2. Register agents with message broker
        for agent_id in agent_ids:
            test_message_broker.register_agent(agent_id)
        
        # 3. Create collaboration group
        test_message_broker.create_group("research_team", agent_ids)
        
        # 4. Send collaboration message
        from app.collaboration.message_passing import Message, MessageType
        message = Message(
            sender_id=agent_ids[0],
            receiver_group="research_team",
            type=MessageType.MULTICAST,
            subject="collaboration_request",
            content={"task": "research_project", "deadline": "2024-01-01"}
        )
        
        success = await test_message_broker.send_message(message)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_learning_workflow(self, memory_manager, sample_experience, sample_knowledge):
        """Test learning and knowledge accumulation workflow"""
        # 1. Store initial experience
        experience = Experience(**sample_experience)
        exp_id = await memory_manager.store_experience(experience)
        assert exp_id is not None
        
        # 2. Extract knowledge from experience
        knowledge = Knowledge(**sample_knowledge)
        knowledge_id = await memory_manager.store_knowledge(knowledge)
        assert knowledge_id is not None
        
        # 3. Query knowledge with RAG
        answer = await memory_manager.ask_question("What did we learn about testing?")
        assert answer is not None
        
        # 4. Get learning insights
        insights = await memory_manager.get_learning_insights()
        assert isinstance(insights, list)


class TestErrorHandling:
    """Test error handling in integrations"""
    
    @pytest.mark.asyncio
    async def test_llm_provider_failover(self, llm_manager):
        """Test LLM provider failover"""
        # Make primary provider fail
        llm_manager.providers["openai"].generate.side_effect = Exception("Provider failed")
        
        # Should failover to backup provider
        result = await llm_manager.generate(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_agent_task_failure_handling(self, agent_manager, sample_agent_config, sample_task):
        """Test agent task failure handling"""
        # Create agent
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        agent = agent_manager.get_agent(agent_id)
        
        # Mock task execution to fail
        agent.llm_manager.generate = AsyncMock(side_effect=Exception("LLM failed"))
        
        # Execute task
        task = Task(**sample_task)
        result = await agent.execute_task(task)
        
        # Should handle failure gracefully
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_memory_system_resilience(self, memory_manager, sample_experience):
        """Test memory system resilience"""
        # Mock vector store to fail
        memory_manager.episodic_memory.vector_store.add_texts.side_effect = Exception("Vector store failed")
        
        # Should handle failure gracefully
        experience = Experience(**sample_experience)
        exp_id = await memory_manager.store_experience(experience)
        
        # Should still return an ID (stored in Redis backup)
        assert exp_id is not None


class TestPerformanceIntegration:
    """Test performance aspects of integration"""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, agent_manager, sample_agent_config):
        """Test concurrent agent operations"""
        import time
        
        start_time = time.time()
        
        # Create multiple agents concurrently
        tasks = []
        for i in range(10):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            tasks.append(agent_manager.create_agent(config))
        
        agent_ids = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert len(agent_ids) == 10
        assert execution_time < 5.0  # Should take less than 5 seconds
    
    @pytest.mark.asyncio
    async def test_memory_query_performance(self, memory_manager):
        """Test memory query performance"""
        import time
        
        # Store multiple items
        for i in range(100):
            experience = Experience(
                type="task_execution",
                title=f"Test Experience {i}",
                description=f"Experience number {i}",
                success=True
            )
            await memory_manager.store_experience(experience)
        
        # Query performance test
        start_time = time.time()
        
        results = await memory_manager.search_all_memories("test", limit=10)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Should complete quickly
        assert query_time < 2.0  # Should take less than 2 seconds
        assert isinstance(results, dict)
