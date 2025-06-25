"""
Tests for agent system
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.agents.base import Agent, AgentConfig, Task, AgentRole, TaskPriority
from app.agents.manager import AgentManager
from app.agents.capabilities import capability_registry


class TestAgent:
    """Test Agent class"""
    
    @pytest.mark.asyncio
    async def test_agent_creation(self, sample_agent_config, llm_manager):
        """Test agent creation"""
        config = AgentConfig(**sample_agent_config)
        agent = Agent(config, llm_manager)
        
        assert agent.config.name == "Test Agent"
        assert agent.config.role == AgentRole.GENERAL_ASSISTANT
        assert agent.status.status == "idle"
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, sample_agent_config, llm_manager):
        """Test agent initialization"""
        config = AgentConfig(**sample_agent_config)
        agent = Agent(config, llm_manager)
        
        await agent.initialize()
        
        assert agent.status.status == "idle"
        assert agent.status.initialized is True
    
    @pytest.mark.asyncio
    async def test_task_execution(self, sample_agent_config, sample_task, llm_manager):
        """Test task execution"""
        config = AgentConfig(**sample_agent_config)
        agent = Agent(config, llm_manager)
        await agent.initialize()
        
        task = Task(**sample_task)
        result = await agent.execute_task(task)
        
        assert result is not None
        assert result.success is True
        assert task.status == "completed"
    
    @pytest.mark.asyncio
    async def test_agent_capabilities(self, sample_agent_config, llm_manager):
        """Test agent capabilities"""
        config = AgentConfig(**sample_agent_config)
        agent = Agent(config, llm_manager)
        
        capabilities = agent.get_capabilities()
        assert len(capabilities) > 0
        assert any(cap.name == "text_generation" for cap in capabilities)
    
    @pytest.mark.asyncio
    async def test_agent_status_updates(self, sample_agent_config, llm_manager):
        """Test agent status updates"""
        config = AgentConfig(**sample_agent_config)
        agent = Agent(config, llm_manager)
        
        # Test status changes
        agent.status.status = "busy"
        status = agent.get_status()
        assert status["status"] == "busy"
        
        agent.status.current_task = "test-task"
        status = agent.get_status()
        assert status["current_task"] == "test-task"


class TestAgentManager:
    """Test Agent Manager"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, agent_manager):
        """Test manager initialization"""
        assert agent_manager.llm_manager is not None
        assert len(agent_manager.agents) == 0
    
    @pytest.mark.asyncio
    async def test_create_agent(self, agent_manager, sample_agent_config):
        """Test agent creation through manager"""
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        
        assert agent_id is not None
        assert agent_id in agent_manager.agents
        assert agent_manager.agents[agent_id].config.name == "Test Agent"
    
    @pytest.mark.asyncio
    async def test_remove_agent(self, agent_manager, sample_agent_config):
        """Test agent removal"""
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        
        success = await agent_manager.remove_agent(agent_id)
        assert success is True
        assert agent_id not in agent_manager.agents
    
    @pytest.mark.asyncio
    async def test_list_agents(self, agent_manager, sample_agent_config):
        """Test listing agents"""
        config = AgentConfig(**sample_agent_config)
        await agent_manager.create_agent(config)
        
        agents = agent_manager.list_agents()
        assert len(agents) == 1
        assert agents[0].config.name == "Test Agent"
    
    @pytest.mark.asyncio
    async def test_submit_task(self, agent_manager, sample_task):
        """Test task submission"""
        task = Task(**sample_task)
        task_id = await agent_manager.submit_task(task)
        
        assert task_id is not None
        assert task_id == task.id
    
    @pytest.mark.asyncio
    async def test_assign_task_to_agent(self, agent_manager, sample_agent_config, sample_task):
        """Test task assignment to specific agent"""
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        
        task = Task(**sample_task)
        success = await agent_manager.assign_task_to_agent(task, agent_id)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_best_agent_for_task(self, agent_manager, sample_agent_config, sample_task):
        """Test finding best agent for task"""
        config = AgentConfig(**sample_agent_config)
        await agent_manager.create_agent(config)
        
        task = Task(**sample_task)
        agent = agent_manager.get_best_agent_for_task(task)
        
        assert agent is not None
        assert agent.config.name == "Test Agent"
    
    @pytest.mark.asyncio
    async def test_agent_statistics(self, agent_manager, sample_agent_config):
        """Test agent statistics"""
        config = AgentConfig(**sample_agent_config)
        await agent_manager.create_agent(config)
        
        stats = agent_manager.get_agent_statistics()
        
        assert "total_agents" in stats
        assert "active_agents" in stats
        assert stats["total_agents"] == 1
    
    @pytest.mark.asyncio
    async def test_task_queue_status(self, agent_manager):
        """Test task queue status"""
        status = agent_manager.get_task_queue_status()
        
        assert "pending_tasks" in status
        assert "completed_tasks" in status
        assert "failed_tasks" in status
    
    @pytest.mark.asyncio
    async def test_create_default_agents(self, agent_manager):
        """Test creating default agents"""
        agent_ids = await agent_manager.create_default_agents()
        
        assert len(agent_ids) > 0
        assert len(agent_manager.agents) == len(agent_ids)
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, agent_manager, sample_agent_config):
        """Test concurrent task execution"""
        import asyncio
        
        # Create multiple agents
        config = AgentConfig(**sample_agent_config)
        agent_ids = []
        for i in range(3):
            config.name = f"Test Agent {i}"
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        # Submit multiple tasks
        tasks = []
        for i in range(5):
            task = Task(
                title=f"Test Task {i}",
                description=f"Test task number {i}",
                priority=TaskPriority.NORMAL
            )
            tasks.append(agent_manager.submit_task(task))
        
        # Wait for all tasks to be submitted
        task_ids = await asyncio.gather(*tasks)
        assert len(task_ids) == 5


class TestCapabilityRegistry:
    """Test capability registry"""
    
    def test_list_capabilities(self):
        """Test listing capabilities"""
        capabilities = capability_registry.list_capabilities()
        assert len(capabilities) > 0
        assert any(cap.name == "text_generation" for cap in capabilities)
    
    def test_list_roles(self):
        """Test listing roles"""
        roles = capability_registry.list_roles()
        assert len(roles) > 0
        assert any(role.role == AgentRole.GENERAL_ASSISTANT for role in roles)
    
    def test_get_recommended_setup(self):
        """Test getting recommended setup for role"""
        setup = capability_registry.get_recommended_setup(AgentRole.RESEARCHER)
        
        assert "capabilities" in setup
        assert "llm_model" in setup
        assert len(setup["capabilities"]) > 0
    
    def test_register_capability(self):
        """Test registering new capability"""
        from app.agents.capabilities import AgentCapability
        
        new_capability = AgentCapability(
            name="test_capability",
            description="A test capability",
            category="testing",
            required_tools=["test_tool"]
        )
        
        capability_registry.register_capability(new_capability)
        capabilities = capability_registry.list_capabilities()
        
        assert any(cap.name == "test_capability" for cap in capabilities)
    
    def test_get_capabilities_for_role(self):
        """Test getting capabilities for specific role"""
        capabilities = capability_registry.get_capabilities_for_role(AgentRole.RESEARCHER)
        
        assert len(capabilities) > 0
        assert any(cap.name == "research" for cap in capabilities)
