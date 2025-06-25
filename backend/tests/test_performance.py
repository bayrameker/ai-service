"""
Performance tests for the AI service
"""

import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock
import statistics

from app.agents.base import AgentConfig, Task
from app.memory.episodic import Experience
from app.memory.semantic import Knowledge


class TestLLMProviderPerformance:
    """Test LLM provider performance"""
    
    @pytest.mark.asyncio
    async def test_single_request_latency(self, llm_manager, performance_config):
        """Test single request latency"""
        start_time = time.time()
        
        result = await llm_manager.generate(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        assert result["content"] == "Test response"
        assert latency < performance_config["max_response_time"]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, llm_manager, performance_config):
        """Test concurrent request handling"""
        concurrent_requests = performance_config["concurrent_requests"]
        
        async def make_request():
            start_time = time.time()
            result = await llm_manager.generate(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
            end_time = time.time()
            return result, end_time - start_time
        
        # Make concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all requests completed
        assert len(results) == concurrent_requests
        for result, latency in results:
            assert result[0]["content"] == "Test response"
            assert latency < performance_config["max_response_time"]
        
        # Check throughput
        throughput = concurrent_requests / total_time
        assert throughput > 1.0  # Should handle at least 1 request per second
    
    @pytest.mark.asyncio
    async def test_load_balancing_performance(self, llm_manager):
        """Test load balancing performance"""
        # Add multiple providers for the same model
        llm_manager.model_providers = {
            "gpt-4": ["openai", "openai-backup"]
        }
        llm_manager.providers["openai-backup"] = AsyncMock()
        llm_manager.providers["openai-backup"].generate.return_value = {
            "content": "Backup response",
            "model": "gpt-4"
        }
        
        # Track which provider was used
        provider_usage = {"openai": 0, "openai-backup": 0}
        
        async def track_request():
            result = await llm_manager.generate(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
            if result["content"] == "Test response":
                provider_usage["openai"] += 1
            else:
                provider_usage["openai-backup"] += 1
            return result
        
        # Make multiple requests
        tasks = [track_request() for _ in range(20)]
        await asyncio.gather(*tasks)
        
        # Verify load was distributed
        assert provider_usage["openai"] > 0
        assert provider_usage["openai-backup"] > 0


class TestAgentPerformance:
    """Test agent system performance"""
    
    @pytest.mark.asyncio
    async def test_agent_creation_performance(self, agent_manager, sample_agent_config, performance_config):
        """Test agent creation performance"""
        start_time = time.time()
        
        # Create multiple agents
        agent_ids = []
        for i in range(10):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        assert len(agent_ids) == 10
        assert creation_time < 5.0  # Should create 10 agents in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_task_execution_performance(self, agent_manager, sample_agent_config, sample_task):
        """Test task execution performance"""
        # Create agent
        config = AgentConfig(**sample_agent_config)
        agent_id = await agent_manager.create_agent(config)
        agent = agent_manager.get_agent(agent_id)
        
        # Measure task execution time
        task = Task(**sample_task)
        start_time = time.time()
        
        result = await agent.execute_task(task)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert result.success is True
        assert execution_time < 3.0  # Should complete task in under 3 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, agent_manager, sample_agent_config, sample_task):
        """Test concurrent task execution performance"""
        # Create multiple agents
        agent_ids = []
        for i in range(5):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        # Execute tasks concurrently
        async def execute_task_on_agent(agent_id):
            agent = agent_manager.get_agent(agent_id)
            task = Task(**sample_task)
            start_time = time.time()
            result = await agent.execute_task(task)
            end_time = time.time()
            return result, end_time - start_time
        
        start_time = time.time()
        tasks = [execute_task_on_agent(agent_id) for agent_id in agent_ids]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all tasks completed
        assert len(results) == 5
        for result, execution_time in results:
            assert result[0].success is True
            assert execution_time < 3.0
        
        # Should complete faster than sequential execution
        assert total_time < 10.0  # 5 tasks should complete in under 10 seconds


class TestMemoryPerformance:
    """Test memory system performance"""
    
    @pytest.mark.asyncio
    async def test_experience_storage_performance(self, memory_manager, sample_experience):
        """Test experience storage performance"""
        experiences = []
        for i in range(100):
            exp = Experience(**sample_experience)
            exp.title = f"Experience {i}"
            experiences.append(exp)
        
        start_time = time.time()
        
        # Store experiences
        tasks = [memory_manager.store_experience(exp) for exp in experiences]
        exp_ids = await asyncio.gather(*tasks)
        
        end_time = time.time()
        storage_time = end_time - start_time
        
        assert len(exp_ids) == 100
        assert storage_time < 10.0  # Should store 100 experiences in under 10 seconds
        
        # Calculate throughput
        throughput = 100 / storage_time
        assert throughput > 10.0  # Should handle at least 10 experiences per second
    
    @pytest.mark.asyncio
    async def test_knowledge_retrieval_performance(self, memory_manager, sample_knowledge):
        """Test knowledge retrieval performance"""
        # Store knowledge items
        knowledge_items = []
        for i in range(50):
            knowledge = Knowledge(**sample_knowledge)
            knowledge.subject = f"Subject {i}"
            knowledge.content = f"Content about subject {i}"
            knowledge_items.append(knowledge)
        
        # Store all items
        for knowledge in knowledge_items:
            await memory_manager.store_knowledge(knowledge)
        
        # Test retrieval performance
        start_time = time.time()
        
        # Perform multiple queries
        query_tasks = []
        for i in range(20):
            query_tasks.append(memory_manager.semantic_memory.get_knowledge(f"subject {i}", limit=5))
        
        results = await asyncio.gather(*query_tasks)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert len(results) == 20
        assert query_time < 5.0  # Should complete 20 queries in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_rag_query_performance(self, memory_manager):
        """Test RAG query performance"""
        from app.memory.rag import RAGQuery, RetrievalStrategy
        
        # Mock vector search to return quickly
        memory_manager.rag_system.vector_store.similarity_search_with_score.return_value = [
            (AsyncMock(page_content="Test content", metadata={"source": "test"}), 0.9)
        ]
        
        queries = [
            RAGQuery(question=f"What is topic {i}?", strategy=RetrievalStrategy.SEMANTIC_SEARCH)
            for i in range(10)
        ]
        
        start_time = time.time()
        
        # Execute queries
        tasks = [memory_manager.rag_system.query(query) for query in queries]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert len(responses) == 10
        assert query_time < 3.0  # Should complete 10 RAG queries in under 3 seconds


class TestSystemResourceUsage:
    """Test system resource usage"""
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, agent_manager, memory_manager, sample_agent_config, performance_config):
        """Test memory usage during operations"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create agents and perform operations
        agent_ids = []
        for i in range(10):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        # Store experiences
        for i in range(50):
            experience = Experience(
                type="task_execution",
                title=f"Experience {i}",
                description=f"Test experience {i}",
                success=True
            )
            await memory_manager.store_experience(experience)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < performance_config["max_memory_usage"]
    
    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self, llm_manager, performance_config):
        """Test CPU usage under load"""
        import psutil
        
        # Monitor CPU usage
        cpu_percentages = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Generate load
        async def make_requests():
            tasks = []
            for _ in range(20):
                tasks.append(llm_manager.generate(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                ))
            await asyncio.gather(*tasks)
        
        await make_requests()
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Check CPU usage
        avg_cpu = statistics.mean(cpu_percentages)
        max_cpu = max(cpu_percentages)
        
        # CPU usage should be reasonable
        assert avg_cpu < 80.0  # Average CPU should be under 80%
        assert max_cpu < 95.0  # Peak CPU should be under 95%


class TestScalabilityLimits:
    """Test scalability limits"""
    
    @pytest.mark.asyncio
    async def test_maximum_concurrent_agents(self, agent_manager, sample_agent_config):
        """Test maximum number of concurrent agents"""
        max_agents = 50
        
        start_time = time.time()
        
        # Create maximum number of agents
        agent_ids = []
        for i in range(max_agents):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        assert len(agent_ids) == max_agents
        assert creation_time < 30.0  # Should create 50 agents in under 30 seconds
        
        # Verify all agents are functional
        stats = agent_manager.get_agent_statistics()
        assert stats["total_agents"] == max_agents
        assert stats["active_agents"] == max_agents
    
    @pytest.mark.asyncio
    async def test_memory_system_capacity(self, memory_manager):
        """Test memory system capacity"""
        large_dataset_size = 1000
        
        start_time = time.time()
        
        # Store large number of experiences
        tasks = []
        for i in range(large_dataset_size):
            experience = Experience(
                type="task_execution",
                title=f"Experience {i}",
                description=f"Large dataset experience {i}",
                success=True
            )
            tasks.append(memory_manager.store_experience(experience))
        
        exp_ids = await asyncio.gather(*tasks)
        
        end_time = time.time()
        storage_time = end_time - start_time
        
        assert len(exp_ids) == large_dataset_size
        assert storage_time < 60.0  # Should store 1000 experiences in under 60 seconds
        
        # Test retrieval performance with large dataset
        start_time = time.time()
        results = await memory_manager.search_all_memories("experience", limit=10)
        query_time = time.time() - start_time
        
        assert query_time < 5.0  # Should query large dataset in under 5 seconds


class TestStressTests:
    """Stress tests for the system"""
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, llm_manager, performance_config):
        """Test system under sustained load"""
        test_duration = 10  # seconds
        request_interval = 0.1  # seconds
        
        start_time = time.time()
        completed_requests = 0
        failed_requests = 0
        
        async def sustained_requests():
            nonlocal completed_requests, failed_requests
            
            while time.time() - start_time < test_duration:
                try:
                    await llm_manager.generate(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    completed_requests += 1
                except Exception:
                    failed_requests += 1
                
                await asyncio.sleep(request_interval)
        
        # Run sustained load test
        await sustained_requests()
        
        total_requests = completed_requests + failed_requests
        success_rate = completed_requests / total_requests if total_requests > 0 else 0
        
        # System should handle sustained load well
        assert success_rate > 0.95  # At least 95% success rate
        assert completed_requests > 50  # Should complete at least 50 requests
    
    @pytest.mark.asyncio
    async def test_burst_load(self, agent_manager, sample_agent_config, sample_task):
        """Test system under burst load"""
        burst_size = 20
        
        # Create agents
        agent_ids = []
        for i in range(5):
            config = AgentConfig(**sample_agent_config)
            config.name = f"Agent {i}"
            agent_id = await agent_manager.create_agent(config)
            agent_ids.append(agent_id)
        
        # Create burst of tasks
        start_time = time.time()
        
        async def submit_burst_task():
            task = Task(**sample_task)
            return await agent_manager.submit_task(task)
        
        # Submit burst of tasks
        tasks = [submit_burst_task() for _ in range(burst_size)]
        task_ids = await asyncio.gather(*tasks)
        
        end_time = time.time()
        burst_time = end_time - start_time
        
        assert len(task_ids) == burst_size
        assert burst_time < 5.0  # Should handle burst in under 5 seconds
        
        # Verify system remains stable
        stats = agent_manager.get_agent_statistics()
        assert stats["total_agents"] == 5  # All agents should still be active
