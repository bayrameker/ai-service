"""
Tests for memory systems
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.memory.episodic import EpisodicMemory, Experience, ExperienceType
from app.memory.semantic import SemanticMemory, Knowledge, KnowledgeType
from app.memory.working import WorkingMemory, ContextType, Priority
from app.memory.rag import RAGSystem, RAGQuery, RetrievalStrategy
from app.memory.manager import MemoryManager


class TestEpisodicMemory:
    """Test episodic memory system"""
    
    @pytest.mark.asyncio
    async def test_store_experience(self, memory_manager, sample_experience):
        """Test storing experience"""
        experience = Experience(**sample_experience)
        experience_id = await memory_manager.episodic_memory.store_experience(experience)
        
        assert experience_id is not None
        assert experience_id == experience.id
    
    @pytest.mark.asyncio
    async def test_get_experiences(self, memory_manager, sample_experience):
        """Test retrieving experiences"""
        experience = Experience(**sample_experience)
        await memory_manager.episodic_memory.store_experience(experience)
        
        # Mock vector search
        memory_manager.episodic_memory.vector_store.similarity_search.return_value = [
            AsyncMock(page_content=experience.json(), metadata={"id": experience.id})
        ]
        
        experiences = await memory_manager.episodic_memory.get_experiences("test", limit=5)
        
        assert len(experiences) > 0
        assert experiences[0].title == "Test Experience"
    
    @pytest.mark.asyncio
    async def test_get_experiences_by_type(self, memory_manager, sample_experience):
        """Test retrieving experiences by type"""
        experience = Experience(**sample_experience)
        await memory_manager.episodic_memory.store_experience(experience)
        
        experiences = await memory_manager.episodic_memory.get_experiences_by_type(
            ExperienceType.TASK_EXECUTION
        )
        
        # Should return empty list since we're using mocks
        assert isinstance(experiences, list)
    
    @pytest.mark.asyncio
    async def test_get_recent_experiences(self, memory_manager, sample_experience):
        """Test retrieving recent experiences"""
        experience = Experience(**sample_experience)
        await memory_manager.episodic_memory.store_experience(experience)
        
        experiences = await memory_manager.episodic_memory.get_recent_experiences(limit=5)
        
        assert isinstance(experiences, list)


class TestSemanticMemory:
    """Test semantic memory system"""
    
    @pytest.mark.asyncio
    async def test_store_knowledge(self, memory_manager, sample_knowledge):
        """Test storing knowledge"""
        knowledge = Knowledge(**sample_knowledge)
        knowledge_id = await memory_manager.semantic_memory.store_knowledge(knowledge)
        
        assert knowledge_id is not None
        assert knowledge_id == knowledge.id
    
    @pytest.mark.asyncio
    async def test_get_knowledge(self, memory_manager, sample_knowledge):
        """Test retrieving knowledge"""
        knowledge = Knowledge(**sample_knowledge)
        await memory_manager.semantic_memory.store_knowledge(knowledge)
        
        # Mock vector search
        memory_manager.semantic_memory.vector_store.similarity_search.return_value = [
            AsyncMock(page_content=knowledge.json(), metadata={"id": knowledge.id})
        ]
        
        knowledge_items = await memory_manager.semantic_memory.get_knowledge("testing", limit=5)
        
        assert len(knowledge_items) > 0
        assert knowledge_items[0].subject == "Testing"
    
    @pytest.mark.asyncio
    async def test_get_knowledge_by_type(self, memory_manager, sample_knowledge):
        """Test retrieving knowledge by type"""
        knowledge = Knowledge(**sample_knowledge)
        await memory_manager.semantic_memory.store_knowledge(knowledge)
        
        knowledge_items = await memory_manager.semantic_memory.get_knowledge_by_type(
            KnowledgeType.FACT
        )
        
        assert isinstance(knowledge_items, list)
    
    @pytest.mark.asyncio
    async def test_update_knowledge(self, memory_manager, sample_knowledge):
        """Test updating knowledge"""
        knowledge = Knowledge(**sample_knowledge)
        await memory_manager.semantic_memory.store_knowledge(knowledge)
        
        # Update knowledge
        knowledge.content = "Updated content"
        success = await memory_manager.semantic_memory.update_knowledge(knowledge)
        
        assert success is True


class TestWorkingMemory:
    """Test working memory system"""
    
    @pytest.mark.asyncio
    async def test_store_context(self, memory_manager):
        """Test storing context"""
        context_id = await memory_manager.working_memory.store_context(
            "test_key", 
            {"data": "test_value"}, 
            context_type=ContextType.TEMPORARY
        )
        
        assert context_id is not None
    
    @pytest.mark.asyncio
    async def test_get_context(self, memory_manager):
        """Test retrieving context"""
        await memory_manager.working_memory.store_context(
            "test_key", 
            {"data": "test_value"}
        )
        
        # Mock Redis get
        memory_manager.working_memory.redis_client.get.return_value = '{"data": "test_value"}'
        
        value = await memory_manager.working_memory.get_context("test_key")
        
        assert value is not None
        assert value["data"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_add_conversation_turn(self, memory_manager):
        """Test adding conversation turn"""
        turn_id = await memory_manager.working_memory.add_conversation_turn(
            "user", 
            "Hello, how are you?"
        )
        
        assert turn_id is not None
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, memory_manager):
        """Test getting conversation history"""
        await memory_manager.working_memory.add_conversation_turn("user", "Hello")
        await memory_manager.working_memory.add_conversation_turn("assistant", "Hi there!")
        
        history = await memory_manager.working_memory.get_conversation_history(limit=10)
        
        assert isinstance(history, list)
    
    @pytest.mark.asyncio
    async def test_context_expiration(self, memory_manager):
        """Test context expiration"""
        # Store context with short TTL
        await memory_manager.working_memory.store_context(
            "temp_key", 
            {"data": "temp_value"}, 
            ttl=1
        )
        
        # Mock Redis to return None (expired)
        memory_manager.working_memory.redis_client.get.return_value = None
        
        value = await memory_manager.working_memory.get_context("temp_key")
        assert value is None


class TestRAGSystem:
    """Test RAG system"""
    
    @pytest.mark.asyncio
    async def test_query_with_rag(self, memory_manager):
        """Test RAG query"""
        query = RAGQuery(
            question="What is testing?",
            strategy=RetrievalStrategy.SEMANTIC_SEARCH,
            max_results=5
        )
        
        # Mock vector search results
        memory_manager.rag_system.vector_store.similarity_search_with_score.return_value = [
            (AsyncMock(page_content="Testing is important", metadata={"source": "test"}), 0.9)
        ]
        
        response = await memory_manager.rag_system.query(query)
        
        assert response.answer is not None
        assert len(response.sources) > 0
    
    @pytest.mark.asyncio
    async def test_add_documents(self, memory_manager):
        """Test adding documents to RAG"""
        documents = [
            {"content": "Testing is important for software quality", "metadata": {"source": "test1"}},
            {"content": "Unit tests verify individual components", "metadata": {"source": "test2"}}
        ]
        
        doc_ids = await memory_manager.rag_system.add_documents(documents)
        
        assert len(doc_ids) == 2
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, memory_manager):
        """Test hybrid search strategy"""
        query = RAGQuery(
            question="What is testing?",
            strategy=RetrievalStrategy.HYBRID,
            max_results=5
        )
        
        # Mock both semantic and keyword search
        memory_manager.rag_system.vector_store.similarity_search_with_score.return_value = [
            (AsyncMock(page_content="Testing is important", metadata={"source": "test"}), 0.9)
        ]
        
        response = await memory_manager.rag_system.query(query)
        
        assert response.answer is not None


class TestMemoryManager:
    """Test memory manager"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, memory_manager):
        """Test memory manager initialization"""
        assert memory_manager.agent_id == "test_agent"
        assert memory_manager.episodic_memory is not None
        assert memory_manager.semantic_memory is not None
        assert memory_manager.working_memory is not None
        assert memory_manager.rag_system is not None
    
    @pytest.mark.asyncio
    async def test_store_experience(self, memory_manager, sample_experience):
        """Test storing experience through manager"""
        experience = Experience(**sample_experience)
        experience_id = await memory_manager.store_experience(experience)
        
        assert experience_id is not None
    
    @pytest.mark.asyncio
    async def test_store_knowledge(self, memory_manager, sample_knowledge):
        """Test storing knowledge through manager"""
        knowledge = Knowledge(**sample_knowledge)
        knowledge_id = await memory_manager.store_knowledge(knowledge)
        
        assert knowledge_id is not None
    
    @pytest.mark.asyncio
    async def test_ask_question(self, memory_manager):
        """Test asking question with RAG"""
        # Mock RAG response
        memory_manager.rag_system.query = AsyncMock()
        memory_manager.rag_system.query.return_value = AsyncMock(
            answer="Testing is important for software quality",
            sources=[],
            confidence=0.9
        )
        
        answer = await memory_manager.ask_question("What is testing?")
        
        assert answer is not None
        assert "testing" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_search_all_memories(self, memory_manager):
        """Test searching across all memory systems"""
        # Mock search results
        memory_manager.episodic_memory.vector_store.similarity_search.return_value = []
        memory_manager.semantic_memory.vector_store.similarity_search.return_value = []
        
        results = await memory_manager.search_all_memories("test query", limit=10)
        
        assert "episodic" in results
        assert "semantic" in results
        assert "working" in results
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_manager):
        """Test getting memory statistics"""
        stats = await memory_manager.get_memory_stats()
        
        assert "episodic_memory" in stats
        assert "semantic_memory" in stats
        assert "working_memory" in stats
        assert "rag_system" in stats
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_memories(self, memory_manager):
        """Test cleaning up expired memories"""
        results = await memory_manager.cleanup_expired_memories()
        
        assert "cleaned_experiences" in results
        assert "cleaned_knowledge" in results
        assert "cleaned_contexts" in results
    
    @pytest.mark.asyncio
    async def test_export_memories(self, memory_manager):
        """Test exporting memories"""
        export_data = await memory_manager.export_memories()
        
        assert "episodic_memories" in export_data
        assert "semantic_memories" in export_data
        assert "metadata" in export_data
    
    @pytest.mark.asyncio
    async def test_health_check(self, memory_manager):
        """Test memory system health check"""
        health = await memory_manager.health_check()
        
        assert "episodic_memory" in health
        assert "semantic_memory" in health
        assert "working_memory" in health
        assert "rag_system" in health
