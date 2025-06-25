"""
Memory Manager - Unified interface for all memory systems
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.memory.vector_store import BaseVectorStore, create_vector_store
from app.memory.episodic import EpisodicMemory, Experience
from app.memory.semantic import SemanticMemory, Knowledge
from app.memory.working import WorkingMemory
from app.memory.rag import RAGSystem, RAGQuery
from app.memory.learning import SelfLearningSystem
from app.llm_providers.manager import LLMProviderManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified memory management system"""
    
    def __init__(self, agent_id: str, llm_manager: LLMProviderManager):
        self.agent_id = agent_id
        self.llm_manager = llm_manager
        
        # Memory systems
        self.vector_store: Optional[BaseVectorStore] = None
        self.episodic_memory: Optional[EpisodicMemory] = None
        self.semantic_memory: Optional[SemanticMemory] = None
        self.working_memory: Optional[WorkingMemory] = None
        self.rag_system: Optional[RAGSystem] = None
        self.learning_system: Optional[SelfLearningSystem] = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all memory systems"""
        try:
            logger.info(f"Initializing memory manager for agent {self.agent_id}")
            
            # Initialize vector store
            self.vector_store = create_vector_store()
            await self.vector_store.initialize()
            
            # Initialize working memory
            self.working_memory = WorkingMemory(self.agent_id)
            await self.working_memory.initialize()
            
            # Initialize episodic memory
            self.episodic_memory = EpisodicMemory(
                self.agent_id, 
                self.vector_store, 
                self.llm_manager
            )
            await self.episodic_memory.initialize()
            
            # Initialize semantic memory
            self.semantic_memory = SemanticMemory(
                self.agent_id,
                self.vector_store,
                self.llm_manager
            )
            await self.semantic_memory.initialize()
            
            # Initialize RAG system
            self.rag_system = RAGSystem(
                self.agent_id,
                self.llm_manager,
                self.episodic_memory,
                self.semantic_memory,
                self.working_memory
            )
            
            # Initialize learning system
            self.learning_system = SelfLearningSystem(
                self.agent_id,
                self.episodic_memory,
                self.semantic_memory,
                self.working_memory,
                self.llm_manager
            )
            await self.learning_system.initialize()
            
            self._initialized = True
            logger.info(f"Memory manager initialized successfully for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    # Episodic Memory Interface
    async def store_experience(self, experience: Experience) -> str:
        """Store an experience in episodic memory"""
        if not self.episodic_memory:
            raise RuntimeError("Episodic memory not initialized")
        
        experience_id = await self.episodic_memory.store_experience(experience)
        
        # Learn from the experience
        if self.learning_system:
            await self.learning_system.learn_from_experience(experience)
        
        return experience_id
    
    async def get_experiences(self, query: str, limit: int = 10) -> List[Experience]:
        """Retrieve experiences from episodic memory"""
        if not self.episodic_memory:
            return []
        
        return await self.episodic_memory.retrieve_experiences(query, limit)
    
    # Semantic Memory Interface
    async def store_knowledge(self, knowledge: Knowledge) -> str:
        """Store knowledge in semantic memory"""
        if not self.semantic_memory:
            raise RuntimeError("Semantic memory not initialized")
        
        return await self.semantic_memory.store_knowledge(knowledge)
    
    async def get_knowledge(self, query: str, limit: int = 10) -> List[Knowledge]:
        """Retrieve knowledge from semantic memory"""
        if not self.semantic_memory:
            return []
        
        return await self.semantic_memory.retrieve_knowledge(query, limit)
    
    # Working Memory Interface
    async def store_context(self, key: str, value: Any, **kwargs) -> str:
        """Store context in working memory"""
        if not self.working_memory:
            raise RuntimeError("Working memory not initialized")
        
        return await self.working_memory.store(key, value, **kwargs)
    
    async def get_context(self, key: str) -> Any:
        """Get context from working memory"""
        if not self.working_memory:
            return None
        
        return await self.working_memory.retrieve(key)
    
    async def add_conversation_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add conversation turn to working memory"""
        if not self.working_memory:
            raise RuntimeError("Working memory not initialized")
        
        return await self.working_memory.add_conversation_turn(role, content, metadata)
    
    # RAG Interface
    async def ask_question(self, question: str, context: Optional[str] = None) -> str:
        """Ask a question using RAG system"""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        return await self.rag_system.ask_with_memory(question, context)
    
    async def query_with_rag(self, rag_query: RAGQuery):
        """Execute RAG query"""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        return await self.rag_system.query(rag_query)
    
    # Learning Interface
    async def get_learning_insights(self, limit: Optional[int] = None):
        """Get learning insights"""
        if not self.learning_system:
            return []
        
        return await self.learning_system.get_learning_insights(limit)
    
    # Unified Search Interface
    async def search_all_memories(self, query: str, limit: int = 10) -> Dict[str, List[Any]]:
        """Search across all memory systems"""
        results = {
            "experiences": [],
            "knowledge": [],
            "working_memory": [],
            "insights": []
        }
        
        try:
            # Search episodic memory
            if self.episodic_memory:
                results["experiences"] = await self.episodic_memory.retrieve_experiences(query, limit)
            
            # Search semantic memory
            if self.semantic_memory:
                results["knowledge"] = await self.semantic_memory.retrieve_knowledge(query, limit)
            
            # Search working memory
            if self.working_memory:
                working_items = await self.working_memory.search_items(query, limit)
                results["working_memory"] = working_items
            
            # Get relevant learning insights
            if self.learning_system:
                all_insights = await self.learning_system.get_learning_insights()
                # Simple text search in insights
                query_lower = query.lower()
                relevant_insights = [
                    insight for insight in all_insights
                    if query_lower in insight.title.lower() or query_lower in insight.description.lower()
                ]
                results["insights"] = relevant_insights[:limit]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
        
        return results
    
    # Memory Statistics
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            "agent_id": self.agent_id,
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if self.episodic_memory:
                stats["episodic"] = await self.episodic_memory.get_stats()
            
            if self.semantic_memory:
                stats["semantic"] = await self.semantic_memory.get_stats()
            
            if self.working_memory:
                stats["working"] = await self.working_memory.get_stats()
            
            if self.rag_system:
                stats["rag"] = await self.rag_system.get_stats()
            
            if self.learning_system:
                stats["learning"] = await self.learning_system.get_stats()
            
            if self.vector_store:
                stats["vector_store"] = await self.vector_store.get_stats()
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    # Memory Maintenance
    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """Clean up expired memories across all systems"""
        cleanup_results = {}
        
        try:
            # Cleanup working memory
            if self.working_memory:
                expired_count = await self.working_memory.clear_expired()
                cleanup_results["working_memory_expired"] = expired_count
            
            # Note: Episodic and semantic memories have their own cleanup mechanisms
            # that run automatically based on importance and retention policies
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            cleanup_results["error"] = str(e)
        
        return cleanup_results
    
    # Memory Export/Import (for backup/restore)
    async def export_memories(self) -> Dict[str, Any]:
        """Export all memories for backup"""
        export_data = {
            "agent_id": self.agent_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "memories": {}
        }
        
        try:
            # Export working memory context
            if self.working_memory:
                export_data["memories"]["working_context"] = await self.working_memory.get_current_context()
                export_data["memories"]["conversation_history"] = await self.working_memory.get_conversation_history()
            
            # Export learning insights
            if self.learning_system:
                insights = await self.learning_system.get_learning_insights()
                export_data["memories"]["learning_insights"] = [insight.dict() for insight in insights]
            
            # Note: Episodic and semantic memories are stored in vector store
            # and would require more complex export mechanisms
            
        except Exception as e:
            logger.error(f"Error exporting memories: {e}")
            export_data["error"] = str(e)
        
        return export_data
    
    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all memory systems"""
        health = {
            "overall": "healthy",
            "systems": {}
        }
        
        try:
            # Check each system
            systems_to_check = [
                ("vector_store", self.vector_store),
                ("episodic_memory", self.episodic_memory),
                ("semantic_memory", self.semantic_memory),
                ("working_memory", self.working_memory),
                ("rag_system", self.rag_system),
                ("learning_system", self.learning_system)
            ]
            
            for system_name, system in systems_to_check:
                if system:
                    try:
                        # Basic health check - try to get stats
                        if hasattr(system, 'get_stats'):
                            await system.get_stats()
                        health["systems"][system_name] = "healthy"
                    except Exception as e:
                        health["systems"][system_name] = f"unhealthy: {str(e)}"
                        health["overall"] = "degraded"
                else:
                    health["systems"][system_name] = "not_initialized"
                    health["overall"] = "degraded"
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health["overall"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown all memory systems"""
        try:
            logger.info(f"Shutting down memory manager for agent {self.agent_id}")
            
            # Shutdown learning system
            if self.learning_system:
                await self.learning_system.shutdown()
            
            # Clear working memory if needed
            # (other systems persist their data automatically)
            
            logger.info("Memory manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during memory manager shutdown: {e}")
    
    def __str__(self) -> str:
        return f"MemoryManager(agent_id={self.agent_id}, initialized={self._initialized})"
