"""
Episodic Memory System - Stores agent experiences and interactions
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import json

from app.memory.vector_store import BaseVectorStore, VectorDocument, create_vector_store
from app.llm_providers.manager import LLMProviderManager
from app.core.database import get_redis

logger = logging.getLogger(__name__)


class ExperienceType(str, Enum):
    """Types of experiences"""
    TASK_EXECUTION = "task_execution"
    INTERACTION = "interaction"
    LEARNING = "learning"
    ERROR = "error"
    SUCCESS = "success"
    COLLABORATION = "collaboration"
    DECISION = "decision"


class Experience(BaseModel):
    """Individual experience/episode"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: ExperienceType
    title: str
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    outcome: Optional[str] = None
    success: Optional[bool] = None
    lessons_learned: List[str] = Field(default_factory=list)
    emotions: Dict[str, float] = Field(default_factory=dict)  # Simulated emotional state
    importance: float = 0.5  # 0.0 to 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = None
    related_experiences: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodicMemory:
    """Episodic memory system for storing and retrieving agent experiences"""
    
    def __init__(self, agent_id: str, vector_store: BaseVectorStore, llm_manager: LLMProviderManager):
        self.agent_id = agent_id
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self.experiences: Dict[str, Experience] = {}
        self.redis_client = None
        
        # Memory configuration
        self.max_experiences = 10000
        self.importance_threshold = 0.1
        self.retention_days = 365
        
    async def initialize(self) -> None:
        """Initialize episodic memory"""
        try:
            # Initialize vector store
            if not self.vector_store.initialized:
                await self.vector_store.initialize()
            
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Load recent experiences from Redis
            await self._load_recent_experiences()
            
            logger.info(f"Episodic memory initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize episodic memory: {e}")
            raise
    
    async def store_experience(self, experience: Experience) -> str:
        """Store a new experience"""
        try:
            # Set agent ID
            experience.agent_id = self.agent_id
            
            # Generate embedding for the experience
            experience_text = self._experience_to_text(experience)
            embedding = await self._get_embedding(experience_text)
            
            # Create vector document
            vector_doc = VectorDocument(
                id=experience.id,
                content=experience_text,
                embedding=embedding,
                metadata={
                    "agent_id": self.agent_id,
                    "type": experience.type.value,
                    "timestamp": experience.timestamp.isoformat(),
                    "importance": experience.importance,
                    "success": experience.success,
                    "tags": experience.tags,
                    **experience.metadata
                }
            )
            
            # Store in vector database
            await self.vector_store.add_documents([vector_doc])
            
            # Store in local cache
            self.experiences[experience.id] = experience
            
            # Store in Redis for quick access
            await self._store_experience_in_redis(experience)
            
            # Update related experiences
            await self._update_related_experiences(experience)
            
            # Cleanup old experiences if needed
            await self._cleanup_old_experiences()
            
            logger.info(f"Stored experience {experience.id} for agent {self.agent_id}")
            return experience.id
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            raise
    
    async def retrieve_experiences(
        self, 
        query: str, 
        limit: int = 10,
        experience_type: Optional[ExperienceType] = None,
        min_importance: float = 0.0,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Experience]:
        """Retrieve relevant experiences based on query"""
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Prepare metadata filter
            metadata_filter = {"agent_id": self.agent_id}
            if experience_type:
                metadata_filter["type"] = experience_type.value
            
            # Search vector store
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=limit * 2,  # Get more results for filtering
                filter_metadata=metadata_filter
            )
            
            # Filter and convert results
            experiences = []
            for result in search_results:
                exp_id = result.document.id
                
                # Get full experience from cache or reconstruct
                experience = self.experiences.get(exp_id)
                if not experience:
                    experience = await self._reconstruct_experience_from_vector_doc(result.document)
                
                if not experience:
                    continue
                
                # Apply filters
                if experience.importance < min_importance:
                    continue
                
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= experience.timestamp <= end_time):
                        continue
                
                experiences.append(experience)
                
                if len(experiences) >= limit:
                    break
            
            return experiences
            
        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    async def get_experience(self, experience_id: str) -> Optional[Experience]:
        """Get specific experience by ID"""
        # Try local cache first
        if experience_id in self.experiences:
            return self.experiences[experience_id]
        
        # Try Redis
        experience = await self._get_experience_from_redis(experience_id)
        if experience:
            self.experiences[experience_id] = experience
            return experience
        
        # Try vector store
        vector_doc = await self.vector_store.get_document(experience_id)
        if vector_doc:
            return await self._reconstruct_experience_from_vector_doc(vector_doc)
        
        return None
    
    async def update_experience(self, experience_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing experience"""
        try:
            experience = await self.get_experience(experience_id)
            if not experience:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(experience, key):
                    setattr(experience, key, value)
            
            # Re-store the experience
            await self.store_experience(experience)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update experience {experience_id}: {e}")
            return False
    
    async def delete_experience(self, experience_id: str) -> bool:
        """Delete an experience"""
        try:
            # Remove from vector store
            await self.vector_store.delete_document(experience_id)
            
            # Remove from local cache
            if experience_id in self.experiences:
                del self.experiences[experience_id]
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"experience:{self.agent_id}:{experience_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete experience {experience_id}: {e}")
            return False
    
    async def get_similar_experiences(self, experience: Experience, limit: int = 5) -> List[Experience]:
        """Find experiences similar to the given one"""
        experience_text = self._experience_to_text(experience)
        return await self.retrieve_experiences(experience_text, limit=limit)
    
    async def get_experiences_by_type(self, experience_type: ExperienceType, limit: int = 10) -> List[Experience]:
        """Get experiences by type"""
        return await self.retrieve_experiences(
            query=f"experiences of type {experience_type.value}",
            limit=limit,
            experience_type=experience_type
        )
    
    async def get_recent_experiences(self, hours: int = 24, limit: int = 10) -> List[Experience]:
        """Get recent experiences"""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        end_time = datetime.utcnow()
        
        return await self.retrieve_experiences(
            query="recent experiences",
            limit=limit,
            time_range=(start_time, end_time)
        )
    
    async def get_successful_experiences(self, limit: int = 10) -> List[Experience]:
        """Get successful experiences for learning"""
        experiences = await self.retrieve_experiences(
            query="successful experiences and achievements",
            limit=limit * 2
        )
        
        # Filter for successful experiences
        successful = [exp for exp in experiences if exp.success is True]
        return successful[:limit]
    
    async def get_learning_insights(self, topic: str) -> List[str]:
        """Extract learning insights from experiences related to a topic"""
        try:
            experiences = await self.retrieve_experiences(topic, limit=20)
            
            insights = []
            for exp in experiences:
                insights.extend(exp.lessons_learned)
            
            # Remove duplicates and return
            return list(set(insights))
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return []
    
    def _experience_to_text(self, experience: Experience) -> str:
        """Convert experience to text for embedding"""
        text_parts = [
            f"Experience: {experience.title}",
            f"Type: {experience.type.value}",
            f"Description: {experience.description}"
        ]
        
        if experience.outcome:
            text_parts.append(f"Outcome: {experience.outcome}")
        
        if experience.lessons_learned:
            text_parts.append(f"Lessons: {', '.join(experience.lessons_learned)}")
        
        if experience.context:
            context_str = ", ".join(f"{k}: {v}" for k, v in experience.context.items())
            text_parts.append(f"Context: {context_str}")
        
        if experience.tags:
            text_parts.append(f"Tags: {', '.join(experience.tags)}")
        
        return "\n".join(text_parts)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            return await self.llm_manager.get_embedding(text)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    async def _store_experience_in_redis(self, experience: Experience) -> None:
        """Store experience in Redis for quick access"""
        if not self.redis_client:
            return
        
        try:
            key = f"experience:{self.agent_id}:{experience.id}"
            value = json.dumps(experience.dict(), default=str)
            await self.redis_client.set(key, value, ex=86400 * 7)  # 7 days
        except Exception as e:
            logger.error(f"Failed to store experience in Redis: {e}")
    
    async def _get_experience_from_redis(self, experience_id: str) -> Optional[Experience]:
        """Get experience from Redis"""
        if not self.redis_client:
            return None
        
        try:
            key = f"experience:{self.agent_id}:{experience_id}"
            value = await self.redis_client.get(key)
            if value:
                data = json.loads(value)
                return Experience(**data)
        except Exception as e:
            logger.error(f"Failed to get experience from Redis: {e}")
        
        return None
    
    async def _load_recent_experiences(self) -> None:
        """Load recent experiences from Redis"""
        if not self.redis_client:
            return
        
        try:
            pattern = f"experience:{self.agent_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys[:100]:  # Limit to 100 recent experiences
                value = await self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    experience = Experience(**data)
                    self.experiences[experience.id] = experience
            
            logger.info(f"Loaded {len(self.experiences)} recent experiences from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load recent experiences: {e}")
    
    async def _reconstruct_experience_from_vector_doc(self, vector_doc: VectorDocument) -> Optional[Experience]:
        """Reconstruct experience from vector document metadata"""
        try:
            # This is a simplified reconstruction
            # In a real system, you might store the full experience data
            metadata = vector_doc.metadata
            
            experience = Experience(
                id=vector_doc.id,
                agent_id=metadata.get("agent_id", self.agent_id),
                type=ExperienceType(metadata.get("type", "interaction")),
                title=f"Experience {vector_doc.id[:8]}",
                description=vector_doc.content,
                importance=metadata.get("importance", 0.5),
                success=metadata.get("success"),
                tags=metadata.get("tags", []),
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
                metadata=metadata
            )
            
            return experience
            
        except Exception as e:
            logger.error(f"Failed to reconstruct experience: {e}")
            return None
    
    async def _update_related_experiences(self, experience: Experience) -> None:
        """Update related experiences based on similarity"""
        try:
            similar_experiences = await self.get_similar_experiences(experience, limit=5)
            
            # Update bidirectional relationships
            for similar_exp in similar_experiences:
                if similar_exp.id != experience.id:
                    # Add to current experience
                    if similar_exp.id not in experience.related_experiences:
                        experience.related_experiences.append(similar_exp.id)
                    
                    # Add to similar experience
                    if experience.id not in similar_exp.related_experiences:
                        similar_exp.related_experiences.append(experience.id)
                        await self._store_experience_in_redis(similar_exp)
            
        except Exception as e:
            logger.error(f"Failed to update related experiences: {e}")
    
    async def _cleanup_old_experiences(self) -> None:
        """Clean up old or low-importance experiences"""
        try:
            if len(self.experiences) <= self.max_experiences:
                return
            
            # Sort by importance and age
            experiences_list = list(self.experiences.values())
            experiences_list.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
            
            # Keep top experiences
            to_keep = experiences_list[:self.max_experiences]
            to_remove = experiences_list[self.max_experiences:]
            
            # Remove old experiences
            for exp in to_remove:
                if exp.importance < self.importance_threshold:
                    await self.delete_experience(exp.id)
            
            logger.info(f"Cleaned up {len(to_remove)} old experiences")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old experiences: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        try:
            total_experiences = len(self.experiences)
            
            # Count by type
            type_counts = {}
            importance_sum = 0
            successful_count = 0
            
            for exp in self.experiences.values():
                type_counts[exp.type.value] = type_counts.get(exp.type.value, 0) + 1
                importance_sum += exp.importance
                if exp.success is True:
                    successful_count += 1
            
            avg_importance = importance_sum / total_experiences if total_experiences > 0 else 0
            success_rate = successful_count / total_experiences if total_experiences > 0 else 0
            
            return {
                "agent_id": self.agent_id,
                "total_experiences": total_experiences,
                "type_distribution": type_counts,
                "average_importance": avg_importance,
                "success_rate": success_rate,
                "vector_store_stats": await self.vector_store.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get episodic memory stats: {e}")
            return {"error": str(e)}
