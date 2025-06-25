"""
Working Memory Management - Short-term memory and context management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Deque
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
from collections import deque
import uuid
import json

from app.core.database import get_redis

logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Types of context information"""
    CONVERSATION = "conversation"
    TASK = "task"
    GOAL = "goal"
    STATE = "state"
    ENVIRONMENT = "environment"
    USER_PREFERENCE = "user_preference"
    TEMPORARY = "temporary"


class Priority(str, Enum):
    """Priority levels for working memory items"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryItem(BaseModel):
    """Individual item in working memory"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: ContextType
    key: str
    value: Any
    priority: Priority = Priority.NORMAL
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """Single turn in a conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkingMemory:
    """Working memory system for short-term context and conversation management"""
    
    def __init__(self, agent_id: str, max_size: int = 1000, default_ttl: int = 3600):
        self.agent_id = agent_id
        self.max_size = max_size
        self.default_ttl = default_ttl  # seconds
        
        # Memory storage
        self.memory_items: Dict[str, MemoryItem] = {}
        self.conversation_history: Deque[ConversationTurn] = deque(maxlen=100)
        self.current_context: Dict[str, Any] = {}
        self.active_goals: List[str] = []
        self.task_stack: List[str] = []
        
        # Redis client for persistence
        self.redis_client = None
        
        # Configuration
        self.max_conversation_length = 50  # Keep last 50 turns
        self.context_window_size = 4000  # Token limit for context
        
    async def initialize(self) -> None:
        """Initialize working memory"""
        try:
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Load persisted data
            await self._load_from_redis()
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_items())
            
            logger.info(f"Working memory initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize working memory: {e}")
            raise
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        context_type: ContextType = ContextType.TEMPORARY,
        priority: Priority = Priority.NORMAL,
        ttl: Optional[int] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store an item in working memory"""
        try:
            # Calculate expiration time
            expires_at = None
            if ttl or self.default_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl or self.default_ttl)
            
            # Create memory item
            item = MemoryItem(
                agent_id=self.agent_id,
                type=context_type,
                key=key,
                value=value,
                priority=priority,
                expires_at=expires_at,
                importance=importance,
                tags=tags or []
            )
            
            # Store in memory
            self.memory_items[item.id] = item
            
            # Update current context if it's a context item
            if context_type in [ContextType.CONVERSATION, ContextType.TASK, ContextType.GOAL, ContextType.STATE]:
                self.current_context[key] = value
            
            # Persist to Redis
            await self._persist_item(item)
            
            # Cleanup if memory is full
            if len(self.memory_items) > self.max_size:
                await self._cleanup_least_important()
            
            logger.debug(f"Stored item {key} in working memory")
            return item.id
            
        except Exception as e:
            logger.error(f"Failed to store item in working memory: {e}")
            raise
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item from working memory by key"""
        try:
            # Find item by key
            for item in self.memory_items.values():
                if item.key == key:
                    # Update access statistics
                    item.last_accessed = datetime.utcnow()
                    item.access_count += 1
                    
                    # Check if expired
                    if item.expires_at and datetime.utcnow() > item.expires_at:
                        await self.remove(item.id)
                        return None
                    
                    return item.value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve item from working memory: {e}")
            return None
    
    async def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """Get memory item by ID"""
        item = self.memory_items.get(item_id)
        if item:
            # Update access statistics
            item.last_accessed = datetime.utcnow()
            item.access_count += 1
            
            # Check if expired
            if item.expires_at and datetime.utcnow() > item.expires_at:
                await self.remove(item_id)
                return None
        
        return item
    
    async def update(self, key: str, value: Any) -> bool:
        """Update an existing item in working memory"""
        try:
            # Find item by key
            for item in self.memory_items.values():
                if item.key == key:
                    item.value = value
                    item.last_accessed = datetime.utcnow()
                    
                    # Update current context if applicable
                    if item.type in [ContextType.CONVERSATION, ContextType.TASK, ContextType.GOAL, ContextType.STATE]:
                        self.current_context[key] = value
                    
                    # Persist to Redis
                    await self._persist_item(item)
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update item in working memory: {e}")
            return False
    
    async def remove(self, item_id: str) -> bool:
        """Remove an item from working memory"""
        try:
            if item_id in self.memory_items:
                item = self.memory_items[item_id]
                
                # Remove from current context if applicable
                if item.key in self.current_context:
                    del self.current_context[item.key]
                
                # Remove from memory
                del self.memory_items[item_id]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.delete(f"working_memory:{self.agent_id}:{item_id}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove item from working memory: {e}")
            return False
    
    async def add_conversation_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a conversation turn to working memory"""
        try:
            turn = ConversationTurn(
                role=role,
                content=content,
                metadata=metadata or {}
            )
            
            self.conversation_history.append(turn)
            
            # Store recent conversation in context
            await self.store(
                key="recent_conversation",
                value=list(self.conversation_history)[-10:],  # Last 10 turns
                context_type=ContextType.CONVERSATION,
                priority=Priority.HIGH,
                importance=0.8
            )
            
            # Persist to Redis
            await self._persist_conversation()
            
            return turn.id
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn: {e}")
            raise
    
    async def get_conversation_history(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """Get conversation history"""
        history = list(self.conversation_history)
        if limit:
            return history[-limit:]
        return history
    
    async def get_conversation_context(self, max_tokens: int = 2000) -> str:
        """Get conversation context as formatted string"""
        try:
            context_parts = []
            total_tokens = 0
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            for turn in reversed(self.conversation_history):
                turn_text = f"{turn.role}: {turn.content}"
                turn_tokens = len(turn_text) // 4
                
                if total_tokens + turn_tokens > max_tokens:
                    break
                
                context_parts.insert(0, turn_text)
                total_tokens += turn_tokens
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""
    
    async def set_current_goal(self, goal: str) -> None:
        """Set current goal"""
        if goal not in self.active_goals:
            self.active_goals.append(goal)
        
        await self.store(
            key="current_goal",
            value=goal,
            context_type=ContextType.GOAL,
            priority=Priority.HIGH,
            importance=0.9
        )
    
    async def push_task(self, task: str) -> None:
        """Push task to task stack"""
        self.task_stack.append(task)
        
        await self.store(
            key="task_stack",
            value=self.task_stack.copy(),
            context_type=ContextType.TASK,
            priority=Priority.HIGH,
            importance=0.8
        )
    
    async def pop_task(self) -> Optional[str]:
        """Pop task from task stack"""
        if self.task_stack:
            task = self.task_stack.pop()
            
            await self.update("task_stack", self.task_stack.copy())
            
            return task
        return None
    
    async def get_current_context(self) -> Dict[str, Any]:
        """Get current context"""
        return self.current_context.copy()
    
    async def get_items_by_type(self, context_type: ContextType) -> List[MemoryItem]:
        """Get all items of a specific type"""
        return [item for item in self.memory_items.values() if item.type == context_type]
    
    async def get_items_by_priority(self, priority: Priority) -> List[MemoryItem]:
        """Get all items of a specific priority"""
        return [item for item in self.memory_items.values() if item.priority == priority]
    
    async def search_items(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search items by content (simple text search)"""
        try:
            results = []
            query_lower = query.lower()
            
            for item in self.memory_items.values():
                # Search in key, value (if string), and tags
                if (query_lower in item.key.lower() or
                    (isinstance(item.value, str) and query_lower in item.value.lower()) or
                    any(query_lower in tag.lower() for tag in item.tags)):
                    
                    # Update access statistics
                    item.last_accessed = datetime.utcnow()
                    item.access_count += 1
                    
                    results.append(item)
                    
                    if len(results) >= limit:
                        break
            
            # Sort by importance and access count
            results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search items: {e}")
            return []
    
    async def clear_expired(self) -> int:
        """Clear expired items"""
        try:
            now = datetime.utcnow()
            expired_items = []
            
            for item_id, item in self.memory_items.items():
                if item.expires_at and now > item.expires_at:
                    expired_items.append(item_id)
            
            for item_id in expired_items:
                await self.remove(item_id)
            
            logger.info(f"Cleared {len(expired_items)} expired items")
            return len(expired_items)
            
        except Exception as e:
            logger.error(f"Failed to clear expired items: {e}")
            return 0
    
    async def _cleanup_expired_items(self) -> None:
        """Periodic cleanup of expired items"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.clear_expired()
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _cleanup_least_important(self) -> None:
        """Remove least important items when memory is full"""
        try:
            # Sort items by importance and access patterns
            items_list = list(self.memory_items.values())
            items_list.sort(key=lambda x: (
                x.priority.value,
                x.importance,
                x.access_count,
                x.last_accessed
            ))
            
            # Remove bottom 10% of items
            items_to_remove = items_list[:len(items_list) // 10]
            
            for item in items_to_remove:
                await self.remove(item.id)
            
            logger.info(f"Cleaned up {len(items_to_remove)} least important items")
            
        except Exception as e:
            logger.error(f"Failed to cleanup least important items: {e}")
    
    async def _persist_item(self, item: MemoryItem) -> None:
        """Persist item to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"working_memory:{self.agent_id}:{item.id}"
            value = json.dumps(item.dict(), default=str)
            
            # Set TTL based on item expiration
            ttl = None
            if item.expires_at:
                ttl = int((item.expires_at - datetime.utcnow()).total_seconds())
                if ttl <= 0:
                    return  # Item already expired
            
            if ttl:
                await self.redis_client.set(key, value, ex=ttl)
            else:
                await self.redis_client.set(key, value, ex=self.default_ttl)
                
        except Exception as e:
            logger.error(f"Failed to persist item to Redis: {e}")
    
    async def _persist_conversation(self) -> None:
        """Persist conversation history to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"conversation:{self.agent_id}"
            value = json.dumps([turn.dict() for turn in self.conversation_history], default=str)
            await self.redis_client.set(key, value, ex=86400)  # 24 hours
        except Exception as e:
            logger.error(f"Failed to persist conversation to Redis: {e}")
    
    async def _load_from_redis(self) -> None:
        """Load persisted data from Redis"""
        if not self.redis_client:
            return
        
        try:
            # Load memory items
            pattern = f"working_memory:{self.agent_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    item = MemoryItem(**data)
                    self.memory_items[item.id] = item
                    
                    # Rebuild current context
                    if item.type in [ContextType.CONVERSATION, ContextType.TASK, ContextType.GOAL, ContextType.STATE]:
                        self.current_context[item.key] = item.value
            
            # Load conversation history
            conv_key = f"conversation:{self.agent_id}"
            conv_value = await self.redis_client.get(conv_key)
            if conv_value:
                conv_data = json.loads(conv_value)
                for turn_data in conv_data:
                    turn = ConversationTurn(**turn_data)
                    self.conversation_history.append(turn)
            
            logger.info(f"Loaded {len(self.memory_items)} items and {len(self.conversation_history)} conversation turns from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load from Redis: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics"""
        try:
            # Count by type
            type_counts = {}
            priority_counts = {}
            total_access_count = 0
            
            for item in self.memory_items.values():
                type_counts[item.type.value] = type_counts.get(item.type.value, 0) + 1
                priority_counts[item.priority.value] = priority_counts.get(item.priority.value, 0) + 1
                total_access_count += item.access_count
            
            return {
                "agent_id": self.agent_id,
                "total_items": len(self.memory_items),
                "max_size": self.max_size,
                "memory_usage": len(self.memory_items) / self.max_size,
                "conversation_turns": len(self.conversation_history),
                "active_goals": len(self.active_goals),
                "task_stack_size": len(self.task_stack),
                "type_distribution": type_counts,
                "priority_distribution": priority_counts,
                "total_access_count": total_access_count,
                "context_items": len(self.current_context)
            }
            
        except Exception as e:
            logger.error(f"Failed to get working memory stats: {e}")
            return {"error": str(e)}
    
    async def clear_all(self) -> None:
        """Clear all working memory (use with caution)"""
        try:
            # Clear local storage
            self.memory_items.clear()
            self.conversation_history.clear()
            self.current_context.clear()
            self.active_goals.clear()
            self.task_stack.clear()
            
            # Clear Redis
            if self.redis_client:
                pattern = f"working_memory:{self.agent_id}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                
                await self.redis_client.delete(f"conversation:{self.agent_id}")
            
            logger.info("Cleared all working memory")
            
        except Exception as e:
            logger.error(f"Failed to clear working memory: {e}")
            raise
