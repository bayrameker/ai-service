"""
Base Agent classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"
    INITIALIZING = "initializing"


class AgentRole(str, Enum):
    """Agent roles"""
    GENERAL = "general"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CODER = "coder"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    """Task model for agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_agent_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentCapability(BaseModel):
    """Agent capability definition"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class AgentConfig(BaseModel):
    """Agent configuration"""
    name: str
    role: AgentRole
    description: str
    llm_model: str = "gpt-3.5-turbo"
    llm_provider: Optional[str] = None
    max_concurrent_tasks: int = 1
    timeout_seconds: int = 300
    capabilities: List[AgentCapability] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    memory_enabled: bool = True
    learning_enabled: bool = True
    collaboration_enabled: bool = True


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.id = str(uuid.uuid4())
        self.config = config
        self.state = AgentState.INITIALIZING
        self.current_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.error_count = 0
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        
        # Event handlers
        self.on_task_started: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_state_changed: Optional[Callable] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the agent"""
        pass
    
    async def assign_task(self, task: Task) -> bool:
        """Assign a task to the agent"""
        if len(self.current_tasks) >= self.config.max_concurrent_tasks:
            return False
        
        if self.state not in [AgentState.IDLE, AgentState.BUSY]:
            return False
        
        task.assigned_agent_id = self.id
        task.started_at = datetime.utcnow()
        task.status = "running"
        
        self.current_tasks[task.id] = task
        self._update_state()
        
        # Trigger event
        if self.on_task_started:
            await self.on_task_started(self, task)
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task_wrapper(task))
        
        return True
    
    async def _execute_task_wrapper(self, task: Task) -> None:
        """Wrapper for task execution with error handling"""
        try:
            logger.info(f"Agent {self.id} starting task {task.id}")
            
            result = await self.execute_task(task)
            
            # Mark task as completed
            task.completed_at = datetime.utcnow()
            task.status = "completed"
            task.result = result
            
            # Move to completed tasks
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
            self.completed_tasks.append(task)
            
            # Trigger event
            if self.on_task_completed:
                await self.on_task_completed(self, task)
            
            logger.info(f"Agent {self.id} completed task {task.id}")
            
        except Exception as e:
            logger.error(f"Agent {self.id} failed task {task.id}: {e}")
            
            # Mark task as failed
            task.completed_at = datetime.utcnow()
            task.status = "failed"
            task.error = str(e)
            
            # Move to completed tasks
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
            self.completed_tasks.append(task)
            
            self.error_count += 1
            
            # Trigger event
            if self.on_task_failed:
                await self.on_task_failed(self, task)
        
        finally:
            self.last_activity = datetime.utcnow()
            self._update_state()
    
    def _update_state(self) -> None:
        """Update agent state based on current tasks"""
        old_state = self.state
        
        if len(self.current_tasks) == 0:
            self.state = AgentState.IDLE
        else:
            self.state = AgentState.BUSY
        
        # Trigger state change event
        if old_state != self.state and self.on_state_changed:
            asyncio.create_task(self.on_state_changed(self, old_state, self.state))
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "id": self.id,
            "name": self.config.name,
            "role": self.config.role,
            "state": self.state,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": len(self.completed_tasks),
            "error_count": self.error_count,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
            "capabilities": [cap.name for cap in self.config.capabilities],
            "metadata": self.metadata
        }
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle a specific task"""
        # Basic checks
        if self.state not in [AgentState.IDLE, AgentState.BUSY]:
            return False
        
        if len(self.current_tasks) >= self.config.max_concurrent_tasks:
            return False
        
        # Role-based checks can be implemented here
        # For now, return True for basic implementation
        return True
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent"""
        self.config.capabilities.append(capability)
    
    def remove_capability(self, capability_name: str) -> bool:
        """Remove a capability from the agent"""
        for i, cap in enumerate(self.config.capabilities):
            if cap.name == capability_name:
                del self.config.capabilities[i]
                return True
        return False
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name for cap in self.config.capabilities)
    
    def __str__(self) -> str:
        return f"Agent(id={self.id[:8]}, name={self.config.name}, role={self.config.role}, state={self.state})"
    
    def __repr__(self) -> str:
        return self.__str__()
