"""
Agent State Management System
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import json

from app.agents.base import AgentState, BaseAgent
from app.core.database import get_redis

logger = logging.getLogger(__name__)


class StateChangeEvent(BaseModel):
    """State change event"""
    agent_id: str
    old_state: AgentState
    new_state: AgentState
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    agent_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    current_load: int = 0
    max_load: int = 1


class HealthStatus(str, Enum):
    """Agent health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AgentHealth(BaseModel):
    """Agent health information"""
    agent_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: datetime = Field(default_factory=datetime.utcnow)
    issues: List[str] = Field(default_factory=list)
    metrics: Optional[AgentMetrics] = None


class StateManager:
    """Manages agent states, metrics, and health monitoring"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.agents: Dict[str, BaseAgent] = {}
        self.state_history: Dict[str, List[StateChangeEvent]] = {}
        self.metrics: Dict[str, AgentMetrics] = {}
        self.health_status: Dict[str, AgentHealth] = {}
        self.state_listeners: List[Callable] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30  # seconds
        
    async def initialize(self) -> None:
        """Initialize state manager"""
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("State Manager initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for state management"""
        self.agents[agent.id] = agent
        
        # Initialize metrics
        self.metrics[agent.id] = AgentMetrics(
            agent_id=agent.id,
            max_load=agent.config.max_concurrent_tasks
        )
        
        # Initialize health status
        self.health_status[agent.id] = AgentHealth(
            agent_id=agent.id,
            status=HealthStatus.HEALTHY
        )
        
        # Initialize state history
        self.state_history[agent.id] = []
        
        # Set up state change listener
        agent.on_state_changed = self._on_agent_state_changed
        
        logger.info(f"Registered agent {agent.id} for state management")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from state management"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.metrics:
            del self.metrics[agent_id]
        if agent_id in self.health_status:
            del self.health_status[agent_id]
        if agent_id in self.state_history:
            del self.state_history[agent_id]
        
        logger.info(f"Unregistered agent {agent_id} from state management")
    
    async def _on_agent_state_changed(self, agent: BaseAgent, old_state: AgentState, new_state: AgentState) -> None:
        """Handle agent state change"""
        event = StateChangeEvent(
            agent_id=agent.id,
            old_state=old_state,
            new_state=new_state,
            reason=f"State transition from {old_state} to {new_state}"
        )
        
        # Add to history
        if agent.id not in self.state_history:
            self.state_history[agent.id] = []
        self.state_history[agent.id].append(event)
        
        # Keep only last 100 events
        if len(self.state_history[agent.id]) > 100:
            self.state_history[agent.id] = self.state_history[agent.id][-100:]
        
        # Persist to Redis
        if self.redis_client:
            await self._persist_state_event(event)
        
        # Notify listeners
        for listener in self.state_listeners:
            try:
                await listener(event)
            except Exception as e:
                logger.error(f"Error in state change listener: {e}")
        
        logger.info(f"Agent {agent.id} state changed: {old_state} -> {new_state}")
    
    async def _persist_state_event(self, event: StateChangeEvent) -> None:
        """Persist state change event to Redis"""
        try:
            key = f"agent_state_history:{event.agent_id}"
            value = json.dumps(event.dict(), default=str)
            await self.redis_client.lpush(key, value)
            await self.redis_client.ltrim(key, 0, 99)  # Keep last 100 events
            await self.redis_client.expire(key, 86400 * 7)  # 7 days
        except Exception as e:
            logger.error(f"Failed to persist state event: {e}")
    
    async def update_metrics(self, agent_id: str) -> None:
        """Update agent metrics"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        metrics = self.metrics[agent_id]
        
        # Update basic metrics
        metrics.total_tasks = len(agent.completed_tasks) + len(agent.current_tasks)
        metrics.completed_tasks = len([t for t in agent.completed_tasks if t.status == "completed"])
        metrics.failed_tasks = len([t for t in agent.completed_tasks if t.status == "failed"])
        metrics.current_load = len(agent.current_tasks)
        metrics.last_activity = agent.last_activity
        
        # Calculate uptime
        if hasattr(agent, 'created_at'):
            metrics.uptime_seconds = (datetime.utcnow() - agent.created_at).total_seconds()
        
        # Calculate rates
        if metrics.total_tasks > 0:
            metrics.success_rate = metrics.completed_tasks / metrics.total_tasks
            metrics.error_rate = metrics.failed_tasks / metrics.total_tasks
        
        # Calculate average task duration
        completed_tasks_with_duration = [
            t for t in agent.completed_tasks 
            if t.started_at and t.completed_at
        ]
        
        if completed_tasks_with_duration:
            total_duration = sum(
                (t.completed_at - t.started_at).total_seconds()
                for t in completed_tasks_with_duration
            )
            metrics.average_task_duration = total_duration / len(completed_tasks_with_duration)
        
        # Persist to Redis
        if self.redis_client:
            await self._persist_metrics(metrics)
    
    async def _persist_metrics(self, metrics: AgentMetrics) -> None:
        """Persist metrics to Redis"""
        try:
            key = f"agent_metrics:{metrics.agent_id}"
            value = json.dumps(metrics.dict(), default=str)
            await self.redis_client.set(key, value, ex=3600)  # 1 hour
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    async def check_agent_health(self, agent_id: str) -> AgentHealth:
        """Check agent health"""
        if agent_id not in self.agents:
            return AgentHealth(
                agent_id=agent_id,
                status=HealthStatus.UNKNOWN,
                issues=["Agent not found"]
            )
        
        agent = self.agents[agent_id]
        metrics = self.metrics[agent_id]
        health = self.health_status[agent_id]
        
        # Update health check timestamp
        health.last_check = datetime.utcnow()
        health.metrics = metrics
        health.issues.clear()
        
        # Check various health indicators
        issues = []
        status = HealthStatus.HEALTHY
        
        # Check if agent is in error state
        if agent.state == AgentState.ERROR:
            issues.append("Agent is in error state")
            status = HealthStatus.CRITICAL
        
        # Check error rate
        if metrics.error_rate > 0.5:  # More than 50% errors
            issues.append(f"High error rate: {metrics.error_rate:.2%}")
            status = max(status, HealthStatus.WARNING)
        
        # Check if agent is stuck
        if agent.last_activity:
            time_since_activity = (datetime.utcnow() - agent.last_activity).total_seconds()
            if time_since_activity > 3600:  # 1 hour
                issues.append(f"No activity for {time_since_activity/3600:.1f} hours")
                status = max(status, HealthStatus.WARNING)
        
        # Check if agent is overloaded
        if metrics.current_load >= metrics.max_load:
            issues.append("Agent is at maximum capacity")
            status = max(status, HealthStatus.WARNING)
        
        # Check average task duration
        if metrics.average_task_duration > 600:  # 10 minutes
            issues.append(f"High average task duration: {metrics.average_task_duration:.1f}s")
            status = max(status, HealthStatus.WARNING)
        
        health.status = status
        health.issues = issues
        
        return health
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Update metrics and health for all agents
                for agent_id in list(self.agents.keys()):
                    await self.update_metrics(agent_id)
                    health = await self.check_agent_health(agent_id)
                    self.health_status[agent_id] = health
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def add_state_listener(self, listener: Callable) -> None:
        """Add a state change listener"""
        self.state_listeners.append(listener)
    
    def remove_state_listener(self, listener: Callable) -> None:
        """Remove a state change listener"""
        if listener in self.state_listeners:
            self.state_listeners.remove(listener)
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get current agent state"""
        agent = self.agents.get(agent_id)
        return agent.state if agent else None
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get agent metrics"""
        return self.metrics.get(agent_id)
    
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealth]:
        """Get agent health status"""
        return self.health_status.get(agent_id)
    
    def get_state_history(self, agent_id: str, limit: Optional[int] = None) -> List[StateChangeEvent]:
        """Get agent state history"""
        history = self.state_history.get(agent_id, [])
        if limit:
            return history[-limit:]
        return history.copy()
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide overview"""
        total_agents = len(self.agents)
        
        if total_agents == 0:
            return {
                "total_agents": 0,
                "healthy_agents": 0,
                "warning_agents": 0,
                "critical_agents": 0,
                "average_load": 0.0,
                "total_tasks": 0,
                "success_rate": 0.0
            }
        
        # Count health statuses
        health_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        total_load = 0
        total_tasks = 0
        total_completed = 0
        
        for health in self.health_status.values():
            health_counts[health.status] += 1
            
            if health.metrics:
                total_load += health.metrics.current_load
                total_tasks += health.metrics.total_tasks
                total_completed += health.metrics.completed_tasks
        
        return {
            "total_agents": total_agents,
            "healthy_agents": health_counts[HealthStatus.HEALTHY],
            "warning_agents": health_counts[HealthStatus.WARNING],
            "critical_agents": health_counts[HealthStatus.CRITICAL],
            "unknown_agents": health_counts[HealthStatus.UNKNOWN],
            "average_load": total_load / total_agents,
            "total_tasks": total_tasks,
            "success_rate": total_completed / total_tasks if total_tasks > 0 else 0.0
        }
    
    async def shutdown(self) -> None:
        """Shutdown state manager"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("State Manager shutdown complete")
