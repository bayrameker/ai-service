"""
Agent Discovery and Registry System
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import json

from app.agents.base import AgentRole, AgentCapability
from app.core.database import get_redis

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status in the registry"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ServiceType(str, Enum):
    """Types of services an agent can provide"""
    LLM_GENERATION = "llm_generation"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    CODING = "coding"
    REVIEW = "review"
    COORDINATION = "coordination"
    INTEGRATION = "integration"
    CUSTOM = "custom"


class AgentService(BaseModel):
    """Service provided by an agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: ServiceType
    description: str
    endpoint: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    cost: Optional[float] = None  # Cost per request
    latency_ms: Optional[int] = None  # Expected latency
    availability: float = 1.0  # 0.0 to 1.0
    rate_limit: Optional[int] = None  # Requests per minute
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRegistration(BaseModel):
    """Agent registration information"""
    agent_id: str
    name: str
    description: str
    role: AgentRole
    capabilities: List[AgentCapability] = Field(default_factory=list)
    services: List[AgentService] = Field(default_factory=list)
    endpoint: Optional[str] = None
    status: AgentStatus = AgentStatus.ONLINE
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Registration details
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    heartbeat_interval: int = 30  # seconds
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    load_factor: float = 0.0  # 0.0 to 1.0


class ServiceQuery(BaseModel):
    """Query for finding services"""
    service_type: Optional[ServiceType] = None
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    max_cost: Optional[float] = None
    max_latency: Optional[int] = None
    min_availability: float = 0.0
    exclude_agents: List[str] = Field(default_factory=list)
    sort_by: str = "availability"  # availability, cost, latency, load
    limit: int = 10


class AgentRegistry:
    """Central registry for agent discovery and management"""
    
    def __init__(self):
        self.agents: Dict[str, AgentRegistration] = {}
        self.services_by_type: Dict[ServiceType, List[str]] = {}
        self.services_by_capability: Dict[str, List[str]] = {}
        self.redis_client = None
        
        # Background tasks
        self.heartbeat_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.heartbeat_timeout = 90  # seconds
        self.cleanup_interval = 300  # seconds
    
    async def initialize(self) -> None:
        """Initialize the agent registry"""
        try:
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Load existing registrations from Redis
            await self._load_registrations()
            
            # Start background tasks
            self.heartbeat_monitor_task = asyncio.create_task(self._heartbeat_monitor())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Agent registry initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {e}")
            raise
    
    async def register_agent(self, registration: AgentRegistration) -> bool:
        """Register an agent in the registry"""
        try:
            # Store registration
            self.agents[registration.agent_id] = registration
            
            # Index services
            await self._index_agent_services(registration)
            
            # Persist to Redis
            await self._persist_registration(registration)
            
            logger.info(f"Registered agent {registration.name} ({registration.agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {registration.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry"""
        try:
            if agent_id not in self.agents:
                return False
            
            registration = self.agents[agent_id]
            
            # Remove from indexes
            await self._remove_agent_from_indexes(registration)
            
            # Remove from registry
            del self.agents[agent_id]
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"agent_registration:{agent_id}")
            
            logger.info(f"Unregistered agent {registration.name} ({agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        try:
            if agent_id not in self.agents:
                return False
            
            registration = self.agents[agent_id]
            registration.status = status
            registration.last_heartbeat = datetime.utcnow()
            
            # Persist to Redis
            await self._persist_registration(registration)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent status: {e}")
            return False
    
    async def heartbeat(self, agent_id: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Record agent heartbeat"""
        try:
            if agent_id not in self.agents:
                return False
            
            registration = self.agents[agent_id]
            registration.last_heartbeat = datetime.utcnow()
            
            # Update metrics if provided
            if metrics:
                registration.total_requests = metrics.get("total_requests", registration.total_requests)
                registration.successful_requests = metrics.get("successful_requests", registration.successful_requests)
                registration.failed_requests = metrics.get("failed_requests", registration.failed_requests)
                registration.average_response_time = metrics.get("average_response_time", registration.average_response_time)
                registration.load_factor = metrics.get("load_factor", registration.load_factor)
            
            # Update status to online if it was offline
            if registration.status == AgentStatus.OFFLINE:
                registration.status = AgentStatus.ONLINE
            
            # Persist to Redis
            await self._persist_registration(registration)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record heartbeat for agent {agent_id}: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID"""
        return self.agents.get(agent_id)
    
    def list_agents(self, status: Optional[AgentStatus] = None, role: Optional[AgentRole] = None) -> List[AgentRegistration]:
        """List agents with optional filtering"""
        agents = list(self.agents.values())
        
        if status:
            agents = [agent for agent in agents if agent.status == status]
        
        if role:
            agents = [agent for agent in agents if agent.role == role]
        
        return agents
    
    def find_agents_by_capability(self, capability_name: str) -> List[AgentRegistration]:
        """Find agents with specific capability"""
        matching_agents = []
        
        for agent in self.agents.values():
            if any(cap.name == capability_name for cap in agent.capabilities):
                matching_agents.append(agent)
        
        return matching_agents
    
    def find_services(self, query: ServiceQuery) -> List[tuple[AgentRegistration, AgentService]]:
        """Find services matching query criteria"""
        try:
            results = []
            
            for agent in self.agents.values():
                # Skip offline agents
                if agent.status == AgentStatus.OFFLINE:
                    continue
                
                # Skip excluded agents
                if agent.agent_id in query.exclude_agents:
                    continue
                
                for service in agent.services:
                    # Check service type
                    if query.service_type and service.type != query.service_type:
                        continue
                    
                    # Check capabilities
                    if query.capabilities:
                        agent_cap_names = [cap.name for cap in agent.capabilities]
                        if not any(cap in agent_cap_names for cap in query.capabilities):
                            continue
                    
                    # Check tags
                    if query.tags:
                        if not any(tag in service.tags for tag in query.tags):
                            continue
                    
                    # Check cost
                    if query.max_cost and service.cost and service.cost > query.max_cost:
                        continue
                    
                    # Check latency
                    if query.max_latency and service.latency_ms and service.latency_ms > query.max_latency:
                        continue
                    
                    # Check availability
                    if service.availability < query.min_availability:
                        continue
                    
                    results.append((agent, service))
            
            # Sort results
            if query.sort_by == "availability":
                results.sort(key=lambda x: x[1].availability, reverse=True)
            elif query.sort_by == "cost":
                results.sort(key=lambda x: x[1].cost or float('inf'))
            elif query.sort_by == "latency":
                results.sort(key=lambda x: x[1].latency_ms or float('inf'))
            elif query.sort_by == "load":
                results.sort(key=lambda x: x[0].load_factor)
            
            return results[:query.limit]
            
        except Exception as e:
            logger.error(f"Failed to find services: {e}")
            return []
    
    def get_best_agent_for_task(self, required_capabilities: List[str], preferred_role: Optional[AgentRole] = None) -> Optional[AgentRegistration]:
        """Get the best agent for a specific task"""
        try:
            candidates = []
            
            for agent in self.agents.values():
                # Skip offline agents
                if agent.status not in [AgentStatus.ONLINE, AgentStatus.IDLE]:
                    continue
                
                # Check capabilities
                agent_cap_names = [cap.name for cap in agent.capabilities]
                if not all(cap in agent_cap_names for cap in required_capabilities):
                    continue
                
                # Calculate score
                score = 0.0
                
                # Role preference
                if preferred_role and agent.role == preferred_role:
                    score += 10.0
                
                # Availability
                score += agent.load_factor * -5.0  # Lower load is better
                
                # Performance
                if agent.total_requests > 0:
                    success_rate = agent.successful_requests / agent.total_requests
                    score += success_rate * 5.0
                
                # Response time (lower is better)
                if agent.average_response_time > 0:
                    score += (1000 / agent.average_response_time) * 2.0
                
                candidates.append((agent, score))
            
            if not candidates:
                return None
            
            # Sort by score and return best
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
            
        except Exception as e:
            logger.error(f"Failed to get best agent for task: {e}")
            return None
    
    async def _index_agent_services(self, registration: AgentRegistration) -> None:
        """Index agent services for fast lookup"""
        try:
            agent_id = registration.agent_id
            
            # Index by service type
            for service in registration.services:
                if service.type not in self.services_by_type:
                    self.services_by_type[service.type] = []
                
                if agent_id not in self.services_by_type[service.type]:
                    self.services_by_type[service.type].append(agent_id)
            
            # Index by capability
            for capability in registration.capabilities:
                if capability.name not in self.services_by_capability:
                    self.services_by_capability[capability.name] = []
                
                if agent_id not in self.services_by_capability[capability.name]:
                    self.services_by_capability[capability.name].append(agent_id)
            
        except Exception as e:
            logger.error(f"Failed to index agent services: {e}")
    
    async def _remove_agent_from_indexes(self, registration: AgentRegistration) -> None:
        """Remove agent from all indexes"""
        try:
            agent_id = registration.agent_id
            
            # Remove from service type index
            for service_list in self.services_by_type.values():
                if agent_id in service_list:
                    service_list.remove(agent_id)
            
            # Remove from capability index
            for capability_list in self.services_by_capability.values():
                if agent_id in capability_list:
                    capability_list.remove(agent_id)
            
        except Exception as e:
            logger.error(f"Failed to remove agent from indexes: {e}")
    
    async def _persist_registration(self, registration: AgentRegistration) -> None:
        """Persist registration to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"agent_registration:{registration.agent_id}"
            value = registration.json()
            await self.redis_client.set(key, value, ex=86400)  # 24 hours
        except Exception as e:
            logger.error(f"Failed to persist registration: {e}")
    
    async def _load_registrations(self) -> None:
        """Load existing registrations from Redis"""
        if not self.redis_client:
            return
        
        try:
            pattern = "agent_registration:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    registration = AgentRegistration(**data)
                    self.agents[registration.agent_id] = registration
                    await self._index_agent_services(registration)
            
            logger.info(f"Loaded {len(self.agents)} agent registrations from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load registrations: {e}")
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats and mark offline agents"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                now = datetime.utcnow()
                timeout_threshold = now - timedelta(seconds=self.heartbeat_timeout)
                
                for agent in self.agents.values():
                    if (agent.status == AgentStatus.ONLINE and 
                        agent.last_heartbeat < timeout_threshold):
                        
                        agent.status = AgentStatus.OFFLINE
                        await self._persist_registration(agent)
                        logger.warning(f"Agent {agent.name} marked as offline due to missed heartbeat")
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old registrations"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                now = datetime.utcnow()
                cleanup_threshold = now - timedelta(hours=24)
                
                # Remove agents that have been offline for too long
                agents_to_remove = []
                for agent_id, agent in self.agents.items():
                    if (agent.status == AgentStatus.OFFLINE and 
                        agent.last_heartbeat < cleanup_threshold):
                        agents_to_remove.append(agent_id)
                
                for agent_id in agents_to_remove:
                    await self.unregister_agent(agent_id)
                    logger.info(f"Cleaned up old registration for agent {agent_id}")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_agents = len(self.agents)
        online_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ONLINE])
        
        # Count by role
        role_counts = {}
        for role in AgentRole:
            role_counts[role.value] = len([a for a in self.agents.values() if a.role == role])
        
        # Count by status
        status_counts = {}
        for status in AgentStatus:
            status_counts[status.value] = len([a for a in self.agents.values() if a.status == status])
        
        return {
            "total_agents": total_agents,
            "online_agents": online_agents,
            "role_distribution": role_counts,
            "status_distribution": status_counts,
            "total_services": sum(len(agent.services) for agent in self.agents.values()),
            "service_types": len(self.services_by_type),
            "indexed_capabilities": len(self.services_by_capability)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the registry"""
        try:
            # Cancel background tasks
            if self.heartbeat_monitor_task:
                self.heartbeat_monitor_task.cancel()
                try:
                    await self.heartbeat_monitor_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Agent registry shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during registry shutdown: {e}")


# Global agent registry instance
agent_registry = AgentRegistry()
