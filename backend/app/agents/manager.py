"""
Agent Manager - Handles agent lifecycle and management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import uuid

from app.agents.base import BaseAgent, AgentConfig, Task, AgentState, AgentRole, TaskPriority
from app.agents.llm_agent import LLMAgent
from app.llm_providers.manager import LLMProviderManager

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages agent lifecycle, task distribution, and coordination"""
    
    def __init__(self, llm_manager: Optional[LLMProviderManager] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.llm_manager = llm_manager
        self.running = False
        self.task_processor_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self.on_agent_created: Optional[Callable] = None
        self.on_agent_removed: Optional[Callable] = None
        self.on_task_assigned: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None
    
    async def initialize(self, llm_manager: LLMProviderManager) -> None:
        """Initialize the agent manager"""
        self.llm_manager = llm_manager
        self.running = True
        
        # Start task processor
        self.task_processor_task = asyncio.create_task(self._process_tasks())
        
        logger.info("Agent Manager initialized")
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new agent"""
        try:
            if not self.llm_manager:
                raise ValueError("LLM Manager not initialized")
            
            # Create agent based on type (for now, only LLM agents)
            agent = LLMAgent(config, self.llm_manager)
            
            # Set up event handlers
            agent.on_task_started = self._on_agent_task_started
            agent.on_task_completed = self._on_agent_task_completed
            agent.on_task_failed = self._on_agent_task_failed
            agent.on_state_changed = self._on_agent_state_changed
            
            # Initialize agent
            await agent.initialize()
            
            # Add to agents registry
            self.agents[agent.id] = agent
            
            # Trigger event
            if self.on_agent_created:
                await self.on_agent_created(agent)
            
            logger.info(f"Created agent {agent.config.name} with ID {agent.id}")
            return agent.id
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent"""
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Shutdown agent
            await agent.shutdown()
            
            # Remove from registry
            del self.agents[agent_id]
            
            # Trigger event
            if self.on_agent_removed:
                await self.on_agent_removed(agent)
            
            logger.info(f"Removed agent {agent.config.name} with ID {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[BaseAgent]:
        """List all agents"""
        return list(self.agents.values())
    
    def get_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """Get agents by role"""
        return [agent for agent in self.agents.values() if agent.config.role == role]
    
    def get_agents_by_state(self, state: AgentState) -> List[BaseAgent]:
        """Get agents by state"""
        return [agent for agent in self.agents.values() if agent.state == state]
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the task queue"""
        task.id = task.id or str(uuid.uuid4())
        self.task_queue.append(task)
        
        logger.info(f"Submitted task {task.title} with ID {task.id}")
        return task.id
    
    async def assign_task_to_agent(self, task: Task, agent_id: str) -> bool:
        """Assign a task directly to a specific agent"""
        agent = self.get_agent(agent_id)
        if not agent:
            return False
        
        success = await agent.assign_task(task)
        if success and self.on_task_assigned:
            await self.on_task_assigned(agent, task)
        
        return success
    
    async def _process_tasks(self) -> None:
        """Process tasks from the queue"""
        while self.running:
            try:
                if not self.task_queue:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task
                task = self.task_queue.pop(0)
                
                # Find suitable agent
                suitable_agent = self._find_suitable_agent(task)
                
                if suitable_agent:
                    success = await suitable_agent.assign_task(task)
                    if success:
                        if self.on_task_assigned:
                            await self.on_task_assigned(suitable_agent, task)
                        logger.info(f"Assigned task {task.id} to agent {suitable_agent.id}")
                    else:
                        # Put task back in queue
                        self.task_queue.insert(0, task)
                        await asyncio.sleep(1)
                else:
                    # No suitable agent found, put task back in queue
                    self.task_queue.insert(0, task)
                    await asyncio.sleep(5)  # Wait longer before retrying
                
            except Exception as e:
                logger.error(f"Error processing tasks: {e}")
                await asyncio.sleep(1)
    
    def _find_suitable_agent(self, task: Task) -> Optional[BaseAgent]:
        """Find the most suitable agent for a task"""
        suitable_agents = []
        
        # Find agents that can handle the task
        for agent in self.agents.values():
            if agent.can_handle_task(task):
                suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Sort by priority (idle agents first, then by load)
        suitable_agents.sort(key=lambda a: (
            a.state != AgentState.IDLE,  # Idle agents first
            len(a.current_tasks),        # Then by current load
            a.error_count               # Then by error count
        ))
        
        return suitable_agents[0]
    
    async def _on_agent_task_started(self, agent: BaseAgent, task: Task) -> None:
        """Handle agent task started event"""
        logger.info(f"Agent {agent.id} started task {task.id}")
    
    async def _on_agent_task_completed(self, agent: BaseAgent, task: Task) -> None:
        """Handle agent task completed event"""
        self.completed_tasks.append(task)
        
        if self.on_task_completed:
            await self.on_task_completed(agent, task)
        
        logger.info(f"Agent {agent.id} completed task {task.id}")
    
    async def _on_agent_task_failed(self, agent: BaseAgent, task: Task) -> None:
        """Handle agent task failed event"""
        self.completed_tasks.append(task)
        logger.error(f"Agent {agent.id} failed task {task.id}: {task.error}")
    
    async def _on_agent_state_changed(self, agent: BaseAgent, old_state: AgentState, new_state: AgentState) -> None:
        """Handle agent state change event"""
        logger.info(f"Agent {agent.id} state changed from {old_state} to {new_state}")
    
    def get_task_queue_status(self) -> Dict[str, Any]:
        """Get task queue status"""
        return {
            "pending_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "total_agents": len(self.agents),
            "idle_agents": len(self.get_agents_by_state(AgentState.IDLE)),
            "busy_agents": len(self.get_agents_by_state(AgentState.BUSY)),
            "error_agents": len(self.get_agents_by_state(AgentState.ERROR))
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "total_agents": len(self.agents),
            "agents_by_role": {},
            "agents_by_state": {},
            "total_tasks_completed": len(self.completed_tasks),
            "average_tasks_per_agent": 0,
            "agents": []
        }
        
        # Count by role
        for role in AgentRole:
            stats["agents_by_role"][role.value] = len(self.get_agents_by_role(role))
        
        # Count by state
        for state in AgentState:
            stats["agents_by_state"][state.value] = len(self.get_agents_by_state(state))
        
        # Calculate average tasks per agent
        if self.agents:
            total_completed = sum(len(agent.completed_tasks) for agent in self.agents.values())
            stats["average_tasks_per_agent"] = total_completed / len(self.agents)
        
        # Individual agent stats
        for agent in self.agents.values():
            agent_stats = agent.get_status()
            stats["agents"].append(agent_stats)
        
        return stats
    
    async def create_default_agents(self) -> List[str]:
        """Create a set of default agents"""
        default_configs = [
            AgentConfig(
                name="General Assistant",
                role=AgentRole.GENERAL,
                description="A general-purpose AI assistant capable of handling various tasks",
                llm_model="gpt-3.5-turbo"
            ),
            AgentConfig(
                name="Research Specialist",
                role=AgentRole.RESEARCHER,
                description="Specialized in research, data gathering, and analysis",
                llm_model="gpt-4"
            ),
            AgentConfig(
                name="Code Assistant",
                role=AgentRole.CODER,
                description="Specialized in programming, code review, and technical tasks",
                llm_model="gpt-4"
            )
        ]
        
        agent_ids = []
        for config in default_configs:
            try:
                agent_id = await self.create_agent(config)
                agent_ids.append(agent_id)
            except Exception as e:
                logger.error(f"Failed to create default agent {config.name}: {e}")
        
        return agent_ids
    
    async def shutdown(self) -> None:
        """Shutdown the agent manager"""
        try:
            logger.info("Shutting down Agent Manager...")
            
            self.running = False
            
            # Cancel task processor
            if self.task_processor_task:
                self.task_processor_task.cancel()
                try:
                    await self.task_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all agents
            for agent in list(self.agents.values()):
                await agent.shutdown()
            
            self.agents.clear()
            
            logger.info("Agent Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Agent Manager shutdown: {e}")
