"""
Agent Management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional, Dict, Any
import logging

from app.agents.base import AgentConfig, Task, AgentRole, TaskPriority
from app.agents.manager import AgentManager
from app.agents.capabilities import capability_registry
from app.memory.episodic import Experience, ExperienceType
from app.memory.semantic import Knowledge, KnowledgeType

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_agent_manager(request: Request) -> AgentManager:
    """Get agent manager from app state"""
    return request.app.state.agent_manager


@router.post("/create")
async def create_agent(
    config: AgentConfig,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Create a new agent"""
    try:
        agent_id = await agent_manager.create_agent(config)
        return {"agent_id": agent_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_agents(
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """List all agents"""
    try:
        agents = agent_manager.list_agents()
        return [agent.get_status() for agent in agents]
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Get agent details"""
    try:
        agent = agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent.get_status()
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}")
async def remove_agent(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Remove an agent"""
    try:
        success = await agent_manager.remove_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {"status": "removed"}
    except Exception as e:
        logger.error(f"Failed to remove agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/tasks")
async def assign_task(
    agent_id: str,
    task: Task,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Assign a task to an agent"""
    try:
        success = await agent_manager.assign_task_to_agent(task, agent_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to assign task")
        
        return {"status": "assigned", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to assign task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/submit")
async def submit_task(
    task: Task,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Submit a task to the task queue"""
    try:
        task_id = await agent_manager.submit_task(task)
        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/status")
async def get_queue_status(
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Get task queue status"""
    try:
        return agent_manager.get_task_queue_status()
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_agent_statistics(
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Get agent statistics"""
    try:
        return agent_manager.get_agent_statistics()
    except Exception as e:
        logger.error(f"Failed to get agent statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roles")
async def list_agent_roles():
    """List available agent roles"""
    try:
        roles = capability_registry.list_roles()
        return [role.dict() for role in roles]
    except Exception as e:
        logger.error(f"Failed to list roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def list_capabilities():
    """List available capabilities"""
    try:
        capabilities = capability_registry.list_capabilities()
        return [cap.dict() for cap in capabilities]
    except Exception as e:
        logger.error(f"Failed to list capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roles/{role}/setup")
async def get_role_setup(role: AgentRole):
    """Get recommended setup for a role"""
    try:
        setup = capability_registry.get_recommended_setup(role)
        return setup
    except Exception as e:
        logger.error(f"Failed to get role setup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/default")
async def create_default_agents(
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Create default agents"""
    try:
        agent_ids = await agent_manager.create_default_agents()
        return {"agent_ids": agent_ids, "count": len(agent_ids)}
    except Exception as e:
        logger.error(f"Failed to create default agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
