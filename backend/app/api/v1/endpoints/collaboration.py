"""
Collaboration API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional, Dict, Any
import logging

from app.collaboration.message_passing import Message, MessageType, MessagePriority, message_broker
from app.collaboration.discovery import AgentRegistration, ServiceQuery, agent_registry

logger = logging.getLogger(__name__)
router = APIRouter()


# Message Passing endpoints
@router.post("/messages/send")
async def send_message(message: Message):
    """Send message through message broker"""
    try:
        success = await message_broker.send_message(message)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to send message")
        return {"status": "sent", "message_id": message.id}
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/stats")
async def get_message_stats():
    """Get message broker statistics"""
    try:
        return message_broker.get_stats()
    except Exception as e:
        logger.error(f"Failed to get message stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/groups/create")
async def create_group(group_name: str, agent_ids: List[str]):
    """Create message group"""
    try:
        message_broker.create_group(group_name, agent_ids)
        return {"status": "created", "group_name": group_name}
    except Exception as e:
        logger.error(f"Failed to create group: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/groups/{group_name}/add")
async def add_to_group(group_name: str, agent_id: str):
    """Add agent to group"""
    try:
        message_broker.add_to_group(group_name, agent_id)
        return {"status": "added"}
    except Exception as e:
        logger.error(f"Failed to add agent to group: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent Registry endpoints
@router.post("/registry/register")
async def register_agent(registration: AgentRegistration):
    """Register agent in registry"""
    try:
        success = await agent_registry.register_agent(registration)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register agent")
        return {"status": "registered", "agent_id": registration.agent_id}
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/registry/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister agent from registry"""
    try:
        success = await agent_registry.unregister_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"status": "unregistered"}
    except Exception as e:
        logger.error(f"Failed to unregister agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/registry/{agent_id}/heartbeat")
async def agent_heartbeat(agent_id: str, metrics: Optional[Dict[str, Any]] = None):
    """Record agent heartbeat"""
    try:
        success = await agent_registry.heartbeat(agent_id, metrics)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record heartbeat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/agents")
async def list_registered_agents():
    """List all registered agents"""
    try:
        agents = agent_registry.list_agents()
        return [agent.dict() for agent in agents]
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/agents/{agent_id}")
async def get_registered_agent(agent_id: str):
    """Get registered agent details"""
    try:
        agent = agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent.dict()
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/registry/services/find")
async def find_services(query: ServiceQuery):
    """Find services matching query"""
    try:
        results = agent_registry.find_services(query)
        return [
            {
                "agent": agent.dict(),
                "service": service.dict()
            }
            for agent, service in results
        ]
    except Exception as e:
        logger.error(f"Failed to find services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/stats")
async def get_registry_stats():
    """Get agent registry statistics"""
    try:
        return agent_registry.get_registry_stats()
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
