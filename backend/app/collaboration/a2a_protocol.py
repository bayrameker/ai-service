"""
Agent-to-Agent (A2A) Protocol Implementation
Based on Google's Agent2Agent protocol specification
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import httpx

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """A2A message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    CAPABILITY = "capability"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    COLLABORATION_INVITE = "collaboration_invite"
    COLLABORATION_ACCEPT = "collaboration_accept"
    COLLABORATION_REJECT = "collaboration_reject"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class A2AMessage(BaseModel):
    """A2A protocol message"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # For request-response correlation
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentCapability(BaseModel):
    """Agent capability description"""
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    cost: Optional[float] = None
    latency_ms: Optional[int] = None
    availability: float = 1.0  # 0.0 to 1.0


class AgentInfo(BaseModel):
    """Agent information for discovery"""
    id: str
    name: str
    description: str
    endpoint: str
    capabilities: List[AgentCapability] = Field(default_factory=list)
    status: str = "online"
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class A2AProtocol:
    """A2A Protocol implementation"""
    
    def __init__(self, agent_id: str, agent_name: str, endpoint: str, port: int = 8001):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.endpoint = endpoint
        self.port = port
        
        # Protocol state
        self.known_agents: Dict[str, AgentInfo] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_requests: Dict[str, A2AMessage] = {}
        self.capabilities: List[AgentCapability] = []
        
        # Network
        self.http_client: Optional[httpx.AsyncClient] = None
        self.server_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.discovery_interval = 60  # seconds
        self.heartbeat_interval = 30  # seconds
        self.message_timeout = 30  # seconds
        
        # Setup default handlers
        self._setup_default_handlers()
    
    async def initialize(self) -> None:
        """Initialize the A2A protocol"""
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
            
            # Start HTTP server for receiving messages
            await self._start_server()
            
            # Start background tasks
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"A2A Protocol initialized for agent {self.agent_id} on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A protocol: {e}")
            raise
    
    def register_capability(self, capability: AgentCapability) -> None:
        """Register an agent capability"""
        self.capabilities.append(capability)
        logger.info(f"Registered capability: {capability.name}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def send_message(self, message: A2AMessage) -> bool:
        """Send a message to another agent"""
        try:
            if not message.receiver_id:
                # Broadcast message
                return await self._broadcast_message(message)
            
            # Get receiver info
            receiver_info = self.known_agents.get(message.receiver_id)
            if not receiver_info:
                logger.warning(f"Unknown receiver: {message.receiver_id}")
                return False
            
            # Send message
            response = await self.http_client.post(
                f"{receiver_info.endpoint}/a2a/message",
                json=message.dict(),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.debug(f"Message sent successfully to {message.receiver_id}")
                return True
            else:
                logger.warning(f"Failed to send message to {message.receiver_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def send_request(self, receiver_id: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[A2AMessage]:
        """Send a request and wait for response"""
        try:
            request = A2AMessage(
                type=MessageType.REQUEST,
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                payload=payload,
                expires_at=datetime.utcnow() + timedelta(seconds=timeout)
            )
            
            # Store pending request
            self.pending_requests[request.id] = request
            
            # Send request
            success = await self.send_message(request)
            if not success:
                del self.pending_requests[request.id]
                return None
            
            # Wait for response
            start_time = datetime.utcnow()
            while (datetime.utcnow() - start_time).total_seconds() < timeout:
                # Check if response received (would be handled by message handler)
                if request.id not in self.pending_requests:
                    # Response was received and processed
                    break
                
                await asyncio.sleep(0.1)
            
            # Cleanup
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
                return None
            
            # Response would be stored in a response cache (simplified here)
            return None
            
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            return None
    
    async def discover_agents(self) -> List[AgentInfo]:
        """Discover other agents in the network"""
        try:
            # Send discovery message
            discovery_message = A2AMessage(
                type=MessageType.DISCOVERY,
                sender_id=self.agent_id,
                payload={
                    "agent_info": {
                        "id": self.agent_id,
                        "name": self.agent_name,
                        "endpoint": self.endpoint,
                        "capabilities": [cap.dict() for cap in self.capabilities]
                    }
                }
            )
            
            await self._broadcast_message(discovery_message)
            
            return list(self.known_agents.values())
            
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return []
    
    async def request_collaboration(self, agent_id: str, task_description: str, requirements: Dict[str, Any]) -> bool:
        """Request collaboration with another agent"""
        try:
            collaboration_request = A2AMessage(
                type=MessageType.COLLABORATION_INVITE,
                sender_id=self.agent_id,
                receiver_id=agent_id,
                payload={
                    "task_description": task_description,
                    "requirements": requirements,
                    "capabilities_needed": requirements.get("capabilities", [])
                }
            )
            
            return await self.send_message(collaboration_request)
            
        except Exception as e:
            logger.error(f"Failed to request collaboration: {e}")
            return False
    
    async def assign_task(self, agent_id: str, task: Dict[str, Any]) -> bool:
        """Assign a task to another agent"""
        try:
            task_message = A2AMessage(
                type=MessageType.TASK_ASSIGNMENT,
                sender_id=self.agent_id,
                receiver_id=agent_id,
                payload={"task": task}
            )
            
            return await self.send_message(task_message)
            
        except Exception as e:
            logger.error(f"Failed to assign task: {e}")
            return False
    
    async def _broadcast_message(self, message: A2AMessage) -> bool:
        """Broadcast message to all known agents"""
        try:
            success_count = 0
            
            for agent_info in self.known_agents.values():
                if agent_info.id != self.agent_id:  # Don't send to self
                    message_copy = message.copy()
                    message_copy.receiver_id = agent_info.id
                    
                    if await self.send_message(message_copy):
                        success_count += 1
            
            logger.debug(f"Broadcast message sent to {success_count} agents")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False
    
    async def _start_server(self) -> None:
        """Start HTTP server for receiving messages"""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            import uvicorn
            
            app = FastAPI(title=f"A2A Agent {self.agent_id}")
            
            @app.post("/a2a/message")
            async def receive_message(message_data: dict):
                try:
                    message = A2AMessage(**message_data)
                    await self._handle_received_message(message)
                    return {"status": "received"}
                except Exception as e:
                    logger.error(f"Failed to handle received message: {e}")
                    raise HTTPException(status_code=400, detail=str(e))
            
            @app.get("/a2a/info")
            async def get_agent_info():
                return {
                    "id": self.agent_id,
                    "name": self.agent_name,
                    "endpoint": self.endpoint,
                    "capabilities": [cap.dict() for cap in self.capabilities],
                    "status": "online"
                }
            
            @app.get("/a2a/health")
            async def health_check():
                return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
            
            # Start server in background
            config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="warning")
            server = uvicorn.Server(config)
            self.server_task = asyncio.create_task(server.serve())
            
        except Exception as e:
            logger.error(f"Failed to start A2A server: {e}")
            raise
    
    async def _handle_received_message(self, message: A2AMessage) -> None:
        """Handle received message"""
        try:
            logger.debug(f"Received message type {message.type} from {message.sender_id}")
            
            # Check if message is expired
            if message.expires_at and datetime.utcnow() > message.expires_at:
                logger.warning(f"Received expired message {message.id}")
                return
            
            # Handle message based on type
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.type}")
            
        except Exception as e:
            logger.error(f"Failed to handle received message: {e}")
    
    def _setup_default_handlers(self) -> None:
        """Setup default message handlers"""
        
        async def handle_discovery(message: A2AMessage):
            """Handle discovery message"""
            try:
                agent_info_data = message.payload.get("agent_info")
                if agent_info_data:
                    agent_info = AgentInfo(**agent_info_data)
                    self.known_agents[agent_info.id] = agent_info
                    logger.info(f"Discovered agent: {agent_info.name} ({agent_info.id})")
                    
                    # Send our info back
                    response = A2AMessage(
                        type=MessageType.DISCOVERY,
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        payload={
                            "agent_info": {
                                "id": self.agent_id,
                                "name": self.agent_name,
                                "endpoint": self.endpoint,
                                "capabilities": [cap.dict() for cap in self.capabilities]
                            }
                        }
                    )
                    await self.send_message(response)
            except Exception as e:
                logger.error(f"Failed to handle discovery message: {e}")
        
        async def handle_heartbeat(message: A2AMessage):
            """Handle heartbeat message"""
            try:
                if message.sender_id in self.known_agents:
                    self.known_agents[message.sender_id].last_seen = datetime.utcnow()
                    self.known_agents[message.sender_id].status = "online"
            except Exception as e:
                logger.error(f"Failed to handle heartbeat message: {e}")
        
        async def handle_response(message: A2AMessage):
            """Handle response message"""
            try:
                correlation_id = message.correlation_id
                if correlation_id and correlation_id in self.pending_requests:
                    # Response received for pending request
                    del self.pending_requests[correlation_id]
                    # Store response (simplified - in real implementation, would notify waiting coroutine)
            except Exception as e:
                logger.error(f"Failed to handle response message: {e}")
        
        async def handle_collaboration_invite(message: A2AMessage):
            """Handle collaboration invite"""
            try:
                task_description = message.payload.get("task_description")
                requirements = message.payload.get("requirements", {})
                
                # Simple auto-accept logic (in real implementation, would have more sophisticated decision making)
                capabilities_needed = requirements.get("capabilities", [])
                can_help = any(cap.name in capabilities_needed for cap in self.capabilities)
                
                response_type = MessageType.COLLABORATION_ACCEPT if can_help else MessageType.COLLABORATION_REJECT
                
                response = A2AMessage(
                    type=response_type,
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    correlation_id=message.id,
                    payload={
                        "accepted": can_help,
                        "reason": "Have required capabilities" if can_help else "Missing required capabilities"
                    }
                )
                
                await self.send_message(response)
                
            except Exception as e:
                logger.error(f"Failed to handle collaboration invite: {e}")
        
        # Register default handlers
        self.register_message_handler(MessageType.DISCOVERY, handle_discovery)
        self.register_message_handler(MessageType.HEARTBEAT, handle_heartbeat)
        self.register_message_handler(MessageType.RESPONSE, handle_response)
        self.register_message_handler(MessageType.COLLABORATION_INVITE, handle_collaboration_invite)
    
    async def _discovery_loop(self) -> None:
        """Periodic agent discovery"""
        while True:
            try:
                await asyncio.sleep(self.discovery_interval)
                await self.discover_agents()
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat sending"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                heartbeat = A2AMessage(
                    type=MessageType.HEARTBEAT,
                    sender_id=self.agent_id,
                    payload={"status": "online", "timestamp": datetime.utcnow().isoformat()}
                )
                
                await self._broadcast_message(heartbeat)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired data"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                now = datetime.utcnow()
                
                # Remove expired pending requests
                expired_requests = [
                    req_id for req_id, req in self.pending_requests.items()
                    if req.expires_at and now > req.expires_at
                ]
                
                for req_id in expired_requests:
                    del self.pending_requests[req_id]
                
                # Mark agents as offline if not seen recently
                for agent_info in self.known_agents.values():
                    if (now - agent_info.last_seen).total_seconds() > 120:  # 2 minutes
                        agent_info.status = "offline"
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_known_agents(self) -> List[AgentInfo]:
        """Get list of known agents"""
        return list(self.known_agents.values())
    
    def get_online_agents(self) -> List[AgentInfo]:
        """Get list of online agents"""
        return [agent for agent in self.known_agents.values() if agent.status == "online"]
    
    def find_agents_with_capability(self, capability_name: str) -> List[AgentInfo]:
        """Find agents with specific capability"""
        matching_agents = []
        
        for agent in self.known_agents.values():
            if any(cap.name == capability_name for cap in agent.capabilities):
                matching_agents.append(agent)
        
        return matching_agents
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get A2A protocol statistics"""
        return {
            "agent_id": self.agent_id,
            "known_agents": len(self.known_agents),
            "online_agents": len(self.get_online_agents()),
            "registered_capabilities": len(self.capabilities),
            "pending_requests": len(self.pending_requests),
            "message_handlers": len(self.message_handlers)
        }
    
    async def shutdown(self) -> None:
        """Shutdown A2A protocol"""
        try:
            # Cancel server task
            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
            
            logger.info("A2A Protocol shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during A2A protocol shutdown: {e}")
