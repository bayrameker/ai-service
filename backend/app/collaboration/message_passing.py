"""
Message Passing System for Agent-to-Agent Communication
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import json

from app.core.database import get_redis

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Message types for agent communication"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """Message delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


class Message(BaseModel):
    """Message model for agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    receiver_group: Optional[str] = None  # For multicast
    type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    subject: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # For request-response
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3


class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler(self, subject: str, handler: Callable) -> None:
        """Register a message handler for a specific subject"""
        self.handlers[subject] = handler
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message"""
        try:
            handler = self.handlers.get(message.subject)
            if handler:
                result = await handler(message)
                
                # If handler returns a response, create response message
                if result and message.type == MessageType.REQUEST:
                    response = Message(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        type=MessageType.RESPONSE,
                        subject=f"RE: {message.subject}",
                        content=result,
                        correlation_id=message.id
                    )
                    return response
            else:
                logger.warning(f"No handler for message subject: {message.subject}")
                
        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")
        
        return None


class MessageQueue:
    """Message queue for agent communication"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.pending_messages: Dict[str, Message] = {}
        self.message_history: List[Message] = []
        self.max_history = 1000
    
    async def send_message(self, message: Message) -> None:
        """Add message to outbox"""
        message.sender_id = self.agent_id
        message.sent_at = datetime.utcnow()
        message.status = MessageStatus.SENT
        
        await self.outbox.put(message)
        self.pending_messages[message.id] = message
        
        # Add to history
        self._add_to_history(message)
    
    async def receive_message(self, message: Message) -> None:
        """Add message to inbox"""
        message.delivered_at = datetime.utcnow()
        message.status = MessageStatus.DELIVERED
        
        await self.inbox.put(message)
        
        # Add to history
        self._add_to_history(message)
    
    async def get_next_outgoing(self) -> Optional[Message]:
        """Get next message from outbox"""
        try:
            return await asyncio.wait_for(self.outbox.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def get_next_incoming(self) -> Optional[Message]:
        """Get next message from inbox"""
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def acknowledge_message(self, message_id: str) -> bool:
        """Acknowledge message delivery"""
        if message_id in self.pending_messages:
            message = self.pending_messages[message_id]
            message.acknowledged_at = datetime.utcnow()
            message.status = MessageStatus.ACKNOWLEDGED
            del self.pending_messages[message_id]
            return True
        return False
    
    def _add_to_history(self, message: Message) -> None:
        """Add message to history"""
        self.message_history.append(message)
        
        # Keep only recent messages
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
    
    def get_conversation_history(self, other_agent_id: str, limit: int = 50) -> List[Message]:
        """Get conversation history with another agent"""
        conversation = []
        
        for message in reversed(self.message_history):
            if ((message.sender_id == self.agent_id and message.receiver_id == other_agent_id) or
                (message.sender_id == other_agent_id and message.receiver_id == self.agent_id)):
                conversation.append(message)
                
                if len(conversation) >= limit:
                    break
        
        return list(reversed(conversation))


class MessageBroker:
    """Central message broker for agent communication"""
    
    def __init__(self):
        self.agents: Dict[str, MessageQueue] = {}
        self.message_handlers: Dict[str, MessageHandler] = {}
        self.groups: Dict[str, List[str]] = {}  # Group name -> agent IDs
        self.redis_client = None
        
        # Background tasks
        self.broker_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_messages = 0
        self.delivered_messages = 0
        self.failed_messages = 0
    
    async def initialize(self) -> None:
        """Initialize message broker"""
        try:
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Start background tasks
            self.broker_task = asyncio.create_task(self._broker_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Message broker initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize message broker: {e}")
            raise
    
    def register_agent(self, agent_id: str) -> MessageQueue:
        """Register an agent with the broker"""
        if agent_id not in self.agents:
            self.agents[agent_id] = MessageQueue(agent_id)
            self.message_handlers[agent_id] = MessageHandler(agent_id)
            logger.info(f"Registered agent {agent_id} with message broker")
        
        return self.agents[agent_id]
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the broker"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.message_handlers[agent_id]
            
            # Remove from groups
            for group_agents in self.groups.values():
                if agent_id in group_agents:
                    group_agents.remove(agent_id)
            
            logger.info(f"Unregistered agent {agent_id} from message broker")
    
    def create_group(self, group_name: str, agent_ids: List[str]) -> None:
        """Create a message group"""
        self.groups[group_name] = agent_ids.copy()
        logger.info(f"Created group {group_name} with {len(agent_ids)} agents")
    
    def add_to_group(self, group_name: str, agent_id: str) -> None:
        """Add agent to group"""
        if group_name not in self.groups:
            self.groups[group_name] = []
        
        if agent_id not in self.groups[group_name]:
            self.groups[group_name].append(agent_id)
    
    def remove_from_group(self, group_name: str, agent_id: str) -> None:
        """Remove agent from group"""
        if group_name in self.groups and agent_id in self.groups[group_name]:
            self.groups[group_name].remove(agent_id)
    
    async def send_message(self, message: Message) -> bool:
        """Send message through broker"""
        try:
            sender_queue = self.agents.get(message.sender_id)
            if not sender_queue:
                logger.error(f"Sender agent {message.sender_id} not registered")
                return False
            
            await sender_queue.send_message(message)
            self.total_messages += 1
            
            # Store in Redis for persistence
            await self._store_message(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def _broker_loop(self) -> None:
        """Main broker loop for message routing"""
        while True:
            try:
                # Process outgoing messages from all agents
                for agent_id, queue in self.agents.items():
                    message = await queue.get_next_outgoing()
                    if message:
                        await self._route_message(message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in broker loop: {e}")
                await asyncio.sleep(1)
    
    async def _route_message(self, message: Message) -> None:
        """Route message to appropriate recipients"""
        try:
            recipients = []
            
            if message.type == MessageType.DIRECT and message.receiver_id:
                # Direct message to specific agent
                recipients = [message.receiver_id]
                
            elif message.type == MessageType.BROADCAST:
                # Broadcast to all agents except sender
                recipients = [aid for aid in self.agents.keys() if aid != message.sender_id]
                
            elif message.type == MessageType.MULTICAST and message.receiver_group:
                # Multicast to group members
                recipients = self.groups.get(message.receiver_group, [])
                # Remove sender from recipients
                recipients = [aid for aid in recipients if aid != message.sender_id]
            
            # Deliver to recipients
            delivered_count = 0
            for recipient_id in recipients:
                if await self._deliver_message(message, recipient_id):
                    delivered_count += 1
            
            if delivered_count > 0:
                self.delivered_messages += 1
            else:
                self.failed_messages += 1
                message.status = MessageStatus.FAILED
            
        except Exception as e:
            logger.error(f"Failed to route message {message.id}: {e}")
            message.status = MessageStatus.FAILED
            self.failed_messages += 1
    
    async def _deliver_message(self, message: Message, recipient_id: str) -> bool:
        """Deliver message to specific recipient"""
        try:
            recipient_queue = self.agents.get(recipient_id)
            if not recipient_queue:
                logger.warning(f"Recipient agent {recipient_id} not found")
                return False
            
            # Create copy for recipient
            message_copy = message.copy()
            message_copy.receiver_id = recipient_id
            
            await recipient_queue.receive_message(message_copy)
            
            # Handle message if handler is registered
            handler = self.message_handlers.get(recipient_id)
            if handler:
                response = await handler.handle_message(message_copy)
                if response:
                    # Send response back
                    await self.send_message(response)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deliver message to {recipient_id}: {e}")
            return False
    
    async def _store_message(self, message: Message) -> None:
        """Store message in Redis for persistence"""
        if not self.redis_client:
            return
        
        try:
            key = f"message:{message.id}"
            value = message.json()
            await self.redis_client.set(key, value, ex=86400)  # 24 hours
        except Exception as e:
            logger.error(f"Failed to store message in Redis: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup expired messages"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = datetime.utcnow()
                
                # Clean up expired messages from all queues
                for queue in self.agents.values():
                    expired_messages = [
                        msg_id for msg_id, msg in queue.pending_messages.items()
                        if msg.expires_at and now > msg.expires_at
                    ]
                    
                    for msg_id in expired_messages:
                        message = queue.pending_messages[msg_id]
                        message.status = MessageStatus.EXPIRED
                        del queue.pending_messages[msg_id]
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_agent_queue(self, agent_id: str) -> Optional[MessageQueue]:
        """Get message queue for agent"""
        return self.agents.get(agent_id)
    
    def get_agent_handler(self, agent_id: str) -> Optional[MessageHandler]:
        """Get message handler for agent"""
        return self.message_handlers.get(agent_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message broker statistics"""
        total_pending = sum(len(queue.pending_messages) for queue in self.agents.values())
        total_inbox = sum(queue.inbox.qsize() for queue in self.agents.values())
        total_outbox = sum(queue.outbox.qsize() for queue in self.agents.values())
        
        return {
            "registered_agents": len(self.agents),
            "total_groups": len(self.groups),
            "total_messages": self.total_messages,
            "delivered_messages": self.delivered_messages,
            "failed_messages": self.failed_messages,
            "pending_messages": total_pending,
            "inbox_messages": total_inbox,
            "outbox_messages": total_outbox,
            "delivery_rate": self.delivered_messages / self.total_messages if self.total_messages > 0 else 0
        }
    
    async def shutdown(self) -> None:
        """Shutdown message broker"""
        try:
            # Cancel background tasks
            if self.broker_task:
                self.broker_task.cancel()
                try:
                    await self.broker_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Message broker shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during message broker shutdown: {e}")


# Global message broker instance
message_broker = MessageBroker()
