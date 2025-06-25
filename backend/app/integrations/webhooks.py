"""
Webhook System for Third-Party Integrations
"""

import asyncio
import logging
import hmac
import hashlib
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import json

from app.core.database import get_redis

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"


class WebhookEvent(str, Enum):
    """Webhook event types"""
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    MEMORY_UPDATED = "memory.updated"
    COLLABORATION_REQUEST = "collaboration.request"
    CUSTOM = "custom"


class WebhookConfig(BaseModel):
    """Webhook configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 5  # seconds
    status: WebhookStatus = WebhookStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookPayload(BaseModel):
    """Webhook payload"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event: WebhookEvent
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    source: str = "ai_service"
    version: str = "1.0"


class WebhookDelivery(BaseModel):
    """Webhook delivery record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    webhook_id: str
    payload: WebhookPayload
    url: str
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 1
    delivered_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IncomingWebhook(BaseModel):
    """Incoming webhook data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    headers: Dict[str, str]
    body: Any
    signature: Optional[str] = None
    verified: bool = False
    processed: bool = False
    received_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookSystem:
    """Webhook management system"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = {}
        self.incoming_webhooks: Dict[str, IncomingWebhook] = {}
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.redis_client = None
        
        # Background tasks
        self.delivery_worker_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_deliveries = 0
        self.successful_deliveries = 0
        self.failed_deliveries = 0
        
    async def initialize(self) -> None:
        """Initialize webhook system"""
        try:
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Start background workers
            self.delivery_worker_task = asyncio.create_task(self._delivery_worker())
            self.cleanup_task = asyncio.create_task(self._cleanup_worker())
            
            logger.info("Webhook system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize webhook system: {e}")
            raise
    
    def register_webhook(self, webhook: WebhookConfig) -> str:
        """Register a webhook"""
        self.webhooks[webhook.id] = webhook
        logger.info(f"Registered webhook: {webhook.name}")
        return webhook.id
    
    def register_event_handler(self, event: WebhookEvent, handler: Callable) -> None:
        """Register an event handler for incoming webhooks"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
        logger.info(f"Registered event handler for: {event}")
    
    async def trigger_webhook(self, event: WebhookEvent, data: Dict[str, Any]) -> List[str]:
        """Trigger webhooks for an event"""
        try:
            triggered_webhooks = []
            
            # Find webhooks that listen to this event
            for webhook in self.webhooks.values():
                if webhook.status == WebhookStatus.ACTIVE and event in webhook.events:
                    # Create payload
                    payload = WebhookPayload(
                        event=event,
                        data=data
                    )
                    
                    # Queue for delivery
                    await self.delivery_queue.put((webhook.id, payload))
                    triggered_webhooks.append(webhook.id)
            
            logger.info(f"Triggered {len(triggered_webhooks)} webhooks for event: {event}")
            return triggered_webhooks
            
        except Exception as e:
            logger.error(f"Failed to trigger webhooks: {e}")
            return []
    
    async def _delivery_worker(self) -> None:
        """Background worker for webhook delivery"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    # Get next delivery
                    webhook_id, payload = await self.delivery_queue.get()
                    
                    webhook = self.webhooks.get(webhook_id)
                    if not webhook:
                        continue
                    
                    # Attempt delivery
                    success = await self._deliver_webhook(client, webhook, payload)
                    
                    # Update webhook statistics
                    webhook.last_triggered = datetime.utcnow()
                    if success:
                        webhook.success_count += 1
                        self.successful_deliveries += 1
                    else:
                        webhook.failure_count += 1
                        self.failed_deliveries += 1
                        
                        # Suspend webhook if too many failures
                        if webhook.failure_count > 10:
                            webhook.status = WebhookStatus.SUSPENDED
                            logger.warning(f"Suspended webhook {webhook.name} due to repeated failures")
                    
                    self.total_deliveries += 1
                    
                except Exception as e:
                    logger.error(f"Error in delivery worker: {e}")
                    await asyncio.sleep(1)
    
    async def _deliver_webhook(self, client: httpx.AsyncClient, webhook: WebhookConfig, payload: WebhookPayload) -> bool:
        """Deliver webhook payload"""
        try:
            # Prepare headers
            headers = webhook.headers.copy()
            headers["Content-Type"] = "application/json"
            headers["User-Agent"] = "AI-Service-Webhook/1.0"
            headers["X-Webhook-Event"] = payload.event.value
            headers["X-Webhook-ID"] = payload.id
            headers["X-Webhook-Timestamp"] = payload.timestamp.isoformat()
            
            # Prepare body
            body = payload.json()
            
            # Add signature if secret is configured
            if webhook.secret:
                signature = self._generate_signature(webhook.secret, body)
                headers["X-Webhook-Signature"] = signature
            
            # Create delivery record
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                payload=payload,
                url=webhook.url
            )
            
            # Attempt delivery with retries
            for attempt in range(1, webhook.retry_count + 1):
                try:
                    delivery.attempt = attempt
                    
                    response = await client.post(
                        webhook.url,
                        headers=headers,
                        content=body,
                        timeout=webhook.timeout
                    )
                    
                    delivery.status_code = response.status_code
                    delivery.response_body = response.text[:1000]  # Limit response body
                    
                    if response.is_success:
                        delivery.delivered_at = datetime.utcnow()
                        await self._store_delivery_record(delivery)
                        logger.debug(f"Webhook delivered successfully: {webhook.name}")
                        return True
                    else:
                        delivery.error = f"HTTP {response.status_code}"
                        
                except httpx.TimeoutException:
                    delivery.error = "Timeout"
                except Exception as e:
                    delivery.error = str(e)
                
                # Wait before retry (except on last attempt)
                if attempt < webhook.retry_count:
                    await asyncio.sleep(webhook.retry_delay * attempt)
            
            # All attempts failed
            await self._store_delivery_record(delivery)
            logger.warning(f"Webhook delivery failed after {webhook.retry_count} attempts: {webhook.name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to deliver webhook: {e}")
            return False
    
    def _generate_signature(self, secret: str, body: str) -> str:
        """Generate webhook signature"""
        signature = hmac.new(
            secret.encode(),
            body.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def _verify_signature(self, secret: str, body: str, signature: str) -> bool:
        """Verify webhook signature"""
        try:
            expected_signature = self._generate_signature(secret, body)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    async def _store_delivery_record(self, delivery: WebhookDelivery) -> None:
        """Store delivery record in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"webhook_delivery:{delivery.id}"
            value = delivery.json()
            await self.redis_client.set(key, value, ex=86400 * 7)  # 7 days
        except Exception as e:
            logger.error(f"Failed to store delivery record: {e}")
    
    async def handle_incoming_webhook(
        self, 
        source: str, 
        headers: Dict[str, str], 
        body: Any,
        signature: Optional[str] = None
    ) -> str:
        """Handle incoming webhook"""
        try:
            # Create incoming webhook record
            incoming = IncomingWebhook(
                source=source,
                headers=headers,
                body=body,
                signature=signature
            )
            
            # Verify signature if provided
            if signature and source in self.webhooks:
                webhook = self.webhooks[source]
                if webhook.secret:
                    body_str = json.dumps(body) if isinstance(body, dict) else str(body)
                    incoming.verified = self._verify_signature(webhook.secret, body_str, signature)
            
            # Store incoming webhook
            self.incoming_webhooks[incoming.id] = incoming
            
            # Process webhook asynchronously
            asyncio.create_task(self._process_incoming_webhook(incoming))
            
            return incoming.id
            
        except Exception as e:
            logger.error(f"Failed to handle incoming webhook: {e}")
            raise
    
    async def _process_incoming_webhook(self, incoming: IncomingWebhook) -> None:
        """Process incoming webhook"""
        try:
            # Determine event type from body or headers
            event_type = self._determine_event_type(incoming)
            
            if event_type and event_type in self.event_handlers:
                # Call registered handlers
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(incoming)
                    except Exception as e:
                        logger.error(f"Webhook handler failed: {e}")
            
            incoming.processed = True
            incoming.processed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to process incoming webhook: {e}")
    
    def _determine_event_type(self, incoming: IncomingWebhook) -> Optional[WebhookEvent]:
        """Determine event type from incoming webhook"""
        try:
            # Check headers first
            event_header = incoming.headers.get("X-Event-Type") or incoming.headers.get("X-Webhook-Event")
            if event_header:
                try:
                    return WebhookEvent(event_header)
                except ValueError:
                    pass
            
            # Check body
            if isinstance(incoming.body, dict):
                event_field = incoming.body.get("event") or incoming.body.get("type")
                if event_field:
                    try:
                        return WebhookEvent(event_field)
                    except ValueError:
                        pass
            
            # Default to custom
            return WebhookEvent.CUSTOM
            
        except Exception as e:
            logger.error(f"Failed to determine event type: {e}")
            return None
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old incoming webhooks
                cutoff = datetime.utcnow() - timedelta(days=7)
                to_remove = [
                    webhook_id for webhook_id, webhook in self.incoming_webhooks.items()
                    if webhook.received_at < cutoff
                ]
                
                for webhook_id in to_remove:
                    del self.incoming_webhooks[webhook_id]
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old incoming webhooks")
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self, status: Optional[WebhookStatus] = None) -> List[WebhookConfig]:
        """List webhooks, optionally filtered by status"""
        webhooks = list(self.webhooks.values())
        
        if status:
            webhooks = [wh for wh in webhooks if wh.status == status]
        
        return webhooks
    
    def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> bool:
        """Update webhook configuration"""
        try:
            if webhook_id not in self.webhooks:
                return False
            
            webhook = self.webhooks[webhook_id]
            
            for key, value in updates.items():
                if hasattr(webhook, key):
                    setattr(webhook, key, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update webhook: {e}")
            return False
    
    def remove_webhook(self, webhook_id: str) -> bool:
        """Remove webhook"""
        try:
            if webhook_id in self.webhooks:
                del self.webhooks[webhook_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove webhook: {e}")
            return False
    
    async def test_webhook(self, webhook_id: str) -> bool:
        """Test webhook with a sample payload"""
        try:
            webhook = self.webhooks.get(webhook_id)
            if not webhook:
                return False
            
            # Create test payload
            test_payload = WebhookPayload(
                event=WebhookEvent.CUSTOM,
                data={"test": True, "message": "This is a test webhook"}
            )
            
            # Queue for delivery
            await self.delivery_queue.put((webhook_id, test_payload))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to test webhook: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook system statistics"""
        active_webhooks = len([wh for wh in self.webhooks.values() if wh.status == WebhookStatus.ACTIVE])
        success_rate = self.successful_deliveries / self.total_deliveries if self.total_deliveries > 0 else 0
        
        return {
            "total_webhooks": len(self.webhooks),
            "active_webhooks": active_webhooks,
            "total_deliveries": self.total_deliveries,
            "successful_deliveries": self.successful_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "success_rate": success_rate,
            "pending_deliveries": self.delivery_queue.qsize(),
            "incoming_webhooks": len(self.incoming_webhooks)
        }
    
    async def shutdown(self) -> None:
        """Shutdown webhook system"""
        try:
            # Cancel background tasks
            if self.delivery_worker_task:
                self.delivery_worker_task.cancel()
                try:
                    await self.delivery_worker_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Webhook system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during webhook system shutdown: {e}")


# Global webhook system instance
webhook_system = WebhookSystem()
