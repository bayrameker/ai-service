"""
Integrations API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional, Dict, Any
import logging

from app.integrations.api_gateway import APIEndpoint, APIRequest, api_gateway
from app.integrations.webhooks import WebhookConfig, WebhookEvent, webhook_system

logger = logging.getLogger(__name__)
router = APIRouter()


# API Gateway endpoints
@router.post("/api-endpoints")
async def register_api_endpoint(endpoint: APIEndpoint):
    """Register a new API endpoint"""
    try:
        endpoint_id = api_gateway.register_endpoint(endpoint)
        return {"endpoint_id": endpoint_id, "status": "registered"}
    except Exception as e:
        logger.error(f"Failed to register API endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-endpoints")
async def list_api_endpoints(tag: Optional[str] = None):
    """List API endpoints"""
    try:
        endpoints = api_gateway.list_endpoints(tag)
        return [endpoint.dict() for endpoint in endpoints]
    except Exception as e:
        logger.error(f"Failed to list API endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-endpoints/{endpoint_id}")
async def get_api_endpoint(endpoint_id: str):
    """Get API endpoint details"""
    try:
        endpoint = api_gateway.get_endpoint(endpoint_id)
        if not endpoint:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        return endpoint.dict()
    except Exception as e:
        logger.error(f"Failed to get API endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api-endpoints/{endpoint_id}/request")
async def make_api_request(endpoint_id: str, request: APIRequest):
    """Make API request to endpoint"""
    try:
        request.endpoint_id = endpoint_id
        response = await api_gateway.make_request(request)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to make API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api-endpoints/{endpoint_id}/test")
async def test_api_endpoint(endpoint_id: str):
    """Test API endpoint"""
    try:
        response = await api_gateway.test_endpoint(endpoint_id)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to test API endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api-endpoints/{endpoint_id}")
async def update_api_endpoint(endpoint_id: str, updates: Dict[str, Any]):
    """Update API endpoint"""
    try:
        success = api_gateway.update_endpoint(endpoint_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Failed to update API endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api-endpoints/{endpoint_id}")
async def remove_api_endpoint(endpoint_id: str):
    """Remove API endpoint"""
    try:
        success = api_gateway.remove_endpoint(endpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        return {"status": "removed"}
    except Exception as e:
        logger.error(f"Failed to remove API endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-gateway/stats")
async def get_api_gateway_stats():
    """Get API gateway statistics"""
    try:
        return api_gateway.get_stats()
    except Exception as e:
        logger.error(f"Failed to get API gateway stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-gateway/health")
async def check_api_gateway_health():
    """Check API gateway health"""
    try:
        health = await api_gateway.health_check()
        return health
    except Exception as e:
        logger.error(f"Failed to check API gateway health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Webhook endpoints
@router.post("/webhooks")
async def register_webhook(webhook: WebhookConfig):
    """Register a new webhook"""
    try:
        webhook_id = webhook_system.register_webhook(webhook)
        return {"webhook_id": webhook_id, "status": "registered"}
    except Exception as e:
        logger.error(f"Failed to register webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks")
async def list_webhooks():
    """List webhooks"""
    try:
        webhooks = webhook_system.list_webhooks()
        return [webhook.dict() for webhook in webhooks]
    except Exception as e:
        logger.error(f"Failed to list webhooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks/{webhook_id}")
async def get_webhook(webhook_id: str):
    """Get webhook details"""
    try:
        webhook = webhook_system.get_webhook(webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return webhook.dict()
    except Exception as e:
        logger.error(f"Failed to get webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/{webhook_id}/trigger")
async def trigger_webhook(webhook_id: str, event: WebhookEvent, data: Dict[str, Any]):
    """Trigger a specific webhook"""
    try:
        triggered = await webhook_system.trigger_webhook(event, data)
        return {"triggered_webhooks": triggered}
    except Exception as e:
        logger.error(f"Failed to trigger webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    """Test webhook"""
    try:
        success = await webhook_system.test_webhook(webhook_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"status": "test_sent"}
    except Exception as e:
        logger.error(f"Failed to test webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/webhooks/{webhook_id}")
async def update_webhook(webhook_id: str, updates: Dict[str, Any]):
    """Update webhook"""
    try:
        success = webhook_system.update_webhook(webhook_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Failed to update webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/webhooks/{webhook_id}")
async def remove_webhook(webhook_id: str):
    """Remove webhook"""
    try:
        success = webhook_system.remove_webhook(webhook_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"status": "removed"}
    except Exception as e:
        logger.error(f"Failed to remove webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/incoming/{source}")
async def handle_incoming_webhook(
    source: str,
    request: Request,
    signature: Optional[str] = None
):
    """Handle incoming webhook"""
    try:
        # Get request body and headers
        body = await request.json()
        headers = dict(request.headers)
        
        webhook_id = await webhook_system.handle_incoming_webhook(
            source, headers, body, signature
        )
        
        return {"webhook_id": webhook_id, "status": "received"}
    except Exception as e:
        logger.error(f"Failed to handle incoming webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks/stats")
async def get_webhook_stats():
    """Get webhook system statistics"""
    try:
        return webhook_system.get_stats()
    except Exception as e:
        logger.error(f"Failed to get webhook stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
