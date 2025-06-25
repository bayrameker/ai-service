"""
REST API Gateway for Third-Party Integrations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import httpx
import json
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(str, Enum):
    """Authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


class APIEndpoint(BaseModel):
    """API endpoint configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    base_url: str
    path: str
    method: HTTPMethod
    auth_type: AuthType = AuthType.NONE
    auth_config: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None  # requests per minute
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class APIRequest(BaseModel):
    """API request model"""
    endpoint_id: str
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None


class APIResponse(BaseModel):
    """API response model"""
    success: bool
    status_code: int
    data: Optional[Any] = None
    error: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    latency_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[datetime] = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        return len(self.requests) < self.max_requests
    
    def record_request(self) -> None:
        """Record a request"""
        self.requests.append(datetime.utcnow())


class APIGateway:
    """REST API Gateway for third-party integrations"""
    
    def __init__(self):
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.request_hooks: List[Callable] = []
        self.response_hooks: List[Callable] = []
        
        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        
    async def initialize(self) -> None:
        """Initialize the API gateway"""
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            
            logger.info("API Gateway initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize API Gateway: {e}")
            raise
    
    def register_endpoint(self, endpoint: APIEndpoint) -> str:
        """Register an API endpoint"""
        self.endpoints[endpoint.id] = endpoint
        
        # Setup rate limiter if specified
        if endpoint.rate_limit:
            self.rate_limiters[endpoint.id] = RateLimiter(endpoint.rate_limit)
        
        logger.info(f"Registered API endpoint: {endpoint.name}")
        return endpoint.id
    
    def add_request_hook(self, hook: Callable) -> None:
        """Add request hook (called before making request)"""
        self.request_hooks.append(hook)
    
    def add_response_hook(self, hook: Callable) -> None:
        """Add response hook (called after receiving response)"""
        self.response_hooks.append(hook)
    
    async def make_request(self, request: APIRequest) -> APIResponse:
        """Make API request"""
        start_time = datetime.utcnow()
        
        try:
            # Get endpoint configuration
            endpoint = self.endpoints.get(request.endpoint_id)
            if not endpoint:
                return APIResponse(
                    success=False,
                    status_code=404,
                    error=f"Endpoint {request.endpoint_id} not found",
                    latency_ms=0
                )
            
            if not endpoint.enabled:
                return APIResponse(
                    success=False,
                    status_code=503,
                    error="Endpoint is disabled",
                    latency_ms=0
                )
            
            # Check rate limit
            if endpoint.id in self.rate_limiters:
                rate_limiter = self.rate_limiters[endpoint.id]
                if not rate_limiter.can_make_request():
                    return APIResponse(
                        success=False,
                        status_code=429,
                        error="Rate limit exceeded",
                        latency_ms=0
                    )
                rate_limiter.record_request()
            
            # Build request
            url = f"{endpoint.base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"
            
            # Prepare headers
            headers = endpoint.headers.copy()
            if request.headers:
                headers.update(request.headers)
            
            # Add authentication
            await self._add_authentication(endpoint, headers)
            
            # Prepare query parameters
            params = endpoint.query_params.copy()
            if request.params:
                params.update(request.params)
            
            # Prepare request data
            json_data = request.data if request.data else None
            
            # Execute request hooks
            for hook in self.request_hooks:
                try:
                    await hook(endpoint, request)
                except Exception as e:
                    logger.warning(f"Request hook failed: {e}")
            
            # Make HTTP request
            timeout = request.timeout or endpoint.timeout
            
            response = await self.http_client.request(
                method=endpoint.method.value,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=timeout
            )
            
            # Calculate latency
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Parse response
            try:
                response_data = response.json() if response.content else None
            except json.JSONDecodeError:
                response_data = response.text if response.content else None
            
            # Create response object
            api_response = APIResponse(
                success=response.is_success,
                status_code=response.status_code,
                data=response_data,
                error=None if response.is_success else f"HTTP {response.status_code}",
                headers=dict(response.headers),
                latency_ms=latency_ms
            )
            
            # Execute response hooks
            for hook in self.response_hooks:
                try:
                    await hook(endpoint, request, api_response)
                except Exception as e:
                    logger.warning(f"Response hook failed: {e}")
            
            # Update statistics
            self.request_count += 1
            self.total_latency += latency_ms
            if not api_response.success:
                self.error_count += 1
            
            return api_response
            
        except httpx.TimeoutException:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.request_count += 1
            self.error_count += 1
            
            return APIResponse(
                success=False,
                status_code=408,
                error="Request timeout",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.request_count += 1
            self.error_count += 1
            
            logger.error(f"API request failed: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error=str(e),
                latency_ms=latency_ms
            )
    
    async def _add_authentication(self, endpoint: APIEndpoint, headers: Dict[str, str]) -> None:
        """Add authentication to request headers"""
        try:
            if endpoint.auth_type == AuthType.API_KEY:
                api_key = endpoint.auth_config.get("api_key")
                header_name = endpoint.auth_config.get("header_name", "X-API-Key")
                if api_key:
                    headers[header_name] = api_key
            
            elif endpoint.auth_type == AuthType.BEARER_TOKEN:
                token = endpoint.auth_config.get("token")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            
            elif endpoint.auth_type == AuthType.BASIC_AUTH:
                username = endpoint.auth_config.get("username")
                password = endpoint.auth_config.get("password")
                if username and password:
                    import base64
                    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                    headers["Authorization"] = f"Basic {credentials}"
            
            elif endpoint.auth_type == AuthType.CUSTOM:
                custom_headers = endpoint.auth_config.get("headers", {})
                headers.update(custom_headers)
            
        except Exception as e:
            logger.error(f"Failed to add authentication: {e}")
    
    async def test_endpoint(self, endpoint_id: str) -> APIResponse:
        """Test an endpoint with a simple request"""
        try:
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint:
                return APIResponse(
                    success=False,
                    status_code=404,
                    error="Endpoint not found",
                    latency_ms=0
                )
            
            # Create test request
            test_request = APIRequest(
                endpoint_id=endpoint_id,
                data={} if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH] else None
            )
            
            return await self.make_request(test_request)
            
        except Exception as e:
            logger.error(f"Failed to test endpoint: {e}")
            return APIResponse(
                success=False,
                status_code=500,
                error=str(e),
                latency_ms=0
            )
    
    def get_endpoint(self, endpoint_id: str) -> Optional[APIEndpoint]:
        """Get endpoint by ID"""
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(self, tag: Optional[str] = None) -> List[APIEndpoint]:
        """List all endpoints, optionally filtered by tag"""
        endpoints = list(self.endpoints.values())
        
        if tag:
            endpoints = [ep for ep in endpoints if tag in ep.tags]
        
        return endpoints
    
    def update_endpoint(self, endpoint_id: str, updates: Dict[str, Any]) -> bool:
        """Update endpoint configuration"""
        try:
            if endpoint_id not in self.endpoints:
                return False
            
            endpoint = self.endpoints[endpoint_id]
            
            for key, value in updates.items():
                if hasattr(endpoint, key):
                    setattr(endpoint, key, value)
            
            # Update rate limiter if rate_limit changed
            if "rate_limit" in updates:
                if updates["rate_limit"]:
                    self.rate_limiters[endpoint_id] = RateLimiter(updates["rate_limit"])
                elif endpoint_id in self.rate_limiters:
                    del self.rate_limiters[endpoint_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update endpoint: {e}")
            return False
    
    def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove endpoint"""
        try:
            if endpoint_id in self.endpoints:
                del self.endpoints[endpoint_id]
            
            if endpoint_id in self.rate_limiters:
                del self.rate_limiters[endpoint_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove endpoint: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API gateway statistics"""
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        return {
            "total_endpoints": len(self.endpoints),
            "enabled_endpoints": len([ep for ep in self.endpoints.values() if ep.enabled]),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "average_latency_ms": avg_latency,
            "rate_limited_endpoints": len(self.rate_limiters)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all endpoints"""
        health_results = {}
        
        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint.enabled:
                try:
                    response = await self.test_endpoint(endpoint_id)
                    health_results[endpoint_id] = {
                        "name": endpoint.name,
                        "healthy": response.success,
                        "status_code": response.status_code,
                        "latency_ms": response.latency_ms,
                        "error": response.error
                    }
                except Exception as e:
                    health_results[endpoint_id] = {
                        "name": endpoint.name,
                        "healthy": False,
                        "error": str(e)
                    }
            else:
                health_results[endpoint_id] = {
                    "name": endpoint.name,
                    "healthy": False,
                    "error": "Endpoint disabled"
                }
        
        return health_results
    
    async def shutdown(self) -> None:
        """Shutdown the API gateway"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            logger.info("API Gateway shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during API Gateway shutdown: {e}")


# Global API gateway instance
api_gateway = APIGateway()
