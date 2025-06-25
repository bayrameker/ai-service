"""
Third-Party API Adapters for Popular Services
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class AdapterType(str, Enum):
    """Third-party adapter types"""
    SOCIAL_MEDIA = "social_media"
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    ANALYTICS = "analytics"
    PAYMENT = "payment"
    WEATHER = "weather"
    NEWS = "news"
    TRANSLATION = "translation"
    EMAIL = "email"
    CUSTOM = "custom"


class AuthMethod(str, Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    CUSTOM_HEADER = "custom_header"


class AdapterConfig(BaseModel):
    """Third-party adapter configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    adapter_type: AdapterType
    base_url: str
    auth_method: AuthMethod
    auth_config: Dict[str, Any] = Field(default_factory=dict)
    rate_limit: Optional[int] = None  # requests per minute
    timeout: int = 30
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdapterRequest(BaseModel):
    """Adapter request model"""
    adapter_id: str
    endpoint: str
    method: str = "GET"
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None


class AdapterResponse(BaseModel):
    """Adapter response model"""
    success: bool
    data: Optional[Any] = None
    status_code: int
    error: Optional[str] = None
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseAdapter:
    """Base class for third-party adapters"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        self.rate_limiter = None
    
    async def initialize(self) -> None:
        """Initialize adapter"""
        self.http_client = httpx.AsyncClient(timeout=self.config.timeout)
        
        if self.config.rate_limit:
            from app.integrations.api_gateway import RateLimiter
            self.rate_limiter = RateLimiter(self.config.rate_limit)
    
    async def shutdown(self) -> None:
        """Shutdown adapter"""
        if self.http_client:
            await self.http_client.aclose()
    
    def _prepare_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        headers = {"User-Agent": "AI-Service/1.0"}
        
        if additional_headers:
            headers.update(additional_headers)
        
        # Add authentication
        if self.config.auth_method == AuthMethod.API_KEY:
            api_key = self.config.auth_config.get("api_key")
            header_name = self.config.auth_config.get("header_name", "X-API-Key")
            if api_key:
                headers[header_name] = api_key
        
        elif self.config.auth_method == AuthMethod.BEARER_TOKEN:
            token = self.config.auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif self.config.auth_method == AuthMethod.CUSTOM_HEADER:
            custom_headers = self.config.auth_config.get("headers", {})
            headers.update(custom_headers)
        
        return headers
    
    async def make_request(self, request: AdapterRequest) -> AdapterResponse:
        """Make HTTP request to third-party API"""
        start_time = datetime.utcnow()
        
        try:
            # Check rate limit
            if self.rate_limiter and not self.rate_limiter.can_make_request():
                return AdapterResponse(
                    success=False,
                    status_code=429,
                    error="Rate limit exceeded",
                    execution_time_ms=0
                )
            
            # Prepare request
            url = f"{self.config.base_url.rstrip('/')}/{request.endpoint.lstrip('/')}"
            headers = self._prepare_headers(request.headers)
            
            # Make request
            response = await self.http_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.params,
                json=request.data
            )
            
            # Record rate limit usage
            if self.rate_limiter:
                self.rate_limiter.record_request()
            
            # Parse response
            try:
                response_data = response.json() if response.content else None
            except json.JSONDecodeError:
                response_data = response.text if response.content else None
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return AdapterResponse(
                success=response.is_success,
                data=response_data,
                status_code=response.status_code,
                error=None if response.is_success else f"HTTP {response.status_code}",
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Adapter request failed: {e}")
            
            return AdapterResponse(
                success=False,
                data=None,
                status_code=500,
                error=str(e),
                execution_time_ms=execution_time
            )


class SlackAdapter(BaseAdapter):
    """Slack API adapter"""
    
    async def send_message(self, channel: str, text: str, **kwargs) -> AdapterResponse:
        """Send message to Slack channel"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="chat.postMessage",
            method="POST",
            data={
                "channel": channel,
                "text": text,
                **kwargs
            }
        )
        return await self.make_request(request)
    
    async def get_channels(self) -> AdapterResponse:
        """Get list of channels"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="conversations.list",
            method="GET"
        )
        return await self.make_request(request)


class DiscordAdapter(BaseAdapter):
    """Discord API adapter"""
    
    async def send_message(self, channel_id: str, content: str, **kwargs) -> AdapterResponse:
        """Send message to Discord channel"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint=f"channels/{channel_id}/messages",
            method="POST",
            data={
                "content": content,
                **kwargs
            }
        )
        return await self.make_request(request)
    
    async def get_guild_channels(self, guild_id: str) -> AdapterResponse:
        """Get guild channels"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint=f"guilds/{guild_id}/channels",
            method="GET"
        )
        return await self.make_request(request)


class TwitterAdapter(BaseAdapter):
    """Twitter API adapter"""
    
    async def post_tweet(self, text: str, **kwargs) -> AdapterResponse:
        """Post a tweet"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="2/tweets",
            method="POST",
            data={
                "text": text,
                **kwargs
            }
        )
        return await self.make_request(request)
    
    async def search_tweets(self, query: str, max_results: int = 10) -> AdapterResponse:
        """Search tweets"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="2/tweets/search/recent",
            method="GET",
            params={
                "query": query,
                "max_results": str(max_results)
            }
        )
        return await self.make_request(request)


class WeatherAdapter(BaseAdapter):
    """Weather API adapter (OpenWeatherMap)"""
    
    async def get_current_weather(self, city: str, units: str = "metric") -> AdapterResponse:
        """Get current weather for city"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="weather",
            method="GET",
            params={
                "q": city,
                "units": units,
                "appid": self.config.auth_config.get("api_key")
            }
        )
        return await self.make_request(request)
    
    async def get_forecast(self, city: str, days: int = 5, units: str = "metric") -> AdapterResponse:
        """Get weather forecast"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="forecast",
            method="GET",
            params={
                "q": city,
                "cnt": str(days * 8),  # 8 forecasts per day (3-hour intervals)
                "units": units,
                "appid": self.config.auth_config.get("api_key")
            }
        )
        return await self.make_request(request)


class NewsAdapter(BaseAdapter):
    """News API adapter"""
    
    async def get_top_headlines(self, country: str = "us", category: Optional[str] = None) -> AdapterResponse:
        """Get top headlines"""
        params = {
            "country": country,
            "apiKey": self.config.auth_config.get("api_key")
        }
        
        if category:
            params["category"] = category
        
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="top-headlines",
            method="GET",
            params=params
        )
        return await self.make_request(request)
    
    async def search_news(self, query: str, language: str = "en", sort_by: str = "publishedAt") -> AdapterResponse:
        """Search news articles"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="everything",
            method="GET",
            params={
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "apiKey": self.config.auth_config.get("api_key")
            }
        )
        return await self.make_request(request)


class TranslationAdapter(BaseAdapter):
    """Translation API adapter (Google Translate)"""
    
    async def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> AdapterResponse:
        """Translate text"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="translate",
            method="POST",
            data={
                "q": text,
                "target": target_language,
                "source": source_language,
                "key": self.config.auth_config.get("api_key")
            }
        )
        return await self.make_request(request)
    
    async def detect_language(self, text: str) -> AdapterResponse:
        """Detect language of text"""
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="detect",
            method="POST",
            data={
                "q": text,
                "key": self.config.auth_config.get("api_key")
            }
        )
        return await self.make_request(request)


class EmailAdapter(BaseAdapter):
    """Email API adapter (SendGrid)"""
    
    async def send_email(self, to_email: str, subject: str, content: str, from_email: Optional[str] = None) -> AdapterResponse:
        """Send email"""
        from_email = from_email or self.config.auth_config.get("from_email", "noreply@ai-service.com")
        
        request = AdapterRequest(
            adapter_id=self.config.id,
            endpoint="mail/send",
            method="POST",
            data={
                "personalizations": [
                    {
                        "to": [{"email": to_email}],
                        "subject": subject
                    }
                ],
                "from": {"email": from_email},
                "content": [
                    {
                        "type": "text/plain",
                        "value": content
                    }
                ]
            }
        )
        return await self.make_request(request)


class ThirdPartyAdapterManager:
    """Manager for third-party API adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, BaseAdapter] = {}
        self.configs: Dict[str, AdapterConfig] = {}
        
        # Predefined adapter classes
        self.adapter_classes = {
            "slack": SlackAdapter,
            "discord": DiscordAdapter,
            "twitter": TwitterAdapter,
            "weather": WeatherAdapter,
            "news": NewsAdapter,
            "translation": TranslationAdapter,
            "email": EmailAdapter
        }
    
    async def register_adapter(self, config: AdapterConfig, adapter_class_name: Optional[str] = None) -> str:
        """Register a third-party adapter"""
        try:
            # Determine adapter class
            if adapter_class_name and adapter_class_name in self.adapter_classes:
                adapter_class = self.adapter_classes[adapter_class_name]
            else:
                adapter_class = BaseAdapter
            
            # Create adapter instance
            adapter = adapter_class(config)
            await adapter.initialize()
            
            self.adapters[config.id] = adapter
            self.configs[config.id] = config
            
            logger.info(f"Registered adapter: {config.name} ({config.adapter_type})")
            return config.id
            
        except Exception as e:
            logger.error(f"Failed to register adapter: {e}")
            raise
    
    async def make_request(self, request: AdapterRequest) -> AdapterResponse:
        """Make request through adapter"""
        try:
            adapter = self.adapters.get(request.adapter_id)
            if not adapter:
                return AdapterResponse(
                    success=False,
                    status_code=404,
                    error=f"Adapter {request.adapter_id} not found",
                    execution_time_ms=0
                )
            
            return await adapter.make_request(request)
            
        except Exception as e:
            logger.error(f"Failed to make adapter request: {e}")
            return AdapterResponse(
                success=False,
                status_code=500,
                error=str(e),
                execution_time_ms=0
            )
    
    def get_adapter(self, adapter_id: str) -> Optional[BaseAdapter]:
        """Get adapter by ID"""
        return self.adapters.get(adapter_id)
    
    def list_adapters(self, adapter_type: Optional[AdapterType] = None) -> List[AdapterConfig]:
        """List adapters with optional filtering"""
        configs = list(self.configs.values())
        
        if adapter_type:
            configs = [config for config in configs if config.adapter_type == adapter_type]
        
        return configs
    
    def remove_adapter(self, adapter_id: str) -> bool:
        """Remove adapter"""
        try:
            if adapter_id in self.adapters:
                adapter = self.adapters[adapter_id]
                asyncio.create_task(adapter.shutdown())
                del self.adapters[adapter_id]
                del self.configs[adapter_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove adapter: {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all adapters"""
        results = {}
        
        for adapter_id, adapter in self.adapters.items():
            try:
                # Simple health check - make a basic request
                test_request = AdapterRequest(
                    adapter_id=adapter_id,
                    endpoint="",
                    method="GET"
                )
                response = await adapter.make_request(test_request)
                results[adapter_id] = response.status_code != 500
            except Exception:
                results[adapter_id] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        total_adapters = len(self.adapters)
        enabled_adapters = len([c for c in self.configs.values() if c.enabled])
        
        # Count by type
        type_counts = {}
        for adapter_type in AdapterType:
            type_counts[adapter_type.value] = len([
                c for c in self.configs.values() if c.adapter_type == adapter_type
            ])
        
        return {
            "total_adapters": total_adapters,
            "enabled_adapters": enabled_adapters,
            "type_distribution": type_counts,
            "available_types": list(self.adapter_classes.keys())
        }
    
    async def shutdown(self) -> None:
        """Shutdown all adapters"""
        try:
            for adapter in self.adapters.values():
                await adapter.shutdown()
            
            logger.info("All third-party adapters shutdown")
            
        except Exception as e:
            logger.error(f"Error during adapters shutdown: {e}")


# Global third-party adapter manager instance
third_party_adapter_manager = ThirdPartyAdapterManager()
