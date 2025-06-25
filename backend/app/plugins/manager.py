"""
Plugin Architecture and Management System
"""

import asyncio
import logging
import importlib
import inspect
import sys
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path
import uuid
import json

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Plugin types"""
    AGENT_CAPABILITY = "agent_capability"
    LLM_PROVIDER = "llm_provider"
    MEMORY_BACKEND = "memory_backend"
    INTEGRATION = "integration"
    MIDDLEWARE = "middleware"
    TOOL = "tool"
    CUSTOM = "custom"


class PluginStatus(str, Enum):
    """Plugin status"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


class HookType(str, Enum):
    """Plugin hook types"""
    BEFORE_REQUEST = "before_request"
    AFTER_REQUEST = "after_request"
    BEFORE_AGENT_CREATE = "before_agent_create"
    AFTER_AGENT_CREATE = "after_agent_create"
    BEFORE_TASK_EXECUTE = "before_task_execute"
    AFTER_TASK_EXECUTE = "after_task_execute"
    BEFORE_MEMORY_STORE = "before_memory_store"
    AFTER_MEMORY_STORE = "after_memory_store"
    CUSTOM = "custom"


class PluginMetadata(BaseModel):
    """Plugin metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = Field(default_factory=list)
    min_system_version: str = "1.0.0"
    max_system_version: Optional[str] = None
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    permissions: List[str] = Field(default_factory=list)
    hooks: List[HookType] = Field(default_factory=list)
    entry_point: str
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PluginConfig(BaseModel):
    """Plugin configuration"""
    plugin_id: str
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    auto_start: bool = True


class PluginInstance(BaseModel):
    """Plugin instance information"""
    metadata: PluginMetadata
    config: PluginConfig
    status: PluginStatus = PluginStatus.LOADED
    instance: Optional[Any] = None
    loaded_at: datetime = Field(default_factory=datetime.utcnow)
    last_error: Optional[str] = None
    execution_count: int = 0
    error_count: int = 0


class BasePlugin:
    """Base class for all plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check plugin health"""
        return True


class PluginHook:
    """Plugin hook for extending system functionality"""
    
    def __init__(self, hook_type: HookType, priority: int = 0):
        self.hook_type = hook_type
        self.priority = priority
        self.handlers: List[Callable] = []
    
    def register(self, handler: Callable) -> None:
        """Register a hook handler"""
        self.handlers.append(handler)
        # Sort by priority (higher priority first)
        self.handlers.sort(key=lambda h: getattr(h, 'priority', 0), reverse=True)
    
    def unregister(self, handler: Callable) -> None:
        """Unregister a hook handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute all hook handlers"""
        results = []
        
        for handler in self.handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook handler {handler.__name__} failed: {e}")
        
        return results


class PluginManager:
    """Plugin management system"""
    
    def __init__(self):
        self.plugins: Dict[str, PluginInstance] = {}
        self.hooks: Dict[HookType, PluginHook] = {}
        self.plugin_directories: List[Path] = []
        
        # Initialize hooks
        for hook_type in HookType:
            self.hooks[hook_type] = PluginHook(hook_type)
        
        # Plugin discovery
        self.plugin_directories = [
            Path("app/plugins/builtin"),
            Path("app/plugins/external"),
            Path("plugins")  # User plugins directory
        ]
    
    async def initialize(self) -> None:
        """Initialize plugin manager"""
        try:
            # Create plugin directories if they don't exist
            for directory in self.plugin_directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Discover and load plugins
            await self.discover_plugins()
            await self.load_enabled_plugins()
            
            logger.info("Plugin manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin manager: {e}")
            raise
    
    async def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins"""
        discovered_plugins = []
        
        for directory in self.plugin_directories:
            if not directory.exists():
                continue
            
            # Look for plugin.json files
            for plugin_file in directory.rglob("plugin.json"):
                try:
                    with open(plugin_file, 'r') as f:
                        plugin_data = json.load(f)
                    
                    metadata = PluginMetadata(**plugin_data)
                    discovered_plugins.append(metadata)
                    
                    logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")
                    
                except Exception as e:
                    logger.error(f"Failed to load plugin metadata from {plugin_file}: {e}")
        
        return discovered_plugins
    
    async def load_plugin(self, metadata: PluginMetadata, config: Optional[PluginConfig] = None) -> bool:
        """Load a specific plugin"""
        try:
            if metadata.id in self.plugins:
                logger.warning(f"Plugin {metadata.name} is already loaded")
                return False
            
            # Create default config if not provided
            if not config:
                config = PluginConfig(plugin_id=metadata.id)
            
            # Check dependencies
            if not await self._check_dependencies(metadata):
                logger.error(f"Plugin {metadata.name} has unmet dependencies")
                return False
            
            # Load plugin module
            plugin_instance = await self._load_plugin_module(metadata, config)
            if not plugin_instance:
                return False
            
            # Initialize plugin
            if not await plugin_instance.initialize():
                logger.error(f"Failed to initialize plugin {metadata.name}")
                return False
            
            # Create plugin instance record
            instance = PluginInstance(
                metadata=metadata,
                config=config,
                status=PluginStatus.ACTIVE,
                instance=plugin_instance
            )
            
            self.plugins[metadata.id] = instance
            
            # Register plugin hooks
            await self._register_plugin_hooks(plugin_instance, metadata)
            
            logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {metadata.name}: {e}")
            
            # Update plugin status to error
            if metadata.id in self.plugins:
                self.plugins[metadata.id].status = PluginStatus.ERROR
                self.plugins[metadata.id].last_error = str(e)
            
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        try:
            if plugin_id not in self.plugins:
                return False
            
            instance = self.plugins[plugin_id]
            
            # Shutdown plugin
            if instance.instance:
                await instance.instance.shutdown()
            
            # Unregister hooks
            await self._unregister_plugin_hooks(instance.instance, instance.metadata)
            
            # Remove from registry
            del self.plugins[plugin_id]
            
            logger.info(f"Unloaded plugin: {instance.metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False
    
    async def reload_plugin(self, plugin_id: str) -> bool:
        """Reload a plugin"""
        try:
            if plugin_id not in self.plugins:
                return False
            
            instance = self.plugins[plugin_id]
            metadata = instance.metadata
            config = instance.config
            
            # Unload plugin
            await self.unload_plugin(plugin_id)
            
            # Reload plugin
            return await self.load_plugin(metadata, config)
            
        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_id}: {e}")
            return False
    
    async def load_enabled_plugins(self) -> None:
        """Load all enabled plugins"""
        discovered = await self.discover_plugins()
        
        for metadata in discovered:
            if metadata.enabled:
                await self.load_plugin(metadata)
    
    async def _load_plugin_module(self, metadata: PluginMetadata, config: PluginConfig) -> Optional[BasePlugin]:
        """Load plugin module and create instance"""
        try:
            # Import plugin module
            module_path = metadata.entry_point
            module = importlib.import_module(module_path)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in {module_path}")
                return None
            
            # Create plugin instance
            plugin_instance = plugin_class(config.config)
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin module {metadata.entry_point}: {e}")
            return None
    
    async def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are met"""
        for dependency in metadata.dependencies:
            # Check if dependency plugin is loaded
            dependency_loaded = any(
                p.metadata.name == dependency and p.status == PluginStatus.ACTIVE
                for p in self.plugins.values()
            )
            
            if not dependency_loaded:
                logger.error(f"Dependency {dependency} not found for plugin {metadata.name}")
                return False
        
        return True
    
    async def _register_plugin_hooks(self, plugin_instance: BasePlugin, metadata: PluginMetadata) -> None:
        """Register plugin hooks"""
        try:
            for hook_type in metadata.hooks:
                # Look for hook handler methods
                handler_method_name = f"on_{hook_type.value}"
                if hasattr(plugin_instance, handler_method_name):
                    handler = getattr(plugin_instance, handler_method_name)
                    self.hooks[hook_type].register(handler)
                    logger.debug(f"Registered hook {hook_type} for plugin {metadata.name}")
        
        except Exception as e:
            logger.error(f"Failed to register hooks for plugin {metadata.name}: {e}")
    
    async def _unregister_plugin_hooks(self, plugin_instance: BasePlugin, metadata: PluginMetadata) -> None:
        """Unregister plugin hooks"""
        try:
            for hook_type in metadata.hooks:
                handler_method_name = f"on_{hook_type.value}"
                if hasattr(plugin_instance, handler_method_name):
                    handler = getattr(plugin_instance, handler_method_name)
                    self.hooks[hook_type].unregister(handler)
                    logger.debug(f"Unregistered hook {hook_type} for plugin {metadata.name}")
        
        except Exception as e:
            logger.error(f"Failed to unregister hooks for plugin {metadata.name}: {e}")
    
    async def execute_hook(self, hook_type: HookType, *args, **kwargs) -> Any:
        """Execute plugin hook"""
        try:
            if hook_type in self.hooks:
                return await self.hooks[hook_type].execute(*args, **kwargs)
            return []
        except Exception as e:
            logger.error(f"Failed to execute hook {hook_type}: {e}")
            return []
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInstance]:
        """Get plugin instance"""
        return self.plugins.get(plugin_id)
    
    def list_plugins(self, status: Optional[PluginStatus] = None, plugin_type: Optional[PluginType] = None) -> List[PluginInstance]:
        """List plugins with optional filtering"""
        plugins = list(self.plugins.values())
        
        if status:
            plugins = [p for p in plugins if p.status == status]
        
        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]
        
        return plugins
    
    async def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin"""
        try:
            if plugin_id in self.plugins:
                instance = self.plugins[plugin_id]
                instance.config.enabled = True
                instance.status = PluginStatus.ACTIVE
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to enable plugin {plugin_id}: {e}")
            return False
    
    async def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin"""
        try:
            if plugin_id in self.plugins:
                instance = self.plugins[plugin_id]
                instance.config.enabled = False
                instance.status = PluginStatus.DISABLED
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to disable plugin {plugin_id}: {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all plugins"""
        results = {}
        
        for plugin_id, instance in self.plugins.items():
            try:
                if instance.instance and instance.status == PluginStatus.ACTIVE:
                    results[plugin_id] = await instance.instance.health_check()
                else:
                    results[plugin_id] = False
            except Exception as e:
                logger.error(f"Health check failed for plugin {plugin_id}: {e}")
                results[plugin_id] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        total_plugins = len(self.plugins)
        active_plugins = len([p for p in self.plugins.values() if p.status == PluginStatus.ACTIVE])
        
        # Count by type
        type_counts = {}
        for plugin_type in PluginType:
            type_counts[plugin_type.value] = len([
                p for p in self.plugins.values() if p.metadata.plugin_type == plugin_type
            ])
        
        # Count by status
        status_counts = {}
        for status in PluginStatus:
            status_counts[status.value] = len([
                p for p in self.plugins.values() if p.status == status
            ])
        
        return {
            "total_plugins": total_plugins,
            "active_plugins": active_plugins,
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "total_hooks": sum(len(hook.handlers) for hook in self.hooks.values()),
            "plugin_directories": [str(d) for d in self.plugin_directories]
        }
    
    async def shutdown(self) -> None:
        """Shutdown plugin manager"""
        try:
            # Shutdown all plugins
            for plugin_id in list(self.plugins.keys()):
                await self.unload_plugin(plugin_id)
            
            logger.info("Plugin manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during plugin manager shutdown: {e}")


# Global plugin manager instance
plugin_manager = PluginManager()
