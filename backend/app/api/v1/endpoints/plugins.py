"""
Plugin Management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional, Dict, Any
import logging

from app.plugins.manager import PluginMetadata, PluginConfig, PluginStatus, PluginType, plugin_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def list_plugins(
    status: Optional[PluginStatus] = None,
    plugin_type: Optional[PluginType] = None
):
    """List plugins with optional filtering"""
    try:
        plugins = plugin_manager.list_plugins(status, plugin_type)
        return [
            {
                "id": plugin.metadata.id,
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "type": plugin.metadata.plugin_type,
                "status": plugin.status,
                "enabled": plugin.config.enabled,
                "loaded_at": plugin.loaded_at,
                "execution_count": plugin.execution_count,
                "error_count": plugin.error_count,
                "last_error": plugin.last_error
            }
            for plugin in plugins
        ]
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str):
    """Get plugin details"""
    try:
        plugin = plugin_manager.get_plugin(plugin_id)
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        
        return {
            "metadata": plugin.metadata.dict(),
            "config": plugin.config.dict(),
            "status": plugin.status,
            "loaded_at": plugin.loaded_at,
            "execution_count": plugin.execution_count,
            "error_count": plugin.error_count,
            "last_error": plugin.last_error
        }
    except Exception as e:
        logger.error(f"Failed to get plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_plugin(metadata: PluginMetadata, config: Optional[PluginConfig] = None):
    """Load a plugin"""
    try:
        success = await plugin_manager.load_plugin(metadata, config)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load plugin")
        return {"status": "loaded", "plugin_id": metadata.id}
    except Exception as e:
        logger.error(f"Failed to load plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_id}/unload")
async def unload_plugin(plugin_id: str):
    """Unload a plugin"""
    try:
        success = await plugin_manager.unload_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"status": "unloaded"}
    except Exception as e:
        logger.error(f"Failed to unload plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_id}/reload")
async def reload_plugin(plugin_id: str):
    """Reload a plugin"""
    try:
        success = await plugin_manager.reload_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"status": "reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_id}/enable")
async def enable_plugin(plugin_id: str):
    """Enable a plugin"""
    try:
        success = await plugin_manager.enable_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"status": "enabled"}
    except Exception as e:
        logger.error(f"Failed to enable plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str):
    """Disable a plugin"""
    try:
        success = await plugin_manager.disable_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"status": "disabled"}
    except Exception as e:
        logger.error(f"Failed to disable plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discover/available")
async def discover_plugins():
    """Discover available plugins"""
    try:
        plugins = await plugin_manager.discover_plugins()
        return [plugin.dict() for plugin in plugins]
    except Exception as e:
        logger.error(f"Failed to discover plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/check")
async def health_check_plugins():
    """Check health of all plugins"""
    try:
        health_results = await plugin_manager.health_check_all()
        return health_results
    except Exception as e:
        logger.error(f"Failed to check plugin health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_plugin_stats():
    """Get plugin system statistics"""
    try:
        return plugin_manager.get_stats()
    except Exception as e:
        logger.error(f"Failed to get plugin stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
