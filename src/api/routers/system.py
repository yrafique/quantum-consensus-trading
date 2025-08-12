"""
System API Router
================

System-level endpoints for health checks, metrics, and administration.
"""

from typing import Dict, Any, List
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel
import structlog

# Import fix for standalone execution
try:
    from ...core.config import get_config
    from ...core.monitoring import MetricsCollector
except ImportError:
    # Fallback imports for development
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from core.config import get_config
    from core.monitoring import MetricsCollector

# Create simple models for system info
class ComponentStatus(BaseModel):
    status: str

from ..dependencies import verify_admin_access

logger = structlog.get_logger(__name__)

router = APIRouter()


class SystemStatus(BaseModel):
    """System status response model"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float
    components: Dict[str, ComponentStatus]


class SystemInfo(BaseModel):
    """System information response model"""
    platform: str
    python_version: str
    cpu_count: int
    memory_total_gb: float
    disk_usage_gb: Dict[str, float]
    network_interfaces: List[str]


@router.get("/info", response_model=SystemInfo)
async def get_system_info() -> SystemInfo:
    """
    Get comprehensive system information.
    
    Returns:
        Detailed system information including hardware specs
    """
    try:
        import platform
        import psutil
        import socket
        
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get network interfaces
        interfaces = []
        for interface, addrs in psutil.net_if_addrs().items():
            if any(addr.family == socket.AF_INET for addr in addrs):
                interfaces.append(interface)
        
        return SystemInfo(
            platform=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            cpu_count=psutil.cpu_count(),
            memory_total_gb=round(memory.total / (1024**3), 2),
            disk_usage_gb={
                "total": round(disk.total / (1024**3), 2),
                "used": round(disk.used / (1024**3), 2),
                "free": round(disk.free / (1024**3), 2)
            },
            network_interfaces=interfaces
        )
        
    except Exception as exc:
        logger.error("Failed to get system info", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system information"
        )


@router.get("/logs")
async def get_recent_logs(
    lines: int = 100,
    level: str = "INFO",
    admin: bool = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Get recent log entries.
    
    Args:
        lines: Number of log lines to retrieve
        level: Minimum log level to include
        admin: Admin access verification
        
    Returns:
        Recent log entries
    """
    try:
        config = get_config()
        log_file = f"{config.logging.directory}/app.log"
        
        # Read recent log lines
        import subprocess
        result = subprocess.run(
            ["tail", "-n", str(lines), log_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Log file not found"
            )
        
        log_lines = result.stdout.strip().split('\n')
        
        return {
            "total_lines": len(log_lines),
            "level_filter": level,
            "logs": log_lines[-lines:] if log_lines else []
        }
        
    except Exception as exc:
        logger.error("Failed to get logs", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve logs"
        )


@router.post("/cache/clear")
async def clear_cache(
    background_tasks: BackgroundTasks,
    admin: bool = Depends(verify_admin_access)
) -> Dict[str, str]:
    """
    Clear system caches.
    
    Args:
        background_tasks: FastAPI background tasks
        admin: Admin access verification
        
    Returns:
        Cache clearing status
    """
    try:
        async def clear_caches():
            # Clear Redis caches if available
            try:
                import redis
                config = get_config()
                r = redis.Redis(
                    host=config.redis.host,
                    port=config.redis.port,
                    db=config.redis.db,
                    decode_responses=True
                )
                await r.flushdb()
                logger.info("Redis cache cleared")
            except Exception as exc:
                logger.warning("Failed to clear Redis cache", error=str(exc))
            
            # Clear MLX model cache if available
            try:
                import shutil
                config = get_config()
                cache_dir = config.mlx.cache_dir
                if cache_dir and os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.info("MLX model cache cleared")
            except Exception as exc:
                logger.warning("Failed to clear MLX cache", error=str(exc))
        
        # Run cache clearing in background
        background_tasks.add_task(clear_caches)
        
        return {
            "status": "success",
            "message": "Cache clearing initiated in background"
        }
        
    except Exception as exc:
        logger.error("Failed to clear cache", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/config")
async def get_configuration(
    admin: bool = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Get current system configuration (sanitized).
    
    Args:
        admin: Admin access verification
        
    Returns:
        Sanitized system configuration
    """
    try:
        config = get_config()
        config_dict = config.to_dict()
        
        # Sanitize sensitive information
        sensitive_keys = [
            "password", "secret", "key", "token", "credential"
        ]
        
        def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            sanitized = {}
            for key, value in d.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    sanitized[key] = sanitize_dict(value)
                else:
                    sanitized[key] = value
            return sanitized
        
        return sanitize_dict(config_dict)
        
    except Exception as exc:
        logger.error("Failed to get configuration", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )