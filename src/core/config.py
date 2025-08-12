"""
Enterprise Configuration Management
===================================

Type-safe configuration with validation, environment support,
and hot-reload capabilities.
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, get_type_hints
from dataclasses import dataclass, field, fields
from enum import Enum
import logging
from functools import wraps

# Try to import yaml, fall back to basic functionality if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .exceptions import ConfigurationException, ErrorSeverity
from .logging_config import get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "quantum_consensus"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    pool_timeout: int = 30
    ssl_mode: str = "prefer"
    connection_timeout: int = 5


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: int = 5
    connection_pool_size: int = 10


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "localhost"
    port: int = 5001
    workers: int = 4
    worker_timeout: int = 30
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    enable_swagger: bool = True


@dataclass  
class UIConfig:
    """UI server configuration"""
    host: str = "localhost"
    port: int = 8503
    theme: str = "light"
    enable_caching: bool = True
    cache_ttl: int = 300
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    session_timeout: int = 3600


@dataclass
class MLXConfig:
    """MLX AI configuration"""
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    seed: Optional[int] = None
    cache_dir: str = "models/cache"
    max_memory_gb: float = 8.0
    inference_timeout: int = 30


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    performance_sampling_rate: float = 0.1
    enable_distributed_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    prometheus_enabled: bool = False
    prometheus_endpoint: str = "/metrics"


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_authentication: bool = False
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_expiration_hours: int = 24
    enable_rate_limiting: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_encryption_at_rest: bool = False
    encryption_key: Optional[str] = None


@dataclass
class TradingConfig:
    """Trading system configuration"""
    enable_live_trading: bool = False
    enable_paper_trading: bool = True
    default_position_size: float = 1000.0
    max_position_size: float = 10000.0
    risk_per_trade: float = 0.02
    max_daily_trades: int = 10
    market_data_sources: list = field(default_factory=lambda: ["yahoo", "alpaca"])
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    timezone: str = "America/New_York"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "json"
    directory: str = "logs"
    max_file_size_mb: int = 100
    backup_count: int = 10
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_error_file: bool = True
    enable_performance_log: bool = True


@dataclass
class ResilienceConfig:
    """Resilience patterns configuration"""
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    timeout_default: int = 30
    bulkhead_max_concurrent: int = 10
    rate_limit_per_minute: int = 60


@dataclass
class RiverTradingConfig:
    """Main River Trading System configuration"""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    version: str = "1.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    mlx: MLXConfig = field(default_factory=MLXConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    
    # Additional settings
    startup_timeout: int = 60
    shutdown_timeout: int = 30
    enable_hot_reload: bool = True
    config_file_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self):
        """Validate configuration values"""
        errors = []
        
        # Validate ports
        if not (1024 <= self.api.port <= 65535):
            errors.append("API port must be between 1024 and 65535")
        
        if not (1024 <= self.ui.port <= 65535):
            errors.append("UI port must be between 1024 and 65535")
        
        if self.api.port == self.ui.port:
            errors.append("API and UI ports cannot be the same")
        
        # Validate MLX configuration
        if self.mlx.max_tokens < 1:
            errors.append("MLX max_tokens must be positive")
        
        if not (0.0 <= self.mlx.temperature <= 2.0):
            errors.append("MLX temperature must be between 0.0 and 2.0")
        
        # Validate trading configuration
        if self.trading.risk_per_trade <= 0 or self.trading.risk_per_trade > 1:
            errors.append("Risk per trade must be between 0 and 1")
        
        if self.trading.max_position_size <= 0:
            errors.append("Max position size must be positive")
        
        # Validate security configuration
        if self.environment == Environment.PRODUCTION:
            if self.security.jwt_secret_key == "your-secret-key-change-in-production":
                errors.append("JWT secret key must be changed in production")
            
            if self.debug:
                errors.append("Debug mode should be disabled in production")
        
        if errors:
            raise ConfigurationException(
                f"Configuration validation failed: {'; '.join(errors)}",
                severity=ErrorSeverity.CRITICAL,
                context={"validation_errors": errors}
            )
        
        logger.info(
            "Configuration validation passed",
            extra={"config_validation": {"environment": self.environment.value}}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            
            if hasattr(value, '__dict__'):
                # Nested dataclass
                result[field_info.name] = {
                    f.name: getattr(value, f.name)
                    for f in fields(value)
                }
            elif isinstance(value, Enum):
                result[field_info.name] = value.value
            else:
                result[field_info.name] = value
        
        return result
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        password_part = f":{self.database.password}" if self.database.password else ""
        return (
            f"postgresql://{self.database.username}{password_part}@"
            f"{self.database.host}:{self.database.port}/{self.database.name}"
        )
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        password_part = f":{self.redis.password}@" if self.redis.password else ""
        protocol = "rediss" if self.redis.ssl else "redis"
        return f"{protocol}://{password_part}{self.redis.host}:{self.redis.port}/{self.redis.db}"


class ConfigManager:
    """
    Configuration manager with environment support and hot-reload.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[RiverTradingConfig] = None
        self.watchers = []
        self._lock = threading.RLock()
        
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None
    ) -> RiverTradingConfig:
        """
        Load configuration from file and environment variables.
        
        Args:
            config_path: Path to configuration file
            environment: Override environment
            
        Returns:
            Loaded configuration
        """
        with self._lock:
            # Determine configuration file path
            if config_path:
                self.config_path = Path(config_path)
            elif not self.config_path:
                # Try common configuration file locations
                for possible_path in [
                    "config/config.yaml",
                    "config/config.json", 
                    "config.yaml",
                    "config.json"
                ]:
                    if Path(possible_path).exists():
                        self.config_path = Path(possible_path)
                        break
            
            # Load base configuration
            if self.config_path and self.config_path.exists():
                config_data = self._load_config_file(self.config_path)
                logger.info(
                    f"Configuration loaded from file: {self.config_path}",
                    extra={"config_file": str(self.config_path)}
                )
            else:
                config_data = {}
                logger.info("Using default configuration (no config file found)")
            
            # Override with environment variables
            env_overrides = self._load_environment_variables()
            config_data = self._merge_config(config_data, env_overrides)
            
            # Override environment if specified
            if environment:
                config_data["environment"] = environment
            
            # Create configuration object
            self.config = self._create_config_object(config_data)
            
            logger.info(
                "Configuration loaded successfully",
                extra={
                    "environment": self.config.environment.value,
                    "config_file": str(self.config_path) if self.config_path else None
                }
            )
            
            return self.config
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        raise ConfigurationException(
                            "YAML support not available. Install PyYAML: pip install PyYAML",
                            config_key="yaml_dependency"
                        )
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigurationException(
                        f"Unsupported configuration file format: {config_path.suffix}",
                        config_key="config_file_format"
                    )
        except Exception as exc:
            raise ConfigurationException(
                f"Failed to load configuration file {config_path}: {exc}",
                config_key="config_file_path",
                context={"file_path": str(config_path), "error": str(exc)},
                cause=exc
            )
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Environment variable mappings
        env_mappings = {
            "RIVER_ENVIRONMENT": "environment",
            "RIVER_DEBUG": ("debug", bool),
            "RIVER_API_HOST": "api.host",
            "RIVER_API_PORT": ("api.port", int),
            "RIVER_UI_HOST": "ui.host", 
            "RIVER_UI_PORT": ("ui.port", int),
            "RIVER_DATABASE_HOST": "database.host",
            "RIVER_DATABASE_PORT": ("database.port", int),
            "RIVER_DATABASE_NAME": "database.name",
            "RIVER_DATABASE_USERNAME": "database.username",
            "RIVER_DATABASE_PASSWORD": "database.password",
            "RIVER_REDIS_HOST": "redis.host",
            "RIVER_REDIS_PORT": ("redis.port", int),
            "RIVER_MLX_MODEL": "mlx.model_name",
            "RIVER_MLX_MAX_TOKENS": ("mlx.max_tokens", int),
            "RIVER_MLX_TEMPERATURE": ("mlx.temperature", float),
            "RIVER_LOG_LEVEL": "logging.level",
            "RIVER_LOG_DIR": "logging.directory",
            "RIVER_JWT_SECRET": "security.jwt_secret_key",
            "RIVER_ENABLE_LIVE_TRADING": ("trading.enable_live_trading", bool),
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Handle type conversion
                if isinstance(config_key, tuple):
                    config_key, value_type = config_key
                    try:
                        if value_type == bool:
                            env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            env_value = value_type(env_value)
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            f"Invalid environment variable value: {env_var}={env_value}",
                            extra={"env_var": env_var, "value": env_value, "expected_type": value_type.__name__}
                        )
                        continue
                
                # Set nested configuration value
                self._set_nested_value(env_config, config_key, env_value)
        
        if env_config:
            logger.info(
                "Environment variables loaded",
                extra={"env_overrides": list(env_config.keys())}
            )
        
        return env_config
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> RiverTradingConfig:
        """Create configuration object from dictionary"""
        try:
            # Handle environment enum
            if "environment" in config_data:
                env_value = config_data["environment"]
                if isinstance(env_value, str):
                    config_data["environment"] = Environment(env_value.lower())
            
            # Handle logging level enum
            if "logging" in config_data and "level" in config_data["logging"]:
                level_value = config_data["logging"]["level"]
                if isinstance(level_value, str):
                    config_data["logging"]["level"] = LogLevel(level_value.upper())
            
            # Create nested configuration objects
            config_kwargs = {}
            
            for field_info in fields(RiverTradingConfig):
                if field_info.name in config_data:
                    field_value = config_data[field_info.name]
                    
                    # Handle nested dataclass fields
                    if hasattr(field_info.type, '__dataclass_fields__'):
                        if isinstance(field_value, dict):
                            config_kwargs[field_info.name] = field_info.type(**field_value)
                        else:
                            config_kwargs[field_info.name] = field_value
                    else:
                        config_kwargs[field_info.name] = field_value
            
            return RiverTradingConfig(**config_kwargs)
            
        except Exception as exc:
            raise ConfigurationException(
                f"Failed to create configuration object: {exc}",
                context={"config_data": config_data},
                cause=exc
            )
    
    def reload_config(self) -> RiverTradingConfig:
        """Reload configuration from file"""
        logger.info("Reloading configuration")
        return self.load_config()
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file"""
        if not self.config:
            raise ConfigurationException("No configuration loaded to save")
        
        save_path = Path(config_path) if config_path else self.config_path
        if not save_path:
            raise ConfigurationException("No configuration file path specified")
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self.config.to_dict()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        raise ConfigurationException(
                            "YAML support not available. Install PyYAML: pip install PyYAML",
                            config_key="yaml_dependency"
                        )
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, default=str)
                
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as exc:
            raise ConfigurationException(
                f"Failed to save configuration to {save_path}: {exc}",
                context={"file_path": str(save_path)},
                cause=exc
            )


# Global configuration manager
config_manager = ConfigManager()


def requires_config(func):
    """Decorator to ensure configuration is loaded"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if config_manager.config is None:
            raise ConfigurationException(
                "Configuration not loaded. Call config_manager.load_config() first."
            )
        return func(*args, **kwargs)
    return wrapper


def get_config() -> RiverTradingConfig:
    """Get current configuration"""
    if config_manager.config is None:
        # Try to load default configuration
        config_manager.load_config()
    return config_manager.config


# Configuration validation decorators
def validate_environment(allowed_environments: list):
    """Decorator to validate current environment"""
    def decorator(func):
        @wraps(func)
        @requires_config
        def wrapper(*args, **kwargs):
            current_env = config_manager.config.environment
            if current_env not in allowed_environments:
                raise ConfigurationException(
                    f"Function {func.__name__} not allowed in {current_env.value} environment",
                    context={
                        "function": func.__name__,
                        "current_environment": current_env.value,
                        "allowed_environments": [env.value for env in allowed_environments]
                    }
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator