"""Configuration management service for tools, data sources, and agents."""

import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class AuthType(str, Enum):
    """Authentication type enumeration."""
    API_KEY = "api_key"
    OAUTH = "oauth"
    BASIC = "basic"
    NONE = "none"


class ToolType(str, Enum):
    """Tool type enumeration."""
    API = "api"
    DATABASE = "database"
    WEB_SCRAPER = "web_scraper"
    CALCULATOR = "calculator"
    FILE_PROCESSOR = "file_processor"


class SourceType(str, Enum):
    """Data source type enumeration."""
    API = "api"
    DATABASE = "database"
    FILE = "file"
    WEB = "web"
    RSS = "rss"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    auth_type: AuthType
    credentials: Dict[str, Any]
    headers: Optional[Dict[str, str]] = None
    
    def is_valid(self) -> bool:
        """Validate authentication configuration."""
        if self.auth_type == AuthType.API_KEY:
            return "api_key" in self.credentials
        elif self.auth_type == AuthType.OAUTH:
            return all(key in self.credentials for key in ["client_id", "client_secret"])
        elif self.auth_type == AuthType.BASIC:
            return all(key in self.credentials for key in ["username", "password"])
        elif self.auth_type == AuthType.NONE:
            return True
        return False


@dataclass
class RateLimit:
    """Rate limiting configuration."""
    requests_per_minute: int
    requests_per_hour: int
    concurrent_requests: int
    
    def is_valid(self) -> bool:
        """Validate rate limit configuration."""
        return (
            self.requests_per_minute > 0 and
            self.requests_per_hour > 0 and
            self.concurrent_requests > 0 and
            self.requests_per_hour >= self.requests_per_minute
        )


@dataclass
class ToolConfig:
    """Tool configuration."""
    tool_id: str
    name: str
    description: str
    tool_type: ToolType
    agent_types: List[str]
    api_endpoint: Optional[str] = None
    authentication: Optional[AuthConfig] = None
    rate_limits: Optional[RateLimit] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    is_active: bool = True
    last_health_check: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def validate(self) -> List[str]:
        """Validate tool configuration."""
        errors = []
        
        if not self.tool_id or not self.tool_id.strip():
            errors.append("Tool ID is required")
        
        if not self.name or not self.name.strip():
            errors.append("Tool name is required")
        
        if self.tool_type in [ToolType.API, ToolType.WEB_SCRAPER] and not self.api_endpoint:
            errors.append(f"API endpoint is required for {self.tool_type}")
        
        if self.authentication and not self.authentication.is_valid():
            errors.append("Invalid authentication configuration")
        
        if self.rate_limits and not self.rate_limits.is_valid():
            errors.append("Invalid rate limit configuration")
        
        return errors


@dataclass
class AccessConfig:
    """Data source access configuration."""
    connection_string: Optional[str] = None
    api_endpoint: Optional[str] = None
    file_path: Optional[str] = None
    authentication: Optional[AuthConfig] = None
    timeout_seconds: int = 30
    
    def validate(self, source_type: SourceType) -> List[str]:
        """Validate access configuration for given source type."""
        errors = []
        
        if source_type == SourceType.API and not self.api_endpoint:
            errors.append("API endpoint is required for API sources")
        elif source_type == SourceType.DATABASE and not self.connection_string:
            errors.append("Connection string is required for database sources")
        elif source_type == SourceType.FILE and not self.file_path:
            errors.append("File path is required for file sources")
        
        if self.authentication and not self.authentication.is_valid():
            errors.append("Invalid authentication configuration")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        return errors


@dataclass
class DataSourceConfig:
    """Data source configuration."""
    source_id: str
    source_name: str
    source_type: SourceType
    agent_types: List[str]
    access_config: AccessConfig
    update_frequency: str = "daily"
    reliability_score: float = 0.5
    data_schema: Optional[Dict[str, Any]] = None
    is_active: bool = True
    last_sync: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def validate(self) -> List[str]:
        """Validate data source configuration."""
        errors = []
        
        if not self.source_id or not self.source_id.strip():
            errors.append("Source ID is required")
        
        if not self.source_name or not self.source_name.strip():
            errors.append("Source name is required")
        
        if not 0 <= self.reliability_score <= 1:
            errors.append("Reliability score must be between 0 and 1")
        
        if self.update_frequency not in ["real-time", "hourly", "daily", "weekly", "monthly"]:
            errors.append("Invalid update frequency")
        
        # Validate access configuration
        access_errors = self.access_config.validate(self.source_type)
        errors.extend(access_errors)
        
        return errors


@dataclass
class LLMConfig:
    """LLM configuration for agents."""
    model_name: str
    api_endpoint: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 30
    
    def validate(self) -> List[str]:
        """Validate LLM configuration."""
        errors = []
        
        if not self.model_name:
            errors.append("Model name is required")
        
        if not self.api_endpoint:
            errors.append("API endpoint is required")
        
        if self.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        
        if not 0 <= self.temperature <= 2:
            errors.append("Temperature must be between 0 and 2")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        return errors


@dataclass
class AgentConfig:
    """Agent configuration."""
    agent_id: str
    agent_type: str
    name: str
    description: str
    available_tools: List[str]
    available_data_sources: List[str]
    llm_config: LLMConfig
    prompt_templates: Dict[str, str]
    is_active: bool = True
    performance_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    def validate(self) -> List[str]:
        """Validate agent configuration."""
        errors = []
        
        if not self.agent_id or not self.agent_id.strip():
            errors.append("Agent ID is required")
        
        if not self.name or not self.name.strip():
            errors.append("Agent name is required")
        
        if self.agent_type not in ["industry", "financial", "market", "risk"]:
            errors.append("Invalid agent type")
        
        # Validate LLM configuration
        llm_errors = self.llm_config.validate()
        errors.extend([f"LLM config: {error}" for error in llm_errors])
        
        return errors


class ConfigurationManager:
    """Service for managing tools, data sources, and agent configurations."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._tools: Dict[str, ToolConfig] = {}
        self._data_sources: Dict[str, DataSourceConfig] = {}
        self._agents: Dict[str, AgentConfig] = {}
        self._permissions: Dict[str, Set[str]] = {}  # agent_type -> set of tool/source IDs
        
    # Tool Management
    
    async def register_tool(self, tool_config: ToolConfig) -> bool:
        """Register a new tool configuration."""
        try:
            # Validate configuration
            errors = tool_config.validate()
            if errors:
                logger.error(f"Tool validation failed: {errors}")
                return False
            
            # Check for duplicate tool ID
            if tool_config.tool_id in self._tools:
                logger.warning(f"Tool {tool_config.tool_id} already exists, updating")
            
            # Store configuration
            tool_config.updated_at = datetime.utcnow()
            self._tools[tool_config.tool_id] = tool_config
            
            # Update permissions for agent types
            for agent_type in tool_config.agent_types:
                if agent_type not in self._permissions:
                    self._permissions[agent_type] = set()
                self._permissions[agent_type].add(tool_config.tool_id)
            
            logger.info(f"Successfully registered tool: {tool_config.tool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool: {e}")
            return False
    
    async def register_data_source(self, source_config: DataSourceConfig) -> bool:
        """Register a new data source configuration."""
        try:
            # Validate configuration
            errors = source_config.validate()
            if errors:
                logger.error(f"Data source validation failed: {errors}")
                return False
            
            # Check for duplicate source ID
            if source_config.source_id in self._data_sources:
                logger.warning(f"Data source {source_config.source_id} already exists, updating")
            
            # Store configuration
            source_config.updated_at = datetime.utcnow()
            self._data_sources[source_config.source_id] = source_config
            
            # Update permissions for agent types
            for agent_type in source_config.agent_types:
                if agent_type not in self._permissions:
                    self._permissions[agent_type] = set()
                self._permissions[agent_type].add(source_config.source_id)
            
            logger.info(f"Successfully registered data source: {source_config.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering data source: {e}")
            return False
    
    async def register_agent(self, agent_config: AgentConfig) -> bool:
        """Register a new agent configuration."""
        try:
            # Validate configuration
            errors = agent_config.validate()
            if errors:
                logger.error(f"Agent validation failed: {errors}")
                return False
            
            # Validate tool and data source references
            for tool_id in agent_config.available_tools:
                if tool_id not in self._tools:
                    logger.error(f"Referenced tool {tool_id} not found")
                    return False
            
            for source_id in agent_config.available_data_sources:
                if source_id not in self._data_sources:
                    logger.error(f"Referenced data source {source_id} not found")
                    return False
            
            # Store configuration
            agent_config.updated_at = datetime.utcnow()
            self._agents[agent_config.agent_id] = agent_config
            
            logger.info(f"Successfully registered agent: {agent_config.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return False
    
    # Configuration Retrieval
    
    def get_agent_tools(self, agent_type: str) -> List[ToolConfig]:
        """Get available tools for an agent type."""
        if agent_type not in self._permissions:
            return []
        
        tool_ids = self._permissions[agent_type]
        return [
            self._tools[tool_id] 
            for tool_id in tool_ids 
            if tool_id in self._tools and self._tools[tool_id].is_active
        ]
    
    def get_agent_data_sources(self, agent_type: str) -> List[DataSourceConfig]:
        """Get available data sources for an agent type."""
        if agent_type not in self._permissions:
            return []
        
        source_ids = self._permissions[agent_type]
        return [
            self._data_sources[source_id] 
            for source_id in source_ids 
            if source_id in self._data_sources and self._data_sources[source_id].is_active
        ]
    
    def get_tool_config(self, tool_id: str) -> Optional[ToolConfig]:
        """Get tool configuration by ID."""
        return self._tools.get(tool_id)
    
    def get_data_source_config(self, source_id: str) -> Optional[DataSourceConfig]:
        """Get data source configuration by ID."""
        return self._data_sources.get(source_id)
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID."""
        return self._agents.get(agent_id)
    
    # Configuration Validation and Health Checks
    
    async def validate_tool_access(self, tool_id: str) -> bool:
        """Validate tool access and connectivity."""
        tool_config = self.get_tool_config(tool_id)
        if not tool_config or not tool_config.is_active:
            return False
        
        try:
            # Perform basic connectivity check
            if tool_config.api_endpoint:
                # This would typically make an HTTP request to check connectivity
                # For now, we'll just validate the configuration
                pass
            
            # Update last health check
            tool_config.last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Tool access validation failed for {tool_id}: {e}")
            return False
    
    async def update_tool_config(self, tool_id: str, updates: Dict[str, Any]) -> bool:
        """Update tool configuration."""
        if tool_id not in self._tools:
            logger.error(f"Tool {tool_id} not found")
            return False
        
        try:
            tool_config = self._tools[tool_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(tool_config, key):
                    setattr(tool_config, key, value)
            
            # Update timestamp
            tool_config.updated_at = datetime.utcnow()
            
            # Validate updated configuration
            errors = tool_config.validate()
            if errors:
                logger.error(f"Updated tool configuration is invalid: {errors}")
                return False
            
            logger.info(f"Successfully updated tool configuration: {tool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating tool configuration: {e}")
            return False
    
    async def update_data_source_config(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """Update data source configuration."""
        if source_id not in self._data_sources:
            logger.error(f"Data source {source_id} not found")
            return False
        
        try:
            source_config = self._data_sources[source_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(source_config, key):
                    setattr(source_config, key, value)
            
            # Update timestamp
            source_config.updated_at = datetime.utcnow()
            
            # Validate updated configuration
            errors = source_config.validate()
            if errors:
                logger.error(f"Updated data source configuration is invalid: {errors}")
                return False
            
            logger.info(f"Successfully updated data source configuration: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data source configuration: {e}")
            return False
    
    # Permission Management
    
    def grant_permission(self, agent_type: str, resource_id: str) -> bool:
        """Grant permission for an agent type to access a resource."""
        try:
            if agent_type not in self._permissions:
                self._permissions[agent_type] = set()
            
            self._permissions[agent_type].add(resource_id)
            logger.info(f"Granted permission for {agent_type} to access {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error granting permission: {e}")
            return False
    
    def revoke_permission(self, agent_type: str, resource_id: str) -> bool:
        """Revoke permission for an agent type to access a resource."""
        try:
            if agent_type in self._permissions:
                self._permissions[agent_type].discard(resource_id)
                logger.info(f"Revoked permission for {agent_type} to access {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking permission: {e}")
            return False
    
    def has_permission(self, agent_type: str, resource_id: str) -> bool:
        """Check if an agent type has permission to access a resource."""
        return (
            agent_type in self._permissions and 
            resource_id in self._permissions[agent_type]
        )
    
    # Configuration Export/Import
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export all configurations to a dictionary."""
        return {
            "tools": {tool_id: asdict(config) for tool_id, config in self._tools.items()},
            "data_sources": {source_id: asdict(config) for source_id, config in self._data_sources.items()},
            "agents": {agent_id: asdict(config) for agent_id, config in self._agents.items()},
            "permissions": {agent_type: list(perms) for agent_type, perms in self._permissions.items()},
            "exported_at": datetime.utcnow().isoformat()
        }
    
    async def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Import configurations from a dictionary."""
        try:
            # Import tools
            if "tools" in config_data:
                for tool_id, tool_data in config_data["tools"].items():
                    # Convert datetime strings back to datetime objects
                    if "created_at" in tool_data and isinstance(tool_data["created_at"], str):
                        tool_data["created_at"] = datetime.fromisoformat(tool_data["created_at"])
                    if "updated_at" in tool_data and isinstance(tool_data["updated_at"], str):
                        tool_data["updated_at"] = datetime.fromisoformat(tool_data["updated_at"])
                    
                    tool_config = ToolConfig(**tool_data)
                    await self.register_tool(tool_config)
            
            # Import data sources
            if "data_sources" in config_data:
                for source_id, source_data in config_data["data_sources"].items():
                    # Convert datetime strings back to datetime objects
                    if "created_at" in source_data and isinstance(source_data["created_at"], str):
                        source_data["created_at"] = datetime.fromisoformat(source_data["created_at"])
                    if "updated_at" in source_data and isinstance(source_data["updated_at"], str):
                        source_data["updated_at"] = datetime.fromisoformat(source_data["updated_at"])
                    
                    source_config = DataSourceConfig(**source_data)
                    await self.register_data_source(source_config)
            
            # Import agents
            if "agents" in config_data:
                for agent_id, agent_data in config_data["agents"].items():
                    # Convert datetime strings back to datetime objects
                    if "created_at" in agent_data and isinstance(agent_data["created_at"], str):
                        agent_data["created_at"] = datetime.fromisoformat(agent_data["created_at"])
                    if "updated_at" in agent_data and isinstance(agent_data["updated_at"], str):
                        agent_data["updated_at"] = datetime.fromisoformat(agent_data["updated_at"])
                    
                    agent_config = AgentConfig(**agent_data)
                    await self.register_agent(agent_config)
            
            # Import permissions
            if "permissions" in config_data:
                for agent_type, perms in config_data["permissions"].items():
                    self._permissions[agent_type] = set(perms)
            
            logger.info("Successfully imported configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False