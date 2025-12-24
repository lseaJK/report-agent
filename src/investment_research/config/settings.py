"""Application settings and configuration."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")
    
    url: str = Field(
        default="mysql+aiomysql://username:password@localhost:3306/investment_research",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")


class AIServiceSettings(BaseSettings):
    """AI service configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="SILICONCLOUD_")
    
    api_key: Optional[str] = Field(default=None, description="SiliconCloud API key")
    model: str = Field(default="deepseek-ai/DeepSeek-V3.2", description="AI model to use")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    top_p: float = Field(default=0.7, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    frequency_penalty: float = Field(default=0.5, description="Frequency penalty")
    min_p: float = Field(default=0.05, description="Min-p sampling parameter")
    api_base: str = Field(
        default="https://api.siliconflow.cn/v1", 
        description="SiliconCloud API base URL"
    )


class LangChainSettings(BaseSettings):
    """LangChain configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="LANGCHAIN_")
    
    tracing_v2: bool = Field(default=False, description="Enable LangChain tracing")
    api_key: Optional[str] = Field(default=None, description="LangChain API key")
    project: str = Field(
        default="investment-research-reports", 
        description="LangChain project name"
    )


class MCPSearchSettings(BaseSettings):
    """MCP Search service configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="MCP_SEARCH_")
    
    endpoint: str = Field(
        default="http://localhost:8080", 
        description="MCP search service endpoint"
    )
    api_key: Optional[str] = Field(default=None, description="MCP search API key")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class AppSettings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    app_name: str = Field(
        default="Investment Research Reports System", 
        description="Application name"
    )
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production", 
        description="Secret key for JWT tokens"
    )
    access_token_expire_minutes: int = Field(
        default=30, 
        description="Access token expiration time in minutes"
    )
    
    # External services
    bloomberg_api_key: Optional[str] = Field(
        default=None, 
        description="Bloomberg API key"
    )
    reuters_api_key: Optional[str] = Field(
        default=None, 
        description="Reuters API key"
    )
    alpha_vantage_api_key: Optional[str] = Field(
        default=None, 
        description="Alpha Vantage API key"
    )
    
    # Cache
    redis_url: str = Field(
        default="redis://localhost:6379/0", 
        description="Redis connection URL"
    )
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Report generation
    report_output_dir: str = Field(
        default="./reports", 
        description="Report output directory"
    )
    temp_dir: str = Field(default="./temp", description="Temporary files directory")
    max_concurrent_tasks: int = Field(
        default=5, 
        description="Maximum concurrent tasks"
    )
    
    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai_service: AIServiceSettings = Field(default_factory=AIServiceSettings)
    langchain: LangChainSettings = Field(default_factory=LangChainSettings)
    mcp_search: MCPSearchSettings = Field(default_factory=MCPSearchSettings)


# Global settings instance
settings = AppSettings()