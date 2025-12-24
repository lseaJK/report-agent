"""LangChain framework setup and configuration."""

import os
from typing import Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

from ..config.settings import settings
from .siliconcloud_llm import SiliconCloudLLM


class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LangChain operations."""
    
    def on_llm_start(
        self, 
        serialized: dict, 
        prompts: list[str], 
        **kwargs
    ) -> None:
        """Called when LLM starts running."""
        print(f"LLM started with {len(prompts)} prompts")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running."""
        print(f"LLM finished with {len(response.generations)} generations")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        print(f"LLM error: {error}")


def setup_langchain() -> None:
    """Setup LangChain environment and tracing."""
    # Set environment variables for LangChain
    if settings.langchain.tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    if settings.langchain.api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain.api_key
    
    if settings.langchain.project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain.project
    
    # Note: SiliconCloud API key is read directly from environment variable
    # in the SiliconCloudLLM class to prevent accidental exposure


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    callbacks: Optional[list[BaseCallbackHandler]] = None
) -> BaseLanguageModel:
    """Create and configure a language model instance.
    
    Args:
        model: Model name to use (defaults to settings)
        temperature: Model temperature (defaults to settings)
        max_tokens: Maximum tokens per request (defaults to settings)
        callbacks: List of callback handlers
    
    Returns:
        Configured language model instance
    """
    return SiliconCloudLLM(
        model=model or settings.ai_service.model,
        temperature=temperature or settings.ai_service.temperature,
        max_tokens=max_tokens or settings.ai_service.max_tokens,
        top_p=settings.ai_service.top_p,
        top_k=settings.ai_service.top_k,
        frequency_penalty=settings.ai_service.frequency_penalty,
        min_p=settings.ai_service.min_p,
        api_base=settings.ai_service.api_base,
        callbacks=callbacks or [CustomCallbackHandler()],
    )


def create_agent_llm(agent_type: str) -> BaseLanguageModel:
    """Create a specialized LLM instance for a specific agent type.
    
    Args:
        agent_type: Type of agent (industry, financial, market, risk)
    
    Returns:
        Configured language model instance for the agent
    """
    # Agent-specific configurations
    agent_configs = {
        "industry": {
            "temperature": 0.6,
            "max_tokens": 3000,
        },
        "financial": {
            "temperature": 0.3,  # More deterministic for financial calculations
            "max_tokens": 4000,
        },
        "market": {
            "temperature": 0.7,
            "max_tokens": 3500,
        },
        "risk": {
            "temperature": 0.3,  # Conservative for risk assessment
            "max_tokens": 3000,
        }
    }
    
    config = agent_configs.get(agent_type, {})
    
    return create_llm(
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
    )


# Initialize LangChain on module import
setup_langchain()