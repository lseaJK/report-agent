"""SiliconCloud LLM integration for LangChain."""

import os
import json
from typing import Any, Dict, List, Optional, Iterator
import httpx
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import Field

from ..config.settings import settings


class SiliconCloudLLM(BaseLLM):
    """SiliconCloud LLM wrapper for LangChain."""
    
    model: str = Field(default="deepseek-ai/DeepSeek-V3.2")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)
    top_p: float = Field(default=0.7)
    top_k: int = Field(default=50)
    frequency_penalty: float = Field(default=0.5)
    min_p: float = Field(default=0.05)
    api_base: str = Field(default="https://api.siliconflow.cn/v1")
    
    # Private fields that won't be validated by Pydantic
    _client: Optional[httpx.AsyncClient] = None
    
    def __init__(self, **kwargs):
        """Initialize SiliconCloud LLM."""
        super().__init__(**kwargs)
        self._client = httpx.AsyncClient(timeout=60.0)
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating if necessary."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "siliconcloud"
    
    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv("SILICONCLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "SILICONCLOUD_API_KEY environment variable is required but not set"
            )
        return api_key
    
    def _format_messages(self, prompts: List[str]) -> List[Dict[str, str]]:
        """Format prompts as messages for the API."""
        messages = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
        return messages
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call to SiliconCloud API."""
        try:
            api_key = self._get_api_key()
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.frequency_penalty,
                "min_p": self.min_p,
                "stop": stop,
                "n": 1,
                "response_format": {"type": "text"}
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("No response from SiliconCloud API")
                
        except httpx.RequestError as e:
            raise Exception(f"SiliconCloud API request failed: {e}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"SiliconCloud API HTTP error: {e.response.status_code}")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous call to SiliconCloud API."""
        import asyncio
        
        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._acall(prompt, stop, run_manager, **kwargs)
        )
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate responses for multiple prompts."""
        generations = []
        
        for prompt in prompts:
            try:
                response = await self._acall(prompt, stop, run_manager, **kwargs)
                generations.append([Generation(text=response)])
            except Exception as e:
                # Handle individual prompt failures
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        return LLMResult(generations=generations)
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._agenerate(prompts, stop, run_manager, **kwargs)
        )
    
    async def aclose(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            import asyncio
            if self._client:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self._client.aclose())
        except:
            pass  # Ignore cleanup errors