"""Tool abstraction layer for unified tool calling interface."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import aiohttp
from urllib.parse import urljoin

from .configuration_manager import ConfigurationManager, ToolConfig, AuthType

logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    """Tool status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class ToolCall:
    """Represents a tool call request."""
    tool_id: str
    method: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize call ID and timestamp."""
        if self.call_id is None:
            # Generate unique call ID
            content = f"{self.tool_id}_{self.method}_{json.dumps(self.parameters, sort_keys=True)}"
            self.call_id = hashlib.md5(content.encode()).hexdigest()[:12]
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ToolResult:
    """Represents a tool call result."""
    call_id: str
    tool_id: str
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    timestamp: Optional[datetime] = None
    cached: bool = False
    
    def __post_init__(self):
        """Initialize timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class RateLimitState:
    """Tracks rate limiting state for a tool."""
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    concurrent_requests: int = 0
    last_reset_minute: datetime = None
    last_reset_hour: datetime = None
    
    def __post_init__(self):
        """Initialize reset times."""
        now = datetime.utcnow()
        if self.last_reset_minute is None:
            self.last_reset_minute = now
        if self.last_reset_hour is None:
            self.last_reset_hour = now


class ToolManager:
    """Unified tool calling interface with rate limiting, caching, and monitoring."""
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize tool manager."""
        self.config_manager = config_manager
        self._rate_limits: Dict[str, RateLimitState] = {}
        self._tool_status: Dict[str, ToolStatus] = {}
        self._cache: Dict[str, ToolResult] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._default_cache_ttl = timedelta(minutes=15)
        self._max_cache_size = 1000
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], 
                       method: str = "execute") -> ToolResult:
        """Call a single tool with the given parameters."""
        tool_call = ToolCall(
            tool_id=tool_name,
            method=method,
            parameters=parameters
        )
        
        return await self._execute_tool_call(tool_call)
    
    async def batch_call_tools(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls concurrently."""
        # Group calls by tool to respect rate limits
        calls_by_tool = {}
        for call in calls:
            if call.tool_id not in calls_by_tool:
                calls_by_tool[call.tool_id] = []
            calls_by_tool[call.tool_id].append(call)
        
        # Execute calls with proper rate limiting
        results = []
        for tool_id, tool_calls in calls_by_tool.items():
            tool_results = await self._execute_tool_calls_batch(tool_id, tool_calls)
            results.extend(tool_results)
        
        return results
    
    def get_tool_status(self, tool_name: str) -> ToolStatus:
        """Get current status of a tool."""
        return self._tool_status.get(tool_name, ToolStatus.INACTIVE)
    
    async def cache_tool_result(self, call: ToolCall, result: ToolResult, 
                               ttl: Optional[timedelta] = None) -> None:
        """Cache a tool result."""
        if ttl is None:
            ttl = self._default_cache_ttl
        
        cache_key = self._get_cache_key(call)
        
        # Clean cache if it's getting too large
        if len(self._cache) >= self._max_cache_size:
            await self._clean_cache()
        
        # Store result and TTL
        result.cached = True
        self._cache[cache_key] = result
        self._cache_ttl[cache_key] = datetime.utcnow() + ttl
        
        logger.debug(f"Cached result for {call.tool_id}.{call.method}")
    
    async def _execute_tool_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = await self._get_cached_result(call)
            if cached_result:
                return cached_result
            
            # Get tool configuration
            tool_config = self.config_manager.get_tool_config(call.tool_id)
            if not tool_config:
                return ToolResult(
                    call_id=call.call_id,
                    tool_id=call.tool_id,
                    success=False,
                    error_message=f"Tool {call.tool_id} not found"
                )
            
            # Check if tool is active
            if not tool_config.is_active:
                return ToolResult(
                    call_id=call.call_id,
                    tool_id=call.tool_id,
                    success=False,
                    error_message=f"Tool {call.tool_id} is inactive"
                )
            
            # Check rate limits
            if not await self._check_rate_limits(call.tool_id, tool_config):
                return ToolResult(
                    call_id=call.call_id,
                    tool_id=call.tool_id,
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Execute the tool call
            result = await self._execute_tool_method(call, tool_config)
            
            # Update execution time
            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time
            
            # Cache successful results
            if result.success:
                await self.cache_tool_result(call, result)
            
            # Update tool status
            self._tool_status[call.tool_id] = ToolStatus.ACTIVE
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool call {call.call_id}: {e}")
            self._tool_status[call.tool_id] = ToolStatus.ERROR
            
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=False,
                error_message=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _execute_tool_calls_batch(self, tool_id: str, calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple calls for the same tool with rate limiting."""
        tool_config = self.config_manager.get_tool_config(tool_id)
        if not tool_config:
            return [
                ToolResult(
                    call_id=call.call_id,
                    tool_id=call.tool_id,
                    success=False,
                    error_message=f"Tool {tool_id} not found"
                )
                for call in calls
            ]
        
        # Respect concurrent request limits
        max_concurrent = (
            tool_config.rate_limits.concurrent_requests 
            if tool_config.rate_limits 
            else 5
        )
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(call):
            async with semaphore:
                return await self._execute_tool_call(call)
        
        # Execute calls in batches
        tasks = [execute_with_semaphore(call) for call in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    call_id=calls[i].call_id,
                    tool_id=calls[i].tool_id,
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_tool_method(self, call: ToolCall, tool_config: ToolConfig) -> ToolResult:
        """Execute the actual tool method based on tool type."""
        try:
            if tool_config.tool_type.value == "api":
                return await self._execute_api_call(call, tool_config)
            elif tool_config.tool_type.value == "database":
                return await self._execute_database_call(call, tool_config)
            elif tool_config.tool_type.value == "calculator":
                return await self._execute_calculator_call(call, tool_config)
            elif tool_config.tool_type.value == "file_processor":
                return await self._execute_file_processor_call(call, tool_config)
            else:
                return ToolResult(
                    call_id=call.call_id,
                    tool_id=call.tool_id,
                    success=False,
                    error_message=f"Unsupported tool type: {tool_config.tool_type}"
                )
                
        except Exception as e:
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=False,
                error_message=f"Tool execution error: {str(e)}"
            )
    
    async def _execute_api_call(self, call: ToolCall, tool_config: ToolConfig) -> ToolResult:
        """Execute an API-based tool call."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        
        # Add authentication
        if tool_config.authentication:
            auth_config = tool_config.authentication
            if auth_config.auth_type == AuthType.API_KEY:
                api_key = auth_config.credentials.get("api_key")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
            elif auth_config.auth_type == AuthType.BASIC:
                username = auth_config.credentials.get("username")
                password = auth_config.credentials.get("password")
                if username and password:
                    import base64
                    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                    headers["Authorization"] = f"Basic {credentials}"
            
            # Add custom headers
            if auth_config.headers:
                headers.update(auth_config.headers)
        
        # Prepare request
        url = tool_config.api_endpoint
        if call.method != "execute":
            url = urljoin(url, call.method)
        
        try:
            async with self._session.post(
                url,
                json=call.parameters,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return ToolResult(
                        call_id=call.call_id,
                        tool_id=call.tool_id,
                        success=True,
                        data=data
                    )
                else:
                    error_text = await response.text()
                    return ToolResult(
                        call_id=call.call_id,
                        tool_id=call.tool_id,
                        success=False,
                        error_message=f"HTTP {response.status}: {error_text}"
                    )
                    
        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=False,
                error_message=f"Request failed: {str(e)}"
            )
    
    async def _execute_database_call(self, call: ToolCall, tool_config: ToolConfig) -> ToolResult:
        """Execute a database-based tool call."""
        # This would integrate with database connections
        # For now, return a placeholder implementation
        return ToolResult(
            call_id=call.call_id,
            tool_id=call.tool_id,
            success=True,
            data={"message": "Database tool executed", "parameters": call.parameters}
        )
    
    async def _execute_calculator_call(self, call: ToolCall, tool_config: ToolConfig) -> ToolResult:
        """Execute a calculator-based tool call."""
        try:
            # Simple calculator implementation
            if call.method == "add":
                result = call.parameters.get("a", 0) + call.parameters.get("b", 0)
            elif call.method == "multiply":
                result = call.parameters.get("a", 1) * call.parameters.get("b", 1)
            elif call.method == "divide":
                a = call.parameters.get("a", 0)
                b = call.parameters.get("b", 1)
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
            else:
                raise ValueError(f"Unknown calculator method: {call.method}")
            
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=True,
                data={"result": result}
            )
            
        except Exception as e:
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_file_processor_call(self, call: ToolCall, tool_config: ToolConfig) -> ToolResult:
        """Execute a file processor tool call."""
        # This would integrate with file processing capabilities
        # For now, return a placeholder implementation
        return ToolResult(
            call_id=call.call_id,
            tool_id=call.tool_id,
            success=True,
            data={"message": "File processor executed", "parameters": call.parameters}
        )
    
    async def _check_rate_limits(self, tool_id: str, tool_config: ToolConfig) -> bool:
        """Check if tool call is within rate limits."""
        if not tool_config.rate_limits:
            return True
        
        # Initialize rate limit state if needed
        if tool_id not in self._rate_limits:
            self._rate_limits[tool_id] = RateLimitState()
        
        state = self._rate_limits[tool_id]
        now = datetime.utcnow()
        
        # Reset counters if needed
        if (now - state.last_reset_minute).total_seconds() >= 60:
            state.requests_this_minute = 0
            state.last_reset_minute = now
        
        if (now - state.last_reset_hour).total_seconds() >= 3600:
            state.requests_this_hour = 0
            state.last_reset_hour = now
        
        # Check limits
        rate_limits = tool_config.rate_limits
        
        if state.requests_this_minute >= rate_limits.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {tool_id}: minute limit")
            self._tool_status[tool_id] = ToolStatus.RATE_LIMITED
            return False
        
        if state.requests_this_hour >= rate_limits.requests_per_hour:
            logger.warning(f"Rate limit exceeded for {tool_id}: hour limit")
            self._tool_status[tool_id] = ToolStatus.RATE_LIMITED
            return False
        
        if state.concurrent_requests >= rate_limits.concurrent_requests:
            logger.warning(f"Rate limit exceeded for {tool_id}: concurrent limit")
            self._tool_status[tool_id] = ToolStatus.RATE_LIMITED
            return False
        
        # Update counters
        state.requests_this_minute += 1
        state.requests_this_hour += 1
        state.concurrent_requests += 1
        
        # Schedule concurrent request decrement
        asyncio.create_task(self._decrement_concurrent_requests(tool_id))
        
        return True
    
    async def _decrement_concurrent_requests(self, tool_id: str):
        """Decrement concurrent request counter after a delay."""
        await asyncio.sleep(1)  # Assume average request duration of 1 second
        if tool_id in self._rate_limits:
            self._rate_limits[tool_id].concurrent_requests = max(
                0, self._rate_limits[tool_id].concurrent_requests - 1
            )
    
    def _get_cache_key(self, call: ToolCall) -> str:
        """Generate cache key for a tool call."""
        content = f"{call.tool_id}_{call.method}_{json.dumps(call.parameters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_cached_result(self, call: ToolCall) -> Optional[ToolResult]:
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(call)
        
        if cache_key not in self._cache:
            return None
        
        # Check if cache entry has expired
        if cache_key in self._cache_ttl:
            if datetime.utcnow() > self._cache_ttl[cache_key]:
                # Remove expired entry
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
                return None
        
        result = self._cache[cache_key]
        result.cached = True
        logger.debug(f"Cache hit for {call.tool_id}.{call.method}")
        return result
    
    async def _clean_cache(self):
        """Clean expired cache entries."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self._cache_ttl.items()
            if now > expiry
        ]
        
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        
        # If still too large, remove oldest entries
        if len(self._cache) >= self._max_cache_size:
            # Sort by timestamp and remove oldest
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].timestamp or datetime.min
            )
            
            items_to_remove = len(sorted_items) - self._max_cache_size + 100
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del self._cache[key]
                if key in self._cache_ttl:
                    del self._cache_ttl[key]
        
        logger.info(f"Cache cleaned, {len(self._cache)} entries remaining")
    
    # Health and Monitoring
    
    async def health_check_all_tools(self) -> Dict[str, ToolStatus]:
        """Perform health check on all registered tools."""
        results = {}
        
        # Get all tool configurations
        # This would need to be implemented in ConfigurationManager
        # For now, check tools we have status for
        for tool_id in self._tool_status.keys():
            try:
                is_healthy = await self.config_manager.validate_tool_access(tool_id)
                results[tool_id] = ToolStatus.ACTIVE if is_healthy else ToolStatus.ERROR
                self._tool_status[tool_id] = results[tool_id]
            except Exception as e:
                logger.error(f"Health check failed for {tool_id}: {e}")
                results[tool_id] = ToolStatus.ERROR
                self._tool_status[tool_id] = ToolStatus.ERROR
        
        return results
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        stats = {
            "cache_size": len(self._cache),
            "cache_hit_ratio": 0.0,  # Would need to track hits/misses
            "tools_status": dict(self._tool_status),
            "rate_limit_states": {
                tool_id: {
                    "requests_this_minute": state.requests_this_minute,
                    "requests_this_hour": state.requests_this_hour,
                    "concurrent_requests": state.concurrent_requests
                }
                for tool_id, state in self._rate_limits.items()
            }
        }
        
        return stats