"""Concurrent and robust tool execution system."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .state import ToolCall, ToolResult, ResearchState


logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Tool execution error."""
    pass


@dataclass
class ToolConfig:
    """Tool configuration."""
    name: str
    description: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    concurrent_limit: int = 5
    required_params: List[str] = None
    optional_params: List[str] = None


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.name = config.name
        self._semaphore = asyncio.Semaphore(config.concurrent_limit)
    
    @abstractmethod
    async def _execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters."""
        if self.config.required_params:
            for param in self.config.required_params:
                if param not in parameters:
                    raise ValueError(f"Required parameter '{param}' missing")
        return True
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute tool with retry logic and error handling."""
        call_id = parameters.get("call_id", "unknown")
        start_time = time.time()
        
        # Validate parameters
        try:
            self.validate_parameters(parameters)
        except ValueError as e:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
        
        # Execute with semaphore for concurrency control
        async with self._semaphore:
            for attempt in range(self.config.max_retries + 1):
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self._execute(parameters),
                        timeout=self.config.timeout
                    )
                    
                    return ToolResult(
                        call_id=call_id,
                        tool_name=self.name,
                        success=True,
                        result=result,
                        execution_time=time.time() - start_time
                    )
                
                except asyncio.TimeoutError:
                    error_msg = f"Tool '{self.name}' timed out after {self.config.timeout}s"
                    logger.warning(f"{error_msg} (attempt {attempt + 1})")
                    
                    if attempt == self.config.max_retries:
                        return ToolResult(
                            call_id=call_id,
                            tool_name=self.name,
                            success=False,
                            error_message=error_msg,
                            execution_time=time.time() - start_time
                        )
                
                except Exception as e:
                    error_msg = f"Tool '{self.name}' failed: {str(e)}"
                    logger.warning(f"{error_msg} (attempt {attempt + 1})")
                    
                    if attempt == self.config.max_retries:
                        return ToolResult(
                            call_id=call_id,
                            tool_name=self.name,
                            success=False,
                            error_message=error_msg,
                            execution_time=time.time() - start_time
                        )
                
                # Wait before retry
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff


class MCPSearchTool(BaseTool):
    """MCP search tool implementation."""
    
    def __init__(self):
        config = ToolConfig(
            name="mcp_search",
            description="Search external data using MCP protocol",
            timeout=30.0,
            max_retries=3,
            concurrent_limit=10,
            required_params=["query"],
            optional_params=["domain", "limit", "filters"]
        )
        super().__init__(config)
    
    async def _execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute MCP search."""
        from ..services.mcp_search import MCPSearchService, SearchQuery
        
        service = MCPSearchService()
        try:
            query = SearchQuery(
                query=parameters["query"],
                domain=parameters.get("domain"),
                filters=parameters.get("filters", {}),
                limit=parameters.get("limit", 10)
            )
            
            result = await service.search_market_data(query)
            return {
                "symbol": result.symbol,
                "data": result.data,
                "timestamp": result.timestamp.isoformat(),
                "source": result.source,
                "quality_score": result.quality_score
            }
        finally:
            await service.close()


class RAGRetrievalTool(BaseTool):
    """RAG retrieval tool implementation."""
    
    def __init__(self):
        config = ToolConfig(
            name="rag_retrieval",
            description="Retrieve relevant context from knowledge base",
            timeout=20.0,
            max_retries=2,
            concurrent_limit=15,
            required_params=["query", "domain"],
            optional_params=["limit", "min_relevance"]
        )
        super().__init__(config)
    
    async def _execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute RAG retrieval."""
        from ..services.rag_service import RAGService
        
        service = RAGService()
        documents = await service.retrieve_context(
            query=parameters["query"],
            domain=parameters["domain"],
            limit=parameters.get("limit", 5),
            min_relevance=parameters.get("min_relevance", 0.3)
        )
        
        return [
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "content": doc.content[:500],  # Truncate for efficiency
                "relevance_score": doc.relevance_score,
                "document_type": doc.document_type,
                "tags": doc.tags
            }
            for doc in documents
        ]


class DataValidationTool(BaseTool):
    """Data validation and quality assessment tool."""
    
    def __init__(self):
        config = ToolConfig(
            name="data_validation",
            description="Validate and assess data quality",
            timeout=10.0,
            max_retries=1,
            concurrent_limit=20,
            required_params=["data"],
            optional_params=["validation_rules"]
        )
        super().__init__(config)
    
    async def _execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute data validation."""
        data = parameters["data"]
        validation_rules = parameters.get("validation_rules", {})
        
        # Simple validation logic
        issues = []
        quality_score = 1.0
        
        if not data:
            issues.append("Data is empty")
            quality_score = 0.0
        elif isinstance(data, dict):
            # Check for missing values
            missing_count = sum(1 for v in data.values() if v is None)
            if missing_count > 0:
                issues.append(f"{missing_count} missing values found")
                quality_score -= (missing_count / len(data)) * 0.5
        
        return {
            "is_valid": len(issues) == 0,
            "quality_score": max(0.0, quality_score),
            "issues": issues,
            "data_size": len(str(data)) if data else 0
        }


class ToolExecutor:
    """Concurrent tool executor with robust error handling."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        self.register_tool(MCPSearchTool())
        self.register_tool(RAGRetrievalTool())
        self.register_tool(DataValidationTool())
    
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self.tools.keys())
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self.get_tool(tool_call.tool_name)
        if not tool:
            return ToolResult(
                call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                success=False,
                error_message=f"Tool '{tool_call.tool_name}' not found"
            )
        
        # Add call_id to parameters
        parameters = tool_call.parameters.copy()
        parameters["call_id"] = tool_call.id
        
        return await tool.execute(parameters)
    
    async def execute_tools_concurrent(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls concurrently."""
        if not tool_calls:
            return []
        
        logger.info(f"Executing {len(tool_calls)} tools concurrently")
        
        # Create tasks for concurrent execution
        tasks = [
            asyncio.create_task(self.execute_tool(call))
            for call in tool_calls
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    call_id=tool_calls[i].id,
                    tool_name=tool_calls[i].tool_name,
                    success=False,
                    error_message=f"Execution failed: {str(result)}"
                ))
            else:
                final_results.append(result)
        
        logger.info(f"Completed {len(final_results)} tool executions")
        return final_results
    
    async def execute_tools_for_state(self, state: ResearchState) -> None:
        """Execute pending tool calls for a research state."""
        # Find pending tool calls
        pending_calls = [
            call for call in state.tool_calls
            if call.id not in state.tool_results
        ]
        
        if not pending_calls:
            return
        
        # Execute tools concurrently
        results = await self.execute_tools_concurrent(pending_calls)
        
        # Add results to state
        for result in results:
            state.add_tool_result(result)


# Global tool executor instance
tool_executor = ToolExecutor()