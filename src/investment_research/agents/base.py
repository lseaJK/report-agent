"""Base agent class and interfaces for state-based multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
import time
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel

from ..core.models import AgentType
from ..core.langchain_setup import create_agent_llm
from ..core.state import ResearchState, Message, MessageRole, AgentResult, AgentStatus, ToolCall
from ..core.tool_executor import tool_executor


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all specialized agents using shared state."""
    
    def __init__(
        self, 
        agent_id: str,
        agent_type: AgentType,
        llm: Optional[BaseLanguageModel] = None
    ):
        """Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of the agent
            llm: Language model instance (optional)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.llm = llm or create_agent_llm(agent_type.value)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    @abstractmethod
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform analysis and update the shared state.
        
        Args:
            state: Current research state
        
        Returns:
            Updated research state
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    @abstractmethod
    def get_required_tools(self) -> List[str]:
        """Get list of tools this agent requires."""
        pass
    
    async def execute_with_state(self, state: ResearchState) -> ResearchState:
        """Execute agent analysis with proper state management and error handling."""
        start_time = time.time()
        
        # Update agent status to running
        state.update_agent_status(self.agent_id, AgentStatus.RUNNING)
        state.current_agent = self.agent_id
        
        # Add system message
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.get_system_prompt(),
            agent_id=self.agent_id
        )
        state.add_message(system_message)
        
        try:
            self.logger.info(f"Starting analysis for {self.agent_type.value} agent")
            
            # Perform the actual analysis
            updated_state = await self.analyze(state)
            
            # Create agent result
            execution_time = time.time() - start_time
            result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type.value,
                status=AgentStatus.COMPLETED,
                content=self._extract_analysis_content(updated_state),
                insights=self._extract_insights(updated_state),
                confidence_score=self._calculate_confidence(updated_state),
                data_sources=self._extract_data_sources(updated_state),
                execution_time=execution_time
            )
            
            # Add result to state
            updated_state.add_agent_result(result)
            
            self.logger.info(f"Completed analysis in {execution_time:.2f}s")
            return updated_state
            
        except Exception as e:
            # Handle errors gracefully
            execution_time = time.time() - start_time
            error_result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type.value,
                status=AgentStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
            
            state.add_agent_result(error_result)
            
            # Add error message to conversation
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=f"分析过程中发生错误: {str(e)}",
                agent_id=self.agent_id,
                metadata={"error": True}
            )
            state.add_message(error_message)
            
            self.logger.error(f"Analysis failed after {execution_time:.2f}s: {e}")
            return state
    
    async def call_tools(self, state: ResearchState, tool_calls: List[ToolCall]) -> ResearchState:
        """Call tools concurrently and update state with results."""
        if not tool_calls:
            return state
        
        self.logger.info(f"Calling {len(tool_calls)} tools concurrently")
        
        # Add tool calls to state
        for call in tool_calls:
            call.agent_id = self.agent_id
            state.add_tool_call(call)
        
        # Execute tools concurrently
        await tool_executor.execute_tools_for_state(state)
        
        # Add tool results to conversation
        for call in tool_calls:
            result = state.tool_results.get(call.id)
            if result:
                tool_message = Message(
                    role=MessageRole.TOOL,
                    content=f"工具 {result.tool_name} 执行{'成功' if result.success else '失败'}",
                    agent_id=self.agent_id,
                    metadata={
                        "tool_name": result.tool_name,
                        "success": result.success,
                        "execution_time": result.execution_time
                    }
                )
                state.add_message(tool_message)
        
        return state
    
    async def generate_response(self, state: ResearchState, prompt: str) -> str:
        """Generate response using LLM with context from state."""
        try:
            # Build context from state
            context = self._build_context_from_state(state)
            full_prompt = f"{context}\n\n{prompt}"
            
            # Generate response
            response = await self.llm._acall(full_prompt)
            
            # Add to conversation
            message = Message(
                role=MessageRole.ASSISTANT,
                content=response,
                agent_id=self.agent_id
            )
            state.add_message(message)
            
            return response
            
        except Exception as e:
            error_msg = f"LLM调用失败: {str(e)}"
            self.logger.error(error_msg)
            
            # Add error message
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=error_msg,
                agent_id=self.agent_id,
                metadata={"error": True}
            )
            state.add_message(error_message)
            
            return error_msg
    
    def _build_context_from_state(self, state: ResearchState) -> str:
        """Build context string from research state."""
        context_parts = [
            f"研究主题: {state.topic}",
            f"当前步骤: {state.current_step}"
        ]
        
        # Add previous agent results
        if state.agent_results:
            context_parts.append("\n已完成的分析:")
            for agent_id, result in state.agent_results.items():
                if result.status == AgentStatus.COMPLETED and agent_id != self.agent_id:
                    context_parts.append(f"- {result.agent_type}: {result.insights[:2] if result.insights else '无特殊见解'}")
        
        # Add collected data summary
        if state.collected_data:
            context_parts.append(f"\n可用数据源: {len(state.collected_data)} 个")
        
        return "\n".join(context_parts)
    
    def _extract_analysis_content(self, state: ResearchState) -> Dict[str, Any]:
        """Extract analysis content from state for this agent."""
        # Get messages from this agent
        agent_messages = [
            msg for msg in state.messages 
            if msg.agent_id == self.agent_id and msg.role == MessageRole.ASSISTANT
        ]
        
        return {
            "messages": [msg.content for msg in agent_messages],
            "data_used": len([call for call in state.tool_calls if call.agent_id == self.agent_id])
        }
    
    def _extract_insights(self, state: ResearchState) -> List[str]:
        """Extract insights from agent's analysis."""
        # Simple extraction - in practice, this would be more sophisticated
        agent_messages = [
            msg.content for msg in state.messages 
            if msg.agent_id == self.agent_id and msg.role == MessageRole.ASSISTANT
        ]
        
        # Extract key points (simplified)
        insights = []
        for content in agent_messages:
            if "关键发现" in content or "重要观点" in content:
                insights.append(content[:100] + "...")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _calculate_confidence(self, state: ResearchState) -> float:
        """Calculate confidence score based on data quality and completeness."""
        # Simple confidence calculation
        base_confidence = 0.5
        
        # Boost confidence based on successful tool calls
        successful_tools = sum(
            1 for call in state.tool_calls 
            if call.agent_id == self.agent_id and 
            state.tool_results.get(call.id, {}).success
        )
        
        confidence = base_confidence + (successful_tools * 0.1)
        return min(1.0, confidence)
    
    def _extract_data_sources(self, state: ResearchState) -> List[str]:
        """Extract data sources used by this agent."""
        sources = []
        for call in state.tool_calls:
            if call.agent_id == self.agent_id:
                result = state.tool_results.get(call.id)
                if result and result.success:
                    sources.append(f"{result.tool_name}:{call.parameters.get('query', 'unknown')}")
        
        return sources