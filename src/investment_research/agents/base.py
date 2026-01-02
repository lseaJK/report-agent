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


class AgentLifecycleManager:
    """Manages the lifecycle of agents in the multi-agent system."""
    
    def __init__(self):
        """Initialize the lifecycle manager."""
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_registry: Dict[str, type] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(f"{__name__}.lifecycle")
    
    def register_agent_type(self, agent_type: str, agent_class: type, 
                           dependencies: Optional[List[str]] = None):
        """Register an agent type with its class and dependencies."""
        self.agent_registry[agent_type] = agent_class
        self.agent_dependencies[agent_type] = dependencies or []
        self.logger.info(f"Registered agent type: {agent_type}")
    
    async def create_agent(self, agent_id: str, agent_type: AgentType, 
                          config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Create and initialize an agent instance."""
        if agent_type.value not in self.agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type.value}")
        
        agent_class = self.agent_registry[agent_type.value]
        
        # Create agent instance
        if config:
            agent = agent_class(agent_id, agent_type, **config)
        else:
            agent = agent_class(agent_id, agent_type)
        
        # Initialize agent
        await self._initialize_agent(agent)
        
        # Register as active
        self.active_agents[agent_id] = agent
        
        self.logger.info(f"Created agent: {agent_id} ({agent_type.value})")
        return agent
    
    async def destroy_agent(self, agent_id: str):
        """Destroy an agent and clean up resources."""
        if agent_id in self.active_agents:
            agent = self.active_agents[agent_id]
            await self._cleanup_agent(agent)
            del self.active_agents[agent_id]
            self.logger.info(f"Destroyed agent: {agent_id}")
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an active agent by ID."""
        return self.active_agents.get(agent_id)
    
    def get_active_agents(self) -> List[BaseAgent]:
        """Get all active agents."""
        return list(self.active_agents.values())
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all active agents of a specific type."""
        return [
            agent for agent in self.active_agents.values()
            if agent.agent_type == agent_type
        ]
    
    async def _initialize_agent(self, agent: BaseAgent):
        """Initialize an agent (setup resources, validate tools, etc.)."""
        try:
            # Validate required tools are available
            required_tools = agent.get_required_tools()
            # This would check with the tool manager
            
            # Initialize agent-specific resources
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            
            self.logger.debug(f"Initialized agent: {agent.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {agent.agent_id}: {e}")
            raise
    
    async def _cleanup_agent(self, agent: BaseAgent):
        """Clean up agent resources."""
        try:
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
            
            self.logger.debug(f"Cleaned up agent: {agent.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup agent {agent.agent_id}: {e}")


class AgentCommunicationProtocol:
    """Handles communication between agents in the multi-agent system."""
    
    def __init__(self):
        """Initialize the communication protocol."""
        self.message_queue: Dict[str, List[Message]] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> list of agent_ids
        self.logger = logging.getLogger(f"{__name__}.communication")
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a communication topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        
        if agent_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(agent_id)
            self.logger.debug(f"Agent {agent_id} subscribed to topic: {topic}")
    
    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a communication topic."""
        if topic in self.subscriptions and agent_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(agent_id)
            self.logger.debug(f"Agent {agent_id} unsubscribed from topic: {topic}")
    
    async def broadcast_message(self, sender_id: str, topic: str, 
                              content: str, metadata: Optional[Dict[str, Any]] = None):
        """Broadcast a message to all subscribers of a topic."""
        if topic not in self.subscriptions:
            return
        
        message = Message(
            role=MessageRole.SYSTEM,
            content=content,
            agent_id=sender_id,
            metadata=metadata or {}
        )
        
        # Add to message queues of all subscribers
        for subscriber_id in self.subscriptions[topic]:
            if subscriber_id != sender_id:  # Don't send to self
                if subscriber_id not in self.message_queue:
                    self.message_queue[subscriber_id] = []
                self.message_queue[subscriber_id].append(message)
        
        self.logger.debug(f"Broadcasted message from {sender_id} to topic {topic}")
    
    async def send_direct_message(self, sender_id: str, recipient_id: str, 
                                 content: str, metadata: Optional[Dict[str, Any]] = None):
        """Send a direct message to a specific agent."""
        message = Message(
            role=MessageRole.SYSTEM,
            content=content,
            agent_id=sender_id,
            metadata=metadata or {}
        )
        
        if recipient_id not in self.message_queue:
            self.message_queue[recipient_id] = []
        
        self.message_queue[recipient_id].append(message)
        self.logger.debug(f"Sent direct message from {sender_id} to {recipient_id}")
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get all pending messages for an agent."""
        messages = self.message_queue.get(agent_id, [])
        self.message_queue[agent_id] = []  # Clear after retrieval
        return messages
    
    def has_messages(self, agent_id: str) -> bool:
        """Check if an agent has pending messages."""
        return agent_id in self.message_queue and len(self.message_queue[agent_id]) > 0


class AgentCoordinator:
    """Coordinates multiple agents working on a research task."""
    
    def __init__(self, lifecycle_manager: AgentLifecycleManager, 
                 communication_protocol: AgentCommunicationProtocol):
        """Initialize the agent coordinator."""
        self.lifecycle_manager = lifecycle_manager
        self.communication_protocol = communication_protocol
        self.logger = logging.getLogger(f"{__name__}.coordinator")
    
    async def execute_parallel_analysis(self, state: ResearchState, 
                                      agent_types: List[AgentType]) -> ResearchState:
        """Execute analysis with multiple agents in parallel."""
        self.logger.info(f"Starting parallel analysis with {len(agent_types)} agents")
        
        # Create agents if they don't exist
        agents = []
        for i, agent_type in enumerate(agent_types):
            agent_id = f"{agent_type.value}_agent_{i}"
            
            # Try to get existing agent or create new one
            agent = await self.lifecycle_manager.get_agent(agent_id)
            if not agent:
                agent = await self.lifecycle_manager.create_agent(agent_id, agent_type)
            
            agents.append(agent)
        
        # Execute agents in parallel
        tasks = []
        for agent in agents:
            task = asyncio.create_task(agent.execute_with_state(state.copy()))
            tasks.append(task)
        
        # Wait for all agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results back into main state
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Agent {agents[i].agent_id} failed: {result}")
                # Add error result to state
                error_result = AgentResult(
                    agent_id=agents[i].agent_id,
                    agent_type=agents[i].agent_type.value,
                    status=AgentStatus.FAILED,
                    error_message=str(result)
                )
                state.add_agent_result(error_result)
            else:
                # Merge successful result
                state = self._merge_states(state, result)
        
        self.logger.info("Parallel analysis completed")
        return state
    
    async def execute_sequential_analysis(self, state: ResearchState, 
                                        agent_types: List[AgentType]) -> ResearchState:
        """Execute analysis with multiple agents sequentially."""
        self.logger.info(f"Starting sequential analysis with {len(agent_types)} agents")
        
        for i, agent_type in enumerate(agent_types):
            agent_id = f"{agent_type.value}_agent_{i}"
            
            # Get or create agent
            agent = await self.lifecycle_manager.get_agent(agent_id)
            if not agent:
                agent = await self.lifecycle_manager.create_agent(agent_id, agent_type)
            
            # Execute agent
            try:
                state = await agent.execute_with_state(state)
                self.logger.info(f"Completed analysis with {agent_type.value} agent")
            except Exception as e:
                self.logger.error(f"Agent {agent_id} failed: {e}")
                # Continue with next agent
        
        self.logger.info("Sequential analysis completed")
        return state
    
    async def coordinate_collaborative_analysis(self, state: ResearchState, 
                                              agent_types: List[AgentType]) -> ResearchState:
        """Execute collaborative analysis where agents can communicate."""
        self.logger.info(f"Starting collaborative analysis with {len(agent_types)} agents")
        
        # Create agents
        agents = []
        for i, agent_type in enumerate(agent_types):
            agent_id = f"{agent_type.value}_agent_{i}"
            agent = await self.lifecycle_manager.get_agent(agent_id)
            if not agent:
                agent = await self.lifecycle_manager.create_agent(agent_id, agent_type)
            agents.append(agent)
        
        # Subscribe agents to collaboration topics
        for agent in agents:
            self.communication_protocol.subscribe(agent.agent_id, "collaboration")
            self.communication_protocol.subscribe(agent.agent_id, "data_sharing")
        
        # Execute first round of analysis
        initial_tasks = []
        for agent in agents:
            task = asyncio.create_task(agent.execute_with_state(state.copy()))
            initial_tasks.append(task)
        
        initial_results = await asyncio.gather(*initial_tasks, return_exceptions=True)
        
        # Process initial results and enable collaboration
        for i, result in enumerate(initial_results):
            if not isinstance(result, Exception):
                state = self._merge_states(state, result)
                
                # Share insights with other agents
                agent_result = state.agent_results.get(agents[i].agent_id)
                if agent_result and agent_result.insights:
                    await self.communication_protocol.broadcast_message(
                        agents[i].agent_id,
                        "collaboration",
                        f"分享见解: {'; '.join(agent_result.insights[:3])}",
                        {"type": "insight_sharing", "agent_type": agents[i].agent_type.value}
                    )
        
        # Second round with collaboration
        collaboration_tasks = []
        for agent in agents:
            # Get messages from other agents
            messages = self.communication_protocol.get_messages(agent.agent_id)
            if messages:
                # Add messages to state for this agent
                for msg in messages:
                    state.add_message(msg)
            
            task = asyncio.create_task(agent.execute_with_state(state.copy()))
            collaboration_tasks.append(task)
        
        collaboration_results = await asyncio.gather(*collaboration_tasks, return_exceptions=True)
        
        # Merge final results
        for i, result in enumerate(collaboration_results):
            if not isinstance(result, Exception):
                state = self._merge_states(state, result)
        
        self.logger.info("Collaborative analysis completed")
        return state
    
    def _merge_states(self, main_state: ResearchState, agent_state: ResearchState) -> ResearchState:
        """Merge agent state results into main state."""
        # Merge messages
        for message in agent_state.messages:
            if message not in main_state.messages:
                main_state.add_message(message)
        
        # Merge tool calls and results
        for call in agent_state.tool_calls:
            if call.id not in [c.id for c in main_state.tool_calls]:
                main_state.add_tool_call(call)
        
        for call_id, result in agent_state.tool_results.items():
            if call_id not in main_state.tool_results:
                main_state.tool_results[call_id] = result
        
        # Merge agent results
        for agent_id, result in agent_state.agent_results.items():
            main_state.add_agent_result(result)
        
        # Merge collected data
        for key, data in agent_state.collected_data.items():
            if key not in main_state.collected_data:
                main_state.collected_data[key] = data
        
        return main_state


# Global instances
agent_lifecycle_manager = AgentLifecycleManager()
agent_communication_protocol = AgentCommunicationProtocol()
agent_coordinator = AgentCoordinator(agent_lifecycle_manager, agent_communication_protocol)