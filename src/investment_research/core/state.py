"""Global state management for multi-agent research system."""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor


class MessageRole(str, Enum):
    """Message role enumeration."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"


class AgentStatus(str, Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class Message:
    """Message in the conversation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    agent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result from an agent's analysis."""
    agent_id: str
    agent_type: str
    status: AgentStatus
    content: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolCall:
    """Tool call request."""
    id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolResult:
    """Tool call result."""
    call_id: str
    tool_name: str
    success: bool = True
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResearchState:
    """Global state for research workflow."""
    
    # Basic information
    task_id: str = field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation history
    messages: List[Message] = field(default_factory=list)
    
    # Agent management
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    agent_status: Dict[str, AgentStatus] = field(default_factory=dict)
    current_agent: Optional[str] = None
    
    # Data and analysis
    collected_data: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    
    # Tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: Dict[str, ToolResult] = field(default_factory=dict)
    
    # Workflow control
    current_step: str = "initialization"
    completed_steps: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    
    # Synchronization
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update agent status."""
        self.agent_status[agent_id] = status
        self.updated_at = datetime.utcnow()
    
    def add_agent_result(self, result: AgentResult) -> None:
        """Add agent analysis result."""
        self.agent_results[result.agent_id] = result
        self.agent_status[result.agent_id] = result.status
        self.updated_at = datetime.utcnow()
    
    def add_tool_call(self, tool_call: ToolCall) -> str:
        """Add a tool call and return its ID."""
        self.tool_calls.append(tool_call)
        self.updated_at = datetime.utcnow()
        return tool_call.id
    
    def add_tool_result(self, result: ToolResult) -> None:
        """Add tool call result."""
        self.tool_results[result.call_id] = result
        self.updated_at = datetime.utcnow()
    
    def get_agent_results(self, agent_type: Optional[str] = None) -> List[AgentResult]:
        """Get agent results, optionally filtered by type."""
        results = list(self.agent_results.values())
        if agent_type:
            results = [r for r in results if r.agent_type == agent_type]
        return results
    
    def get_completed_agents(self) -> List[str]:
        """Get list of completed agents."""
        return [
            agent_id for agent_id, status in self.agent_status.items()
            if status == AgentStatus.COMPLETED
        ]
    
    def get_failed_agents(self) -> List[str]:
        """Get list of failed agents."""
        return [
            agent_id for agent_id, status in self.agent_status.items()
            if status == AgentStatus.FAILED
        ]
    
    def is_all_agents_completed(self, required_agents: List[str]) -> bool:
        """Check if all required agents have completed."""
        for agent_id in required_agents:
            if self.agent_status.get(agent_id) != AgentStatus.COMPLETED:
                return False
        return True
    
    def get_data_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get relevant data for a specific agent type."""
        # Return data relevant to the agent type
        agent_data = {
            "topic": self.topic,
            "parameters": self.parameters,
            "collected_data": self.collected_data,
            "previous_results": {}
        }
        
        # Add results from other agents
        for agent_id, result in self.agent_results.items():
            if result.status == AgentStatus.COMPLETED:
                agent_data["previous_results"][result.agent_type] = result.content
        
        return agent_data
    
    async def wait_for_agents(self, agent_ids: List[str], timeout: float = 300.0) -> bool:
        """Wait for specific agents to complete."""
        start_time = datetime.utcnow()
        
        while True:
            if self.is_all_agents_completed(agent_ids):
                return True
            
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                return False
            
            # Wait a bit before checking again
            await asyncio.sleep(1.0)


class StateManager:
    """Manager for research state with thread-safe operations."""
    
    def __init__(self):
        self._states: Dict[str, ResearchState] = {}
        self._lock = asyncio.Lock()
    
    async def create_state(self, topic: str, parameters: Dict[str, Any] = None) -> ResearchState:
        """Create a new research state."""
        async with self._lock:
            state = ResearchState(
                topic=topic,
                parameters=parameters or {}
            )
            self._states[state.task_id] = state
            return state
    
    async def get_state(self, task_id: str) -> Optional[ResearchState]:
        """Get research state by task ID."""
        async with self._lock:
            return self._states.get(task_id)
    
    async def update_state(self, task_id: str, state: ResearchState) -> None:
        """Update research state."""
        async with self._lock:
            if task_id in self._states:
                self._states[task_id] = state
    
    async def delete_state(self, task_id: str) -> bool:
        """Delete research state."""
        async with self._lock:
            if task_id in self._states:
                del self._states[task_id]
                return True
            return False
    
    async def list_states(self) -> List[str]:
        """List all state task IDs."""
        async with self._lock:
            return list(self._states.keys())


# Global state manager instance
state_manager = StateManager()