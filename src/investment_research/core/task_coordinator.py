"""Task coordination system for multi-agent investment research."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4

from .state import ResearchState, AgentStatus, state_manager
from .workflow import research_workflow, WorkflowStep
from ..agents.base import BaseAgent
from ..core.models import AgentType

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """Task execution status."""
    CREATED = "created"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchTask:
    """Research task definition."""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Task configuration
    priority: TaskPriority = TaskPriority.NORMAL
    required_agents: List[str] = field(default_factory=list)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    deadline: Optional[datetime] = None
    
    # Task status
    status: TaskStatus = TaskStatus.CREATED
    assigned_agents: Set[str] = field(default_factory=set)
    progress: float = 0.0
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Results
    state_id: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration(self) -> Optional[timedelta]:
        """Get task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.utcnow() - self.started_at
        return None
    
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.deadline:
            return False
        return datetime.utcnow() > self.deadline
    
    def get_priority_score(self) -> int:
        """Get numeric priority score for sorting."""
        priority_scores = {
            TaskPriority.LOW: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.URGENT: 4
        }
        score = priority_scores.get(self.priority, 2)
        
        # Boost score if overdue
        if self.is_overdue():
            score += 2
        
        return score


@dataclass
class AgentAssignment:
    """Agent assignment to a task."""
    agent_id: str
    agent_type: str
    task_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: AgentStatus = AgentStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None


class TaskCoordinator:
    """Coordinates research task creation, assignment, and monitoring."""
    
    def __init__(self):
        self.tasks: Dict[str, ResearchTask] = {}
        self.agent_assignments: Dict[str, List[AgentAssignment]] = {}
        self.available_agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[str] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._coordinator_task: Optional[asyncio.Task] = None
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator."""
        self.available_agents[agent.agent_id] = agent
        research_workflow.register_agent(agent)
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        return self.available_agents.get(agent_id)
    
    def list_available_agents(self) -> List[str]:
        """List available agent IDs."""
        return list(self.available_agents.keys())
    
    async def create_task(
        self,
        topic: str,
        description: str = "",
        parameters: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_agents: List[str] = None,
        deadline: Optional[datetime] = None
    ) -> ResearchTask:
        """Create a new research task."""
        async with self._lock:
            # Default required agents if not specified
            if required_agents is None:
                required_agents = ["industry_agent", "financial_agent", "market_agent", "risk_agent"]
            
            # Validate required agents are available
            missing_agents = [
                agent_id for agent_id in required_agents
                if agent_id not in self.available_agents
            ]
            if missing_agents:
                logger.warning(f"Missing required agents: {missing_agents}")
            
            task = ResearchTask(
                topic=topic,
                description=description,
                parameters=parameters or {},
                priority=priority,
                required_agents=required_agents,
                deadline=deadline,
                status=TaskStatus.CREATED
            )
            
            self.tasks[task.task_id] = task
            logger.info(f"Created task: {task.task_id} - {topic}")
            
            return task
    
    async def queue_task(self, task_id: str) -> bool:
        """Queue a task for execution."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return False
            
            if task.status != TaskStatus.CREATED:
                logger.warning(f"Task {task_id} is not in created state")
                return False
            
            task.status = TaskStatus.QUEUED
            self.task_queue.append(task_id)
            
            # Sort queue by priority
            self.task_queue.sort(
                key=lambda tid: self.tasks[tid].get_priority_score(),
                reverse=True
            )
            
            logger.info(f"Queued task: {task_id}")
            return True
    
    async def assign_task(self, task_id: str) -> bool:
        """Assign agents to a task."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return False
            
            if task.status != TaskStatus.QUEUED:
                logger.warning(f"Task {task_id} is not queued")
                return False
            
            # Check agent availability
            available_agents = []
            for agent_id in task.required_agents:
                if agent_id in self.available_agents:
                    # Check if agent is not busy with other tasks
                    if not self._is_agent_busy(agent_id):
                        available_agents.append(agent_id)
            
            if len(available_agents) < len(task.required_agents):
                logger.warning(f"Not enough available agents for task {task_id}")
                return False
            
            # Create assignments
            assignments = []
            for agent_id in task.required_agents:
                if agent_id in available_agents:
                    agent = self.available_agents[agent_id]
                    assignment = AgentAssignment(
                        agent_id=agent_id,
                        agent_type=agent.agent_type.value,
                        task_id=task_id
                    )
                    assignments.append(assignment)
                    task.assigned_agents.add(agent_id)
            
            # Store assignments
            self.agent_assignments[task_id] = assignments
            task.status = TaskStatus.ASSIGNED
            
            logger.info(f"Assigned task {task_id} to agents: {list(task.assigned_agents)}")
            return True
    
    async def execute_task(self, task_id: str) -> bool:
        """Execute a task using the workflow system."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return False
            
            if task.status != TaskStatus.ASSIGNED:
                logger.warning(f"Task {task_id} is not assigned")
                return False
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Update agent assignments
            if task_id in self.agent_assignments:
                for assignment in self.agent_assignments[task_id]:
                    assignment.status = AgentStatus.RUNNING
                    assignment.started_at = datetime.utcnow()
        
        try:
            # Create research state
            state = await research_workflow.create_research_task(
                topic=task.topic,
                parameters=task.parameters
            )
            task.state_id = state.task_id
            
            # Execute research workflow
            logger.info(f"Starting execution of task {task_id}")
            result_state = await research_workflow.execute_research(state.task_id)
            
            if result_state and result_state.current_step == WorkflowStep.COMPLETED.value:
                await self._complete_task(task_id, result_state)
                return True
            else:
                await self._fail_task(task_id, "Workflow execution failed")
                return False
                
        except Exception as e:
            logger.error(f"Task execution failed for {task_id}: {e}")
            await self._fail_task(task_id, str(e))
            return False
    
    async def _complete_task(self, task_id: str, state: ResearchState):
        """Mark task as completed."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 1.0
            
            # Extract results from state
            task.results = {
                "analysis_results": state.analysis_results,
                "agent_results": {
                    agent_id: {
                        "status": result.status.value,
                        "insights": result.insights,
                        "confidence_score": result.confidence_score,
                        "data_sources": result.data_sources
                    }
                    for agent_id, result in state.agent_results.items()
                },
                "completed_steps": state.completed_steps,
                "execution_time": task.get_duration().total_seconds() if task.get_duration() else 0
            }
            
            # Update agent assignments
            if task_id in self.agent_assignments:
                for assignment in self.agent_assignments[task_id]:
                    assignment.status = AgentStatus.COMPLETED
                    assignment.completed_at = datetime.utcnow()
                    assignment.progress = 1.0
            
            logger.info(f"Task {task_id} completed successfully")
    
    async def _fail_task(self, task_id: str, error_message: str):
        """Mark task as failed."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = error_message
            
            # Update agent assignments
            if task_id in self.agent_assignments:
                for assignment in self.agent_assignments[task_id]:
                    assignment.status = AgentStatus.FAILED
                    assignment.completed_at = datetime.utcnow()
                    assignment.error_message = error_message
            
            logger.error(f"Task {task_id} failed: {error_message}")
    
    def _is_agent_busy(self, agent_id: str) -> bool:
        """Check if an agent is currently busy with other tasks."""
        for assignments in self.agent_assignments.values():
            for assignment in assignments:
                if (assignment.agent_id == agent_id and 
                    assignment.status in [AgentStatus.RUNNING, AgentStatus.PENDING]):
                    return True
        return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.warning(f"Task {task_id} cannot be cancelled (status: {task.status})")
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
            # Remove from queue if present
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            # Update agent assignments
            if task_id in self.agent_assignments:
                for assignment in self.agent_assignments[task_id]:
                    if assignment.status not in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
                        assignment.status = AgentStatus.FAILED
                        assignment.completed_at = datetime.utcnow()
                        assignment.error_message = "Task cancelled"
            
            logger.info(f"Task {task_id} cancelled")
            return True
    
    # Monitoring and Status Methods
    
    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[ResearchTask]:
        """List tasks, optionally filtered by status."""
        tasks = list(self.tasks.values())
        if status:
            tasks = [task for task in tasks if task.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed task status."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        assignments = self.agent_assignments.get(task_id, [])
        
        return {
            "task_id": task.task_id,
            "topic": task.topic,
            "status": task.status.value,
            "priority": task.priority.value,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "duration": task.get_duration().total_seconds() if task.get_duration() else None,
            "is_overdue": task.is_overdue(),
            "assigned_agents": list(task.assigned_agents),
            "agent_assignments": [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "status": a.status.value,
                    "progress": a.progress,
                    "started_at": a.started_at.isoformat() if a.started_at else None,
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None
                }
                for a in assignments
            ],
            "error_message": task.error_message,
            "results_available": bool(task.results)
        }
    
    def get_agent_workload(self, agent_id: str) -> Dict[str, Any]:
        """Get agent workload information."""
        if agent_id not in self.available_agents:
            return {"error": "Agent not found"}
        
        assignments = []
        for task_assignments in self.agent_assignments.values():
            for assignment in task_assignments:
                if assignment.agent_id == agent_id:
                    assignments.append(assignment)
        
        active_assignments = [
            a for a in assignments 
            if a.status in [AgentStatus.RUNNING, AgentStatus.PENDING]
        ]
        
        return {
            "agent_id": agent_id,
            "agent_type": self.available_agents[agent_id].agent_type.value,
            "is_busy": len(active_assignments) > 0,
            "active_tasks": len(active_assignments),
            "total_assignments": len(assignments),
            "completed_assignments": len([
                a for a in assignments if a.status == AgentStatus.COMPLETED
            ]),
            "failed_assignments": len([
                a for a in assignments if a.status == AgentStatus.FAILED
            ])
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        tasks_by_status = {}
        for status in TaskStatus:
            tasks_by_status[status.value] = len([
                t for t in self.tasks.values() if t.status == status
            ])
        
        return {
            "total_tasks": len(self.tasks),
            "tasks_by_status": tasks_by_status,
            "queue_length": len(self.task_queue),
            "available_agents": len(self.available_agents),
            "busy_agents": len([
                agent_id for agent_id in self.available_agents.keys()
                if self._is_agent_busy(agent_id)
            ]),
            "overdue_tasks": len([
                t for t in self.tasks.values() if t.is_overdue()
            ])
        }
    
    # Automatic Task Processing
    
    async def start_coordinator(self):
        """Start the automatic task coordinator."""
        if self._running:
            logger.warning("Task coordinator is already running")
            return
        
        self._running = True
        self._coordinator_task = asyncio.create_task(self._coordinator_loop())
        logger.info("Task coordinator started")
    
    async def stop_coordinator(self):
        """Stop the automatic task coordinator."""
        if not self._running:
            return
        
        self._running = False
        if self._coordinator_task:
            self._coordinator_task.cancel()
            try:
                await self._coordinator_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task coordinator stopped")
    
    async def _coordinator_loop(self):
        """Main coordinator loop for automatic task processing."""
        while self._running:
            try:
                await self._process_task_queue()
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in coordinator loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _process_task_queue(self):
        """Process queued tasks automatically."""
        if not self.task_queue:
            return
        
        # Get next task from queue
        task_id = self.task_queue[0]
        task = self.tasks.get(task_id)
        
        if not task or task.status != TaskStatus.QUEUED:
            # Remove invalid task from queue
            self.task_queue.remove(task_id)
            return
        
        # Try to assign and execute task
        if await self.assign_task(task_id):
            # Remove from queue and execute
            self.task_queue.remove(task_id)
            
            # Execute task in background
            asyncio.create_task(self.execute_task(task_id))
            logger.info(f"Started background execution of task {task_id}")


# Global task coordinator instance
task_coordinator = TaskCoordinator()