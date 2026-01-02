"""Enhanced workflow management system for multi-agent collaboration."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
import json

from .state import ResearchState, AgentStatus, state_manager
from .task_coordinator import TaskCoordinator, ResearchTask
from ..agents.base import BaseAgent

logger = logging.getLogger(__name__)


class WorkflowNodeType(str, Enum):
    """Workflow node types."""
    START = "start"
    END = "end"
    AGENT = "agent"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CONDITION = "condition"
    MERGE = "merge"
    TOOL = "tool"
    HUMAN = "human"


class ExecutionMode(str, Enum):
    """Execution modes for workflow nodes."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow graph."""
    node_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    node_type: WorkflowNodeType = WorkflowNodeType.AGENT
    
    # Node configuration
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    condition_func: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies and flow control
    dependencies: Set[str] = field(default_factory=set)
    next_nodes: Set[str] = field(default_factory=set)
    
    # Execution settings
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    max_retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # Status tracking
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Any = None
    
    def is_ready(self, completed_nodes: Set[str]) -> bool:
        """Check if node is ready to execute (all dependencies completed)."""
        return self.dependencies.issubset(completed_nodes)
    
    def get_execution_time(self) -> Optional[timedelta]:
        """Get node execution time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class WorkflowDefinition:
    """Defines a complete workflow structure."""
    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    
    # Workflow structure
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    start_node: Optional[str] = None
    end_nodes: Set[str] = field(default_factory=set)
    
    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    global_timeout: timedelta = field(default_factory=lambda: timedelta(hours=2))
    max_concurrent_nodes: int = 10
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    
    def add_node(self, node: WorkflowNode) -> str:
        """Add a node to the workflow."""
        self.nodes[node.node_id] = node
        return node.node_id
    
    def add_dependency(self, node_id: str, dependency_id: str):
        """Add a dependency between nodes."""
        if node_id in self.nodes and dependency_id in self.nodes:
            self.nodes[node_id].dependencies.add(dependency_id)
            self.nodes[dependency_id].next_nodes.add(node_id)
    
    def validate(self) -> List[str]:
        """Validate workflow definition."""
        errors = []
        
        # Check for start node
        if not self.start_node or self.start_node not in self.nodes:
            errors.append("No valid start node defined")
        
        # Check for end nodes
        if not self.end_nodes:
            errors.append("No end nodes defined")
        
        # Check for cycles
        if self._has_cycles():
            errors.append("Workflow contains cycles")
        
        # Check node references
        for node in self.nodes.values():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    errors.append(f"Node {node.node_id} references unknown dependency {dep_id}")
        
        return errors
    
    def _has_cycles(self) -> bool:
        """Check for cycles in the workflow graph."""
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for next_node in self.nodes[node_id].next_nodes:
                if next_node not in visited:
                    if has_cycle_util(next_node):
                        return True
                elif next_node in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    return True
        
        return False


@dataclass
class WorkflowExecution:
    """Tracks execution of a workflow instance."""
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_id: str = ""
    task_id: Optional[str] = None
    
    # Execution state
    status: str = "created"
    current_nodes: Set[str] = field(default_factory=set)
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    
    # Results and context
    context: Dict[str, Any] = field(default_factory=dict)
    node_results: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def get_execution_time(self) -> Optional[timedelta]:
        """Get total execution time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.utcnow() - self.started_at
        return None


class WorkflowManager:
    """Enhanced workflow management system."""
    
    def __init__(self, task_coordinator: TaskCoordinator):
        self.task_coordinator = task_coordinator
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        self._lock = asyncio.Lock()
        
        # Built-in workflow templates
        self._create_default_workflows()
    
    def _create_default_workflows(self):
        """Create default workflow templates."""
        # Standard investment research workflow
        research_workflow = self._create_investment_research_workflow()
        self.workflow_definitions[research_workflow.workflow_id] = research_workflow
        
        # Quick analysis workflow
        quick_workflow = self._create_quick_analysis_workflow()
        self.workflow_definitions[quick_workflow.workflow_id] = quick_workflow
        
        # Deep dive workflow
        deep_dive_workflow = self._create_deep_dive_workflow()
        self.workflow_definitions[deep_dive_workflow.workflow_id] = deep_dive_workflow
    
    def _create_investment_research_workflow(self) -> WorkflowDefinition:
        """Create standard investment research workflow."""
        workflow = WorkflowDefinition(
            name="Investment Research Workflow",
            description="Standard multi-agent investment research process",
            execution_mode=ExecutionMode.PARALLEL
        )
        
        # Start node
        start_node = WorkflowNode(
            name="Start Research",
            node_type=WorkflowNodeType.START
        )
        workflow.add_node(start_node)
        workflow.start_node = start_node.node_id
        
        # Data collection phase
        data_collection = WorkflowNode(
            name="Data Collection",
            node_type=WorkflowNodeType.TOOL,
            tool_name="data_collector",
            parameters={"collect_all": True}
        )
        workflow.add_node(data_collection)
        workflow.add_dependency(data_collection.node_id, start_node.node_id)
        
        # Parallel agent analysis
        industry_analysis = WorkflowNode(
            name="Industry Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="industry_agent"
        )
        workflow.add_node(industry_analysis)
        workflow.add_dependency(industry_analysis.node_id, data_collection.node_id)
        
        financial_analysis = WorkflowNode(
            name="Financial Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="financial_agent"
        )
        workflow.add_node(financial_analysis)
        workflow.add_dependency(financial_analysis.node_id, data_collection.node_id)
        
        market_analysis = WorkflowNode(
            name="Market Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="market_agent"
        )
        workflow.add_node(market_analysis)
        workflow.add_dependency(market_analysis.node_id, data_collection.node_id)
        
        risk_analysis = WorkflowNode(
            name="Risk Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="risk_agent"
        )
        workflow.add_node(risk_analysis)
        workflow.add_dependency(risk_analysis.node_id, data_collection.node_id)
        
        # Synthesis phase
        synthesis = WorkflowNode(
            name="Result Synthesis",
            node_type=WorkflowNodeType.MERGE,
            parameters={"synthesis_type": "comprehensive"}
        )
        workflow.add_node(synthesis)
        workflow.add_dependency(synthesis.node_id, industry_analysis.node_id)
        workflow.add_dependency(synthesis.node_id, financial_analysis.node_id)
        workflow.add_dependency(synthesis.node_id, market_analysis.node_id)
        workflow.add_dependency(synthesis.node_id, risk_analysis.node_id)
        
        # Report generation
        report_generation = WorkflowNode(
            name="Report Generation",
            node_type=WorkflowNodeType.TOOL,
            tool_name="report_generator",
            parameters={"format": "comprehensive"}
        )
        workflow.add_node(report_generation)
        workflow.add_dependency(report_generation.node_id, synthesis.node_id)
        
        # End node
        end_node = WorkflowNode(
            name="Complete Research",
            node_type=WorkflowNodeType.END
        )
        workflow.add_node(end_node)
        workflow.add_dependency(end_node.node_id, report_generation.node_id)
        workflow.end_nodes.add(end_node.node_id)
        
        return workflow
    
    def _create_quick_analysis_workflow(self) -> WorkflowDefinition:
        """Create quick analysis workflow."""
        workflow = WorkflowDefinition(
            name="Quick Analysis Workflow",
            description="Fast analysis for urgent requests",
            execution_mode=ExecutionMode.SEQUENTIAL,
            global_timeout=timedelta(minutes=30)
        )
        
        # Start
        start_node = WorkflowNode(name="Start", node_type=WorkflowNodeType.START)
        workflow.add_node(start_node)
        workflow.start_node = start_node.node_id
        
        # Quick financial analysis
        financial_quick = WorkflowNode(
            name="Quick Financial Check",
            node_type=WorkflowNodeType.AGENT,
            agent_id="financial_agent",
            parameters={"mode": "quick"},
            timeout=timedelta(minutes=10)
        )
        workflow.add_node(financial_quick)
        workflow.add_dependency(financial_quick.node_id, start_node.node_id)
        
        # Quick risk assessment
        risk_quick = WorkflowNode(
            name="Quick Risk Assessment",
            node_type=WorkflowNodeType.AGENT,
            agent_id="risk_agent",
            parameters={"mode": "quick"},
            timeout=timedelta(minutes=10)
        )
        workflow.add_node(risk_quick)
        workflow.add_dependency(risk_quick.node_id, financial_quick.node_id)
        
        # Quick report
        quick_report = WorkflowNode(
            name="Quick Report",
            node_type=WorkflowNodeType.TOOL,
            tool_name="report_generator",
            parameters={"format": "summary"},
            timeout=timedelta(minutes=5)
        )
        workflow.add_node(quick_report)
        workflow.add_dependency(quick_report.node_id, risk_quick.node_id)
        
        # End
        end_node = WorkflowNode(name="Complete", node_type=WorkflowNodeType.END)
        workflow.add_node(end_node)
        workflow.add_dependency(end_node.node_id, quick_report.node_id)
        workflow.end_nodes.add(end_node.node_id)
        
        return workflow
    
    def _create_deep_dive_workflow(self) -> WorkflowDefinition:
        """Create deep dive analysis workflow."""
        workflow = WorkflowDefinition(
            name="Deep Dive Analysis Workflow",
            description="Comprehensive analysis with multiple iterations",
            execution_mode=ExecutionMode.COLLABORATIVE,
            global_timeout=timedelta(hours=4)
        )
        
        # Start
        start_node = WorkflowNode(name="Start", node_type=WorkflowNodeType.START)
        workflow.add_node(start_node)
        workflow.start_node = start_node.node_id
        
        # Extended data collection
        extended_data = WorkflowNode(
            name="Extended Data Collection",
            node_type=WorkflowNodeType.TOOL,
            tool_name="data_collector",
            parameters={"mode": "comprehensive", "historical_depth": 5},
            timeout=timedelta(minutes=30)
        )
        workflow.add_node(extended_data)
        workflow.add_dependency(extended_data.node_id, start_node.node_id)
        
        # First round of analysis (parallel)
        industry_deep = WorkflowNode(
            name="Deep Industry Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="industry_agent",
            parameters={"mode": "deep", "include_competitors": True},
            timeout=timedelta(minutes=45)
        )
        workflow.add_node(industry_deep)
        workflow.add_dependency(industry_deep.node_id, extended_data.node_id)
        
        financial_deep = WorkflowNode(
            name="Deep Financial Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="financial_agent",
            parameters={"mode": "deep", "include_modeling": True},
            timeout=timedelta(minutes=45)
        )
        workflow.add_node(financial_deep)
        workflow.add_dependency(financial_deep.node_id, extended_data.node_id)
        
        # Cross-validation phase
        cross_validation = WorkflowNode(
            name="Cross Validation",
            node_type=WorkflowNodeType.MERGE,
            parameters={"validation_type": "cross_check"}
        )
        workflow.add_node(cross_validation)
        workflow.add_dependency(cross_validation.node_id, industry_deep.node_id)
        workflow.add_dependency(cross_validation.node_id, financial_deep.node_id)
        
        # Second round with market and risk
        market_deep = WorkflowNode(
            name="Deep Market Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="market_agent",
            parameters={"mode": "deep", "include_forecasting": True},
            timeout=timedelta(minutes=45)
        )
        workflow.add_node(market_deep)
        workflow.add_dependency(market_deep.node_id, cross_validation.node_id)
        
        risk_deep = WorkflowNode(
            name="Deep Risk Analysis",
            node_type=WorkflowNodeType.AGENT,
            agent_id="risk_agent",
            parameters={"mode": "deep", "include_scenarios": True},
            timeout=timedelta(minutes=45)
        )
        workflow.add_node(risk_deep)
        workflow.add_dependency(risk_deep.node_id, cross_validation.node_id)
        
        # Final synthesis
        final_synthesis = WorkflowNode(
            name="Final Synthesis",
            node_type=WorkflowNodeType.MERGE,
            parameters={"synthesis_type": "comprehensive_with_validation"}
        )
        workflow.add_node(final_synthesis)
        workflow.add_dependency(final_synthesis.node_id, market_deep.node_id)
        workflow.add_dependency(final_synthesis.node_id, risk_deep.node_id)
        
        # Comprehensive report
        comprehensive_report = WorkflowNode(
            name="Comprehensive Report",
            node_type=WorkflowNodeType.TOOL,
            tool_name="report_generator",
            parameters={"format": "comprehensive", "include_appendices": True},
            timeout=timedelta(minutes=20)
        )
        workflow.add_node(comprehensive_report)
        workflow.add_dependency(comprehensive_report.node_id, final_synthesis.node_id)
        
        # End
        end_node = WorkflowNode(name="Complete", node_type=WorkflowNodeType.END)
        workflow.add_node(end_node)
        workflow.add_dependency(end_node.node_id, comprehensive_report.node_id)
        workflow.end_nodes.add(end_node.node_id)
        
        return workflow
    
    # Workflow Definition Management
    
    def create_workflow(self, definition: WorkflowDefinition) -> str:
        """Create a new workflow definition."""
        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid workflow definition: {errors}")
        
        self.workflow_definitions[definition.workflow_id] = definition
        logger.info(f"Created workflow: {definition.name} ({definition.workflow_id})")
        return definition.workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID."""
        return self.workflow_definitions.get(workflow_id)
    
    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all workflow definitions."""
        return list(self.workflow_definitions.values())
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow definition."""
        if workflow_id in self.workflow_definitions:
            del self.workflow_definitions[workflow_id]
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        return False
    
    # Workflow Execution
    
    async def execute_workflow(
        self,
        workflow_id: str,
        task_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            task_id=task_id,
            context=context or {},
            status="running",
            started_at=datetime.utcnow()
        )
        
        async with self._lock:
            self.active_executions[execution.execution_id] = execution
        
        try:
            logger.info(f"Starting workflow execution: {execution.execution_id}")
            
            # Execute workflow nodes
            await self._execute_workflow_nodes(workflow, execution)
            
            # Mark as completed
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            
            logger.info(f"Workflow execution completed: {execution.execution_id}")
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution failed: {execution.execution_id} - {e}")
        
        finally:
            # Move to history
            async with self._lock:
                if execution.execution_id in self.active_executions:
                    del self.active_executions[execution.execution_id]
                self.execution_history.append(execution)
        
        return execution
    
    async def _execute_workflow_nodes(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution
    ):
        """Execute workflow nodes according to dependencies."""
        if not workflow.start_node:
            raise ValueError("Workflow has no start node")
        
        # Initialize with start node
        ready_nodes = {workflow.start_node}
        
        while ready_nodes and execution.status == "running":
            # Determine execution strategy
            if workflow.execution_mode == ExecutionMode.PARALLEL:
                # Execute all ready nodes in parallel
                current_batch = list(ready_nodes)
                ready_nodes.clear()
                
                # Limit concurrent execution
                batch_size = min(len(current_batch), workflow.max_concurrent_nodes)
                for i in range(0, len(current_batch), batch_size):
                    batch = current_batch[i:i + batch_size]
                    await self._execute_node_batch(workflow, execution, batch)
            
            elif workflow.execution_mode == ExecutionMode.SEQUENTIAL:
                # Execute one node at a time
                node_id = ready_nodes.pop()
                await self._execute_single_node(workflow, execution, node_id)
            
            elif workflow.execution_mode == ExecutionMode.COLLABORATIVE:
                # Execute with collaboration between agents
                await self._execute_collaborative_batch(workflow, execution, ready_nodes)
                ready_nodes.clear()
            
            # Find next ready nodes
            for node_id, node in workflow.nodes.items():
                if (node_id not in execution.completed_nodes and 
                    node_id not in execution.failed_nodes and
                    node_id not in execution.current_nodes and
                    node.is_ready(execution.completed_nodes)):
                    ready_nodes.add(node_id)
            
            # Check for completion
            if execution.completed_nodes.intersection(workflow.end_nodes):
                break
        
        # Verify completion
        if not execution.completed_nodes.intersection(workflow.end_nodes):
            raise RuntimeError("Workflow did not reach any end node")
    
    async def _execute_node_batch(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        node_ids: List[str]
    ):
        """Execute a batch of nodes in parallel."""
        tasks = []
        for node_id in node_ids:
            task = asyncio.create_task(
                self._execute_single_node(workflow, execution, node_id)
            )
            tasks.append(task)
        
        # Wait for all nodes to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_collaborative_batch(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        node_ids: Set[str]
    ):
        """Execute nodes with collaboration between agents."""
        # Group agent nodes for collaboration
        agent_nodes = []
        other_nodes = []
        
        for node_id in node_ids:
            node = workflow.nodes[node_id]
            if node.node_type == WorkflowNodeType.AGENT:
                agent_nodes.append(node_id)
            else:
                other_nodes.append(node_id)
        
        # Execute non-agent nodes first
        if other_nodes:
            await self._execute_node_batch(workflow, execution, other_nodes)
        
        # Execute agent nodes with collaboration
        if agent_nodes:
            await self._execute_collaborative_agents(workflow, execution, agent_nodes)
    
    async def _execute_collaborative_agents(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        agent_node_ids: List[str]
    ):
        """Execute agent nodes with collaboration."""
        # This would implement agent collaboration logic
        # For now, execute in parallel with shared context
        
        # Create shared context for collaboration
        shared_context = execution.context.copy()
        shared_context["collaboration_mode"] = True
        shared_context["peer_agents"] = [
            workflow.nodes[node_id].agent_id 
            for node_id in agent_node_ids
            if workflow.nodes[node_id].agent_id
        ]
        
        # Execute agents with shared context
        tasks = []
        for node_id in agent_node_ids:
            # Update node parameters with shared context
            node = workflow.nodes[node_id]
            node.parameters.update(shared_context)
            
            task = asyncio.create_task(
                self._execute_single_node(workflow, execution, node_id)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_node(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution,
        node_id: str
    ):
        """Execute a single workflow node."""
        node = workflow.nodes[node_id]
        execution.current_nodes.add(node_id)
        
        try:
            logger.info(f"Executing node: {node.name} ({node_id})")
            node.started_at = datetime.utcnow()
            node.status = "running"
            
            # Execute based on node type
            if node.node_type == WorkflowNodeType.START:
                result = {"status": "started", "timestamp": datetime.utcnow().isoformat()}
            
            elif node.node_type == WorkflowNodeType.END:
                result = {"status": "completed", "timestamp": datetime.utcnow().isoformat()}
            
            elif node.node_type == WorkflowNodeType.AGENT:
                result = await self._execute_agent_node(node, execution)
            
            elif node.node_type == WorkflowNodeType.TOOL:
                result = await self._execute_tool_node(node, execution)
            
            elif node.node_type == WorkflowNodeType.MERGE:
                result = await self._execute_merge_node(node, execution)
            
            elif node.node_type == WorkflowNodeType.CONDITION:
                result = await self._execute_condition_node(node, execution)
            
            else:
                result = {"status": "skipped", "reason": f"Unknown node type: {node.node_type}"}
            
            # Mark as completed
            node.status = "completed"
            node.completed_at = datetime.utcnow()
            node.result = result
            
            execution.completed_nodes.add(node_id)
            execution.node_results[node_id] = result
            
            logger.info(f"Node completed: {node.name} ({node_id})")
            
        except Exception as e:
            node.status = "failed"
            node.error_message = str(e)
            node.completed_at = datetime.utcnow()
            
            execution.failed_nodes.add(node_id)
            logger.error(f"Node failed: {node.name} ({node_id}) - {e}")
            
            # Decide whether to fail the entire workflow
            if node.node_type in [WorkflowNodeType.START, WorkflowNodeType.END]:
                raise  # Critical nodes must succeed
        
        finally:
            execution.current_nodes.discard(node_id)
    
    async def _execute_agent_node(self, node: WorkflowNode, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute an agent node."""
        if not node.agent_id:
            raise ValueError(f"Agent node {node.node_id} has no agent_id")
        
        agent = self.task_coordinator.get_agent(node.agent_id)
        if not agent:
            raise ValueError(f"Agent {node.agent_id} not found")
        
        # Get research state
        if execution.task_id:
            state = await state_manager.get_state(execution.task_id)
            if not state:
                raise ValueError(f"Research state {execution.task_id} not found")
        else:
            # Create temporary state
            state = await state_manager.create_state(
                topic=execution.context.get("topic", "Workflow Execution"),
                parameters=execution.context
            )
        
        # Update state with node parameters
        state.parameters.update(node.parameters)
        
        # Execute agent
        result_state = await agent.execute_with_state(state)
        
        # Extract results
        agent_result = result_state.agent_results.get(node.agent_id)
        if agent_result:
            return {
                "agent_id": node.agent_id,
                "status": agent_result.status.value,
                "insights": agent_result.insights,
                "confidence_score": agent_result.confidence_score,
                "data_sources": agent_result.data_sources,
                "content": agent_result.content
            }
        else:
            return {"status": "no_result", "agent_id": node.agent_id}
    
    async def _execute_tool_node(self, node: WorkflowNode, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a tool node."""
        # This would integrate with the tool execution system
        return {
            "tool_name": node.tool_name,
            "status": "executed",
            "parameters": node.parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_merge_node(self, node: WorkflowNode, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a merge node to combine results."""
        # Collect results from dependency nodes
        dependency_results = {}
        for dep_id in node.dependencies:
            if dep_id in execution.node_results:
                dependency_results[dep_id] = execution.node_results[dep_id]
        
        # Perform merge based on parameters
        merge_type = node.parameters.get("synthesis_type", "basic")
        
        if merge_type == "comprehensive":
            # Comprehensive synthesis
            all_insights = []
            confidence_scores = []
            data_sources = set()
            
            for result in dependency_results.values():
                if isinstance(result, dict):
                    if "insights" in result:
                        all_insights.extend(result["insights"])
                    if "confidence_score" in result:
                        confidence_scores.append(result["confidence_score"])
                    if "data_sources" in result:
                        data_sources.update(result["data_sources"])
            
            return {
                "merge_type": merge_type,
                "total_insights": len(all_insights),
                "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "unique_data_sources": len(data_sources),
                "synthesized_insights": all_insights[:10],  # Top 10 insights
                "dependency_count": len(dependency_results)
            }
        
        else:
            # Basic merge
            return {
                "merge_type": "basic",
                "dependency_results": dependency_results,
                "merged_at": datetime.utcnow().isoformat()
            }
    
    async def _execute_condition_node(self, node: WorkflowNode, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a condition node."""
        if not node.condition_func:
            return {"status": "no_condition", "result": True}
        
        try:
            result = node.condition_func(execution.context, execution.node_results)
            return {"status": "evaluated", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e), "result": False}
    
    # Monitoring and Management
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        # Check active executions
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    def list_active_executions(self) -> List[WorkflowExecution]:
        """List all active workflow executions."""
        return list(self.active_executions.values())
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed execution status."""
        execution = self.get_execution(execution_id)
        if not execution:
            return None
        
        workflow = self.get_workflow(execution.workflow_id)
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "workflow_name": workflow.name if workflow else "Unknown",
            "status": execution.status,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_time": execution.get_execution_time().total_seconds() if execution.get_execution_time() else None,
            "current_nodes": list(execution.current_nodes),
            "completed_nodes": list(execution.completed_nodes),
            "failed_nodes": list(execution.failed_nodes),
            "progress": len(execution.completed_nodes) / len(workflow.nodes) if workflow else 0,
            "error_message": execution.error_message
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active workflow execution."""
        async with self._lock:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = "cancelled"
                execution.completed_at = datetime.utcnow()
                
                # Move to history
                del self.active_executions[execution_id]
                self.execution_history.append(execution)
                
                logger.info(f"Cancelled workflow execution: {execution_id}")
                return True
        
        return False


# Global workflow manager instance
workflow_manager = None

def get_workflow_manager(task_coordinator: TaskCoordinator) -> WorkflowManager:
    """Get or create global workflow manager instance."""
    global workflow_manager
    if workflow_manager is None:
        workflow_manager = WorkflowManager(task_coordinator)
    return workflow_manager