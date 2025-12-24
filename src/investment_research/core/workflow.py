"""Workflow orchestration for multi-agent research system."""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from .state import ResearchState, AgentStatus, state_manager
from ..agents.base import BaseAgent
from ..core.models import AgentType


logger = logging.getLogger(__name__)


class WorkflowStep(str, Enum):
    """Workflow step enumeration."""
    INITIALIZATION = "initialization"
    DATA_COLLECTION = "data_collection"
    PARALLEL_ANALYSIS = "parallel_analysis"
    SYNTHESIS = "synthesis"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    required_agents: List[str]
    parallel_execution: bool = True
    timeout_per_step: float = 300.0  # 5 minutes
    max_retries: int = 2
    enable_synthesis: bool = True


class WorkflowOrchestrator:
    """Orchestrates multi-agent research workflow similar to LangGraph."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.step_handlers: Dict[WorkflowStep, Callable] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default step handlers."""
        self.step_handlers = {
            WorkflowStep.INITIALIZATION: self._handle_initialization,
            WorkflowStep.DATA_COLLECTION: self._handle_data_collection,
            WorkflowStep.PARALLEL_ANALYSIS: self._handle_parallel_analysis,
            WorkflowStep.SYNTHESIS: self._handle_synthesis,
            WorkflowStep.REPORT_GENERATION: self._handle_report_generation,
        }
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    async def execute_workflow(self, state: ResearchState) -> ResearchState:
        """Execute the complete research workflow."""
        logger.info(f"Starting workflow for task: {state.task_id}")
        
        try:
            # Execute workflow steps in sequence
            steps = [
                WorkflowStep.INITIALIZATION,
                WorkflowStep.DATA_COLLECTION,
                WorkflowStep.PARALLEL_ANALYSIS,
                WorkflowStep.SYNTHESIS,
                WorkflowStep.REPORT_GENERATION
            ]
            
            for step in steps:
                logger.info(f"Executing step: {step.value}")
                state.current_step = step.value
                
                # Execute step with timeout
                try:
                    handler = self.step_handlers.get(step)
                    if handler:
                        state = await asyncio.wait_for(
                            handler(state),
                            timeout=self.config.timeout_per_step
                        )
                        state.completed_steps.append(step.value)
                    else:
                        logger.warning(f"No handler for step: {step.value}")
                
                except asyncio.TimeoutError:
                    logger.error(f"Step {step.value} timed out")
                    state.current_step = WorkflowStep.FAILED.value
                    return state
                
                except Exception as e:
                    logger.error(f"Step {step.value} failed: {e}")
                    state.current_step = WorkflowStep.FAILED.value
                    return state
            
            # Mark as completed
            state.current_step = WorkflowStep.COMPLETED.value
            logger.info(f"Workflow completed for task: {state.task_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state.current_step = WorkflowStep.FAILED.value
            return state
    
    async def _handle_initialization(self, state: ResearchState) -> ResearchState:
        """Handle initialization step."""
        logger.info("Initializing research workflow")
        
        # Initialize agent statuses
        for agent_id in self.config.required_agents:
            state.update_agent_status(agent_id, AgentStatus.PENDING)
        
        # Set next steps
        state.next_steps = [WorkflowStep.DATA_COLLECTION.value]
        
        return state
    
    async def _handle_data_collection(self, state: ResearchState) -> ResearchState:
        """Handle data collection step."""
        logger.info("Starting data collection phase")
        
        # For now, we'll collect data as part of agent analysis
        # In a more complex system, this could be a separate step
        state.next_steps = [WorkflowStep.PARALLEL_ANALYSIS.value]
        
        return state
    
    async def _handle_parallel_analysis(self, state: ResearchState) -> ResearchState:
        """Handle parallel analysis by multiple agents."""
        logger.info("Starting parallel agent analysis")
        
        if self.config.parallel_execution:
            # Execute agents in parallel
            tasks = []
            for agent_id in self.config.required_agents:
                agent = self.get_agent(agent_id)
                if agent:
                    task = asyncio.create_task(
                        agent.execute_with_state(state),
                        name=f"agent_{agent_id}"
                    )
                    tasks.append(task)
                else:
                    logger.warning(f"Agent {agent_id} not found")
            
            if tasks:
                # Wait for all agents to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results and exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        agent_id = self.config.required_agents[i]
                        logger.error(f"Agent {agent_id} failed: {result}")
                        state.update_agent_status(agent_id, AgentStatus.FAILED)
                    else:
                        # Update state with the latest version
                        state = result
        
        else:
            # Execute agents sequentially
            for agent_id in self.config.required_agents:
                agent = self.get_agent(agent_id)
                if agent:
                    state = await agent.execute_with_state(state)
                else:
                    logger.warning(f"Agent {agent_id} not found")
        
        # Check if we have enough successful agents
        completed_agents = state.get_completed_agents()
        if len(completed_agents) < len(self.config.required_agents) // 2:
            logger.warning("Less than half of agents completed successfully")
        
        state.next_steps = [WorkflowStep.SYNTHESIS.value]
        return state
    
    async def _handle_synthesis(self, state: ResearchState) -> ResearchState:
        """Handle synthesis of agent results."""
        logger.info("Starting result synthesis")
        
        if not self.config.enable_synthesis:
            state.next_steps = [WorkflowStep.REPORT_GENERATION.value]
            return state
        
        # Get completed agent results
        completed_results = [
            result for result in state.agent_results.values()
            if result.status == AgentStatus.COMPLETED
        ]
        
        if not completed_results:
            logger.warning("No completed agent results to synthesize")
            state.next_steps = [WorkflowStep.REPORT_GENERATION.value]
            return state
        
        # Synthesize insights
        all_insights = []
        for result in completed_results:
            all_insights.extend(result.insights)
        
        # Store synthesized results
        state.analysis_results["synthesis"] = {
            "total_insights": len(all_insights),
            "agent_count": len(completed_results),
            "key_themes": self._extract_key_themes(all_insights),
            "confidence_scores": [r.confidence_score for r in completed_results]
        }
        
        state.next_steps = [WorkflowStep.REPORT_GENERATION.value]
        return state
    
    async def _handle_report_generation(self, state: ResearchState) -> ResearchState:
        """Handle report generation step."""
        logger.info("Starting report generation")
        
        # For now, just mark as ready for report generation
        # The actual report generation would be handled by a separate service
        state.analysis_results["report_ready"] = True
        state.analysis_results["generation_timestamp"] = datetime.utcnow().isoformat()
        
        return state
    
    def _extract_key_themes(self, insights: List[str]) -> List[str]:
        """Extract key themes from insights (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd use NLP techniques to extract themes
        themes = []
        
        # Simple keyword-based theme extraction
        keywords = {
            "财务": ["收入", "利润", "现金流", "债务", "估值"],
            "市场": ["市场份额", "竞争", "需求", "增长", "趋势"],
            "风险": ["风险", "威胁", "挑战", "不确定性", "波动"],
            "行业": ["行业", "监管", "政策", "技术", "创新"]
        }
        
        for theme, theme_keywords in keywords.items():
            for insight in insights:
                if any(keyword in insight for keyword in theme_keywords):
                    if theme not in themes:
                        themes.append(theme)
                    break
        
        return themes[:5]  # Return top 5 themes


class ResearchWorkflow:
    """High-level research workflow manager."""
    
    def __init__(self):
        self.orchestrator = None
        self._setup_default_workflow()
    
    def _setup_default_workflow(self):
        """Setup default research workflow."""
        config = WorkflowConfig(
            required_agents=["industry_agent", "financial_agent", "market_agent", "risk_agent"],
            parallel_execution=True,
            timeout_per_step=300.0,
            enable_synthesis=True
        )
        self.orchestrator = WorkflowOrchestrator(config)
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the workflow."""
        if self.orchestrator:
            self.orchestrator.register_agent(agent)
    
    async def create_research_task(
        self, 
        topic: str, 
        parameters: Dict[str, Any] = None
    ) -> ResearchState:
        """Create a new research task."""
        state = await state_manager.create_state(topic, parameters)
        logger.info(f"Created research task: {state.task_id}")
        return state
    
    async def execute_research(self, task_id: str) -> Optional[ResearchState]:
        """Execute research workflow for a task."""
        state = await state_manager.get_state(task_id)
        if not state:
            logger.error(f"Task {task_id} not found")
            return None
        
        if not self.orchestrator:
            logger.error("Workflow orchestrator not initialized")
            return None
        
        # Execute workflow
        updated_state = await self.orchestrator.execute_workflow(state)
        
        # Update state in manager
        await state_manager.update_state(task_id, updated_state)
        
        return updated_state
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a research task."""
        state = await state_manager.get_state(task_id)
        if not state:
            return None
        
        return {
            "task_id": task_id,
            "topic": state.topic,
            "current_step": state.current_step,
            "completed_steps": state.completed_steps,
            "agent_status": dict(state.agent_status),
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat()
        }


# Global workflow instance
research_workflow = ResearchWorkflow()