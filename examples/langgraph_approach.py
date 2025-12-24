#!/usr/bin/env python3
"""Example showing LangGraph-style multi-agent architecture."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from typing import Dict, Any

from src.investment_research.core.state import ResearchState, Message, MessageRole, ToolCall
from src.investment_research.core.workflow import research_workflow
from src.investment_research.agents.base import BaseAgent
from src.investment_research.core.models import AgentType


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndustryAgent(BaseAgent):
    """Industry analysis agent implementation."""
    
    def __init__(self):
        super().__init__(
            agent_id="industry_agent",
            agent_type=AgentType.INDUSTRY
        )
    
    def get_system_prompt(self) -> str:
        return """你是一个专业的行业分析师。你的任务是分析特定行业的趋势、竞争格局和监管环境。
        请提供深入的行业洞察，包括市场动态、主要参与者和未来发展方向。"""
    
    def get_required_tools(self) -> list[str]:
        return ["mcp_search", "rag_retrieval"]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform industry analysis."""
        logger.info(f"Industry agent analyzing: {state.topic}")
        
        # Step 1: Search for industry data
        industry_search = ToolCall(
            tool_name="mcp_search",
            parameters={
                "query": f"{state.topic} 行业分析 竞争格局",
                "domain": "industry",
                "limit": 5
            }
        )
        
        # Step 2: Retrieve relevant knowledge
        rag_search = ToolCall(
            tool_name="rag_retrieval",
            parameters={
                "query": f"{state.topic} 行业趋势",
                "domain": "industry",
                "limit": 3
            }
        )
        
        # Execute tools concurrently
        state = await self.call_tools(state, [industry_search, rag_search])
        
        # Step 3: Generate analysis
        analysis_prompt = f"""
        基于收集到的数据，请分析 {state.topic} 的行业情况：
        
        1. 行业现状和规模
        2. 主要竞争对手
        3. 市场趋势和驱动因素
        4. 监管环境
        5. 未来发展前景
        
        请提供具体的数据支持和深入的洞察。
        """
        
        response = await self.generate_response(state, analysis_prompt)
        
        # Store analysis results
        state.analysis_results["industry"] = {
            "analysis": response,
            "data_sources": len([c for c in state.tool_calls if c.agent_id == self.agent_id]),
            "key_findings": ["行业增长稳定", "竞争激烈", "监管趋严"]  # Simplified
        }
        
        return state


class FinancialAgent(BaseAgent):
    """Financial analysis agent implementation."""
    
    def __init__(self):
        super().__init__(
            agent_id="financial_agent",
            agent_type=AgentType.FINANCIAL
        )
    
    def get_system_prompt(self) -> str:
        return """你是一个专业的财务分析师。你的任务是分析公司或行业的财务状况、盈利能力和估值。
        请提供详细的财务分析，包括收入、利润、现金流和关键财务比率。"""
    
    def get_required_tools(self) -> list[str]:
        return ["mcp_search", "data_validation"]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform financial analysis."""
        logger.info(f"Financial agent analyzing: {state.topic}")
        
        # Get industry context from previous analysis
        industry_context = state.analysis_results.get("industry", {})
        
        # Search for financial data
        financial_search = ToolCall(
            tool_name="mcp_search",
            parameters={
                "query": f"{state.topic} 财务数据 收入 利润",
                "domain": "financial",
                "limit": 10
            }
        )
        
        state = await self.call_tools(state, [financial_search])
        
        # Generate financial analysis
        analysis_prompt = f"""
        基于收集到的财务数据，请分析 {state.topic} 的财务状况：
        
        1. 收入增长趋势
        2. 盈利能力指标
        3. 现金流状况
        4. 债务水平和偿债能力
        5. 估值分析
        
        行业背景: {industry_context.get('key_findings', [])}
        
        请提供量化的分析和具体的财务指标。
        """
        
        response = await self.generate_response(state, analysis_prompt)
        
        # Store financial analysis
        state.analysis_results["financial"] = {
            "analysis": response,
            "metrics": {
                "revenue_growth": "15%",  # Simplified mock data
                "profit_margin": "12%",
                "debt_ratio": "0.3"
            }
        }
        
        return state


class MarketAgent(BaseAgent):
    """Market analysis agent implementation."""
    
    def __init__(self):
        super().__init__(
            agent_id="market_agent",
            agent_type=AgentType.MARKET
        )
    
    def get_system_prompt(self) -> str:
        return """你是一个专业的市场分析师。你的任务是分析市场规模、需求趋势和消费者行为。
        请提供全面的市场分析，包括市场机会、威胁和增长潜力。"""
    
    def get_required_tools(self) -> list[str]:
        return ["mcp_search", "rag_retrieval"]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform market analysis."""
        logger.info(f"Market agent analyzing: {state.topic}")
        
        # Market research
        market_search = ToolCall(
            tool_name="mcp_search",
            parameters={
                "query": f"{state.topic} 市场规模 需求分析",
                "domain": "market",
                "limit": 8
            }
        )
        
        state = await self.call_tools(state, [market_search])
        
        # Generate market analysis
        analysis_prompt = f"""
        基于市场数据，请分析 {state.topic} 的市场情况：
        
        1. 市场规模和增长率
        2. 目标客户群体
        3. 需求驱动因素
        4. 市场机会和威胁
        5. 价格趋势
        
        请结合行业和财务分析的结果。
        """
        
        response = await self.generate_response(state, analysis_prompt)
        
        state.analysis_results["market"] = {
            "analysis": response,
            "market_size": "100亿元",  # Simplified
            "growth_rate": "8%"
        }
        
        return state


class RiskAgent(BaseAgent):
    """Risk analysis agent implementation."""
    
    def __init__(self):
        super().__init__(
            agent_id="risk_agent",
            agent_type=AgentType.RISK
        )
    
    def get_system_prompt(self) -> str:
        return """你是一个专业的风险分析师。你的任务是识别和评估各种风险因素。
        请提供全面的风险分析，包括市场风险、信用风险、操作风险和合规风险。"""
    
    def get_required_tools(self) -> list[str]:
        return ["mcp_search", "data_validation"]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform risk analysis."""
        logger.info(f"Risk agent analyzing: {state.topic}")
        
        # Risk assessment
        risk_search = ToolCall(
            tool_name="mcp_search",
            parameters={
                "query": f"{state.topic} 风险因素 合规 监管",
                "domain": "risk",
                "limit": 6
            }
        )
        
        state = await self.call_tools(state, [risk_search])
        
        # Generate risk analysis
        analysis_prompt = f"""
        基于收集的信息，请分析 {state.topic} 面临的风险：
        
        1. 市场风险
        2. 信用风险
        3. 操作风险
        4. 合规风险
        5. 风险缓解建议
        
        请结合前面的行业、财务和市场分析结果。
        """
        
        response = await self.generate_response(state, analysis_prompt)
        
        state.analysis_results["risk"] = {
            "analysis": response,
            "risk_level": "中等",  # Simplified
            "key_risks": ["市场波动", "监管变化", "竞争加剧"]
        }
        
        return state


async def demonstrate_langgraph_approach():
    """Demonstrate the LangGraph-style multi-agent approach."""
    print("🚀 LangGraph风格多智能体投资研究系统演示")
    print("=" * 60)
    
    # Register agents with the workflow
    agents = [
        IndustryAgent(),
        FinancialAgent(),
        MarketAgent(),
        RiskAgent()
    ]
    
    for agent in agents:
        research_workflow.register_agent(agent)
        print(f"✅ 注册智能体: {agent.agent_id} ({agent.agent_type.value})")
    
    print()
    
    # Create research task
    print("📋 创建研究任务...")
    state = await research_workflow.create_research_task(
        topic="苹果公司(AAPL)投资分析",
        parameters={
            "analysis_depth": "comprehensive",
            "time_horizon": "12_months",
            "focus_areas": ["financial_performance", "market_position", "risk_factors"]
        }
    )
    
    print(f"   任务ID: {state.task_id}")
    print(f"   研究主题: {state.topic}")
    print()
    
    # Execute workflow
    print("🔄 执行多智能体工作流...")
    print("   注意: 这是演示模式，不会实际调用外部API")
    print()
    
    try:
        # This would normally execute the full workflow
        # For demo purposes, we'll show the structure
        print("📊 工作流步骤:")
        steps = [
            "1. 初始化 - 设置智能体状态",
            "2. 数据收集 - 准备外部数据源",
            "3. 并行分析 - 四个智能体同时工作",
            "   - 行业智能体: 分析行业趋势和竞争",
            "   - 财务智能体: 分析财务数据和估值",
            "   - 市场智能体: 分析市场规模和需求",
            "   - 风险智能体: 识别和评估风险",
            "4. 结果综合 - 整合各智能体分析",
            "5. 报告生成 - 生成最终研究报告"
        ]
        
        for step in steps:
            print(f"   {step}")
            await asyncio.sleep(0.5)  # Simulate processing time
        
        print()
        print("✅ 工作流执行完成!")
        
        # Show state structure
        print("\n📋 最终状态结构:")
        print(f"   - 消息数量: {len(state.messages)}")
        print(f"   - 智能体状态: {len(state.agent_status)} 个智能体")
        print(f"   - 工具调用: {len(state.tool_calls)} 次")
        print(f"   - 分析结果: {len(state.analysis_results)} 个领域")
        
    except Exception as e:
        print(f"❌ 工作流执行失败: {e}")
    
    print()
    print("🎯 关键特性:")
    features = [
        "✅ 状态驱动: 智能体通过共享状态通信",
        "✅ 并发执行: 多个智能体同时工作",
        "✅ 异步工具调用: 高效的外部API调用",
        "✅ 错误恢复: 鲁棒的错误处理机制",
        "✅ 可观测性: 完整的执行日志和状态跟踪"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print()
    print("💡 与传统方法的区别:")
    differences = [
        "❌ 旧方法: 智能体通过HTTP API通信",
        "✅ 新方法: 智能体通过内存状态通信",
        "❌ 旧方法: 串行执行，效率低",
        "✅ 新方法: 并行执行，高效率",
        "❌ 旧方法: 工具调用分散，难以管理",
        "✅ 新方法: 统一工具执行器，并发调用"
    ]
    
    for diff in differences:
        print(f"   {diff}")


if __name__ == "__main__":
    asyncio.run(demonstrate_langgraph_approach())