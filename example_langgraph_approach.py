"""
示例：使用LangGraph风格的多agent实现
"""

from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class InvestmentAnalysisState(TypedDict):
    """共享状态定义"""
    # 输入信息
    symbol: str
    analysis_request: str
    
    # 各agent的分析结果
    industry_analysis: Dict[str, Any]
    financial_analysis: Dict[str, Any]
    market_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    
    # 共享数据
    market_data: Dict[str, Any]
    company_info: Dict[str, Any]
    
    # 对话历史
    messages: List[BaseMessage]
    
    # 最终报告
    final_report: str


class IndustryAgent:
    """行业分析Agent"""
    
    def __call__(self, state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        # 从共享状态获取数据
        symbol = state["symbol"]
        company_info = state.get("company_info", {})
        
        # 执行行业分析
        industry_analysis = {
            "sector": "Technology",
            "competitive_position": "Strong",
            "growth_outlook": "Positive",
            "key_trends": ["AI adoption", "Cloud migration"]
        }
        
        # 更新共享状态
        state["industry_analysis"] = industry_analysis
        state["messages"].append(
            AIMessage(content=f"Industry analysis completed for {symbol}")
        )
        
        return state


class FinancialAgent:
    """财务分析Agent"""
    
    def __call__(self, state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        symbol = state["symbol"]
        market_data = state.get("market_data", {})
        
        # 可以使用其他agent的结果
        industry_analysis = state.get("industry_analysis", {})
        
        financial_analysis = {
            "revenue_growth": "15%",
            "profit_margin": "25%",
            "valuation": "Fair",
            "debt_ratio": "Low"
        }
        
        state["financial_analysis"] = financial_analysis
        state["messages"].append(
            AIMessage(content=f"Financial analysis completed for {symbol}")
        )
        
        return state


class MarketAgent:
    """市场分析Agent"""
    
    def __call__(self, state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        symbol = state["symbol"]
        
        market_analysis = {
            "price_trend": "Upward",
            "volume_analysis": "Above average",
            "technical_indicators": "Bullish",
            "sentiment": "Positive"
        }
        
        state["market_analysis"] = market_analysis
        state["messages"].append(
            AIMessage(content=f"Market analysis completed for {symbol}")
        )
        
        return state


class RiskAgent:
    """风险分析Agent"""
    
    def __call__(self, state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        symbol = state["symbol"]
        
        # 综合其他agent的结果进行风险评估
        industry_analysis = state.get("industry_analysis", {})
        financial_analysis = state.get("financial_analysis", {})
        market_analysis = state.get("market_analysis", {})
        
        risk_analysis = {
            "overall_risk": "Medium",
            "key_risks": ["Market volatility", "Regulatory changes"],
            "risk_score": 6.5,
            "mitigation_strategies": ["Diversification", "Stop-loss orders"]
        }
        
        state["risk_analysis"] = risk_analysis
        state["messages"].append(
            AIMessage(content=f"Risk analysis completed for {symbol}")
        )
        
        return state


class ReportGenerator:
    """报告生成器"""
    
    def __call__(self, state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        # 整合所有分析结果
        industry = state.get("industry_analysis", {})
        financial = state.get("financial_analysis", {})
        market = state.get("market_analysis", {})
        risk = state.get("risk_analysis", {})
        
        report = f"""
        Investment Research Report: {state['symbol']}
        
        Industry Analysis:
        - Sector: {industry.get('sector', 'N/A')}
        - Position: {industry.get('competitive_position', 'N/A')}
        
        Financial Analysis:
        - Revenue Growth: {financial.get('revenue_growth', 'N/A')}
        - Valuation: {financial.get('valuation', 'N/A')}
        
        Market Analysis:
        - Trend: {market.get('price_trend', 'N/A')}
        - Sentiment: {market.get('sentiment', 'N/A')}
        
        Risk Assessment:
        - Overall Risk: {risk.get('overall_risk', 'N/A')}
        - Risk Score: {risk.get('risk_score', 'N/A')}
        """
        
        state["final_report"] = report
        state["messages"].append(
            AIMessage(content="Investment research report generated successfully")
        )
        
        return state


def create_investment_analysis_graph():
    """创建投资分析工作流图"""
    
    # 创建状态图
    workflow = StateGraph(InvestmentAnalysisState)
    
    # 添加节点
    workflow.add_node("industry_agent", IndustryAgent())
    workflow.add_node("financial_agent", FinancialAgent())
    workflow.add_node("market_agent", MarketAgent())
    workflow.add_node("risk_agent", RiskAgent())
    workflow.add_node("report_generator", ReportGenerator())
    
    # 定义执行流程
    workflow.set_entry_point("industry_agent")
    
    # 并行执行基础分析
    workflow.add_edge("industry_agent", "financial_agent")
    workflow.add_edge("financial_agent", "market_agent")
    
    # 风险分析需要等待其他分析完成
    workflow.add_edge("market_agent", "risk_agent")
    
    # 最后生成报告
    workflow.add_edge("risk_agent", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow.compile()


# 使用示例
async def run_investment_analysis(symbol: str, request: str):
    """运行投资分析"""
    
    graph = create_investment_analysis_graph()
    
    # 初始状态
    initial_state = InvestmentAnalysisState(
        symbol=symbol,
        analysis_request=request,
        messages=[HumanMessage(content=f"Analyze {symbol}: {request}")],
        industry_analysis={},
        financial_analysis={},
        market_analysis={},
        risk_analysis={},
        market_data={},
        company_info={},
        final_report=""
    )
    
    # 执行分析流程
    result = await graph.ainvoke(initial_state)
    
    return result["final_report"]


if __name__ == "__main__":
    import asyncio
    
    async def main():
        report = await run_investment_analysis(
            "AAPL", 
            "Comprehensive investment analysis for Q1 2024"
        )
        print(report)
    
    asyncio.run(main())