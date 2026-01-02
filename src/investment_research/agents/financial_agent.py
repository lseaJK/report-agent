"""Financial analysis agent for comprehensive financial modeling and valuation."""

import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from ..core.models import AgentType
from ..core.state import ResearchState, Message, MessageRole, ToolCall
from .base import BaseAgent

logger = logging.getLogger(__name__)


class FinancialAgent(BaseAgent):
    """Specialized agent for financial analysis, valuation, and forecasting."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, **kwargs):
        """Initialize the financial agent."""
        super().__init__(agent_id, agent_type, **kwargs)
        
        # Financial analysis areas
        self.analysis_areas = [
            "financial_statements_analysis",
            "ratio_analysis", 
            "cash_flow_analysis",
            "valuation_modeling",
            "risk_assessment",
            "peer_comparison",
            "forecasting",
            "scenario_analysis"
        ]
        
        # Financial data sources
        self.preferred_data_sources = [
            "financial_statements",
            "market_data",
            "analyst_estimates", 
            "economic_indicators",
            "credit_ratings",
            "options_data"
        ]
        
        # Valuation methods
        self.valuation_methods = [
            "dcf_analysis",
            "comparable_company_analysis", 
            "precedent_transactions",
            "asset_based_valuation",
            "earnings_multiples",
            "book_value_multiples"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for financial analysis."""
        return """你是一个专业的财务分析师和估值专家，专门负责深入的财务分析和投资评估。

你的主要职责包括：
1. 财务报表分析和质量评估
2. 财务比率分析和趋势识别
3. 现金流分析和预测
4. 估值建模（DCF、相对估值等）
5. 财务风险评估和压力测试
6. 同业比较和基准分析
7. 财务预测和敏感性分析
8. 投资回报和风险调整收益计算

分析时请：
- 使用最新的财务数据和市场信息
- 应用多种估值方法进行交叉验证
- 考虑会计政策和一次性项目影响
- 进行敏感性和情景分析
- 评估财务报表质量和可信度
- 识别关键财务驱动因素
- 提供风险调整的投资建议

请以严谨、量化的方式进行分析，并提供明确的投资结论。"""
    
    def get_required_tools(self) -> List[str]:
        """Get list of tools required for financial analysis."""
        return [
            "financial_data_api",
            "market_data_feed",
            "analyst_estimates_api", 
            "economic_data_api",
            "credit_rating_api",
            "options_data_api",
            "peer_comparison_tool",
            "valuation_calculator",
            "risk_metrics_calculator"
        ]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform comprehensive financial analysis."""
        self.logger.info(f"Starting financial analysis for topic: {state.topic}")
        
        # Extract financial context from topic
        financial_context = self._extract_financial_context(state.topic)
        
        # Phase 1: Financial Data Collection
        state = await self._collect_financial_data(state, financial_context)
        
        # Phase 2: Financial Statements Analysis
        state = await self._analyze_financial_statements(state, financial_context)
        
        # Phase 3: Ratio and Performance Analysis
        state = await self._analyze_financial_ratios(state, financial_context)
        
        # Phase 4: Cash Flow Analysis
        state = await self._analyze_cash_flows(state, financial_context)
        
        # Phase 5: Valuation Analysis
        state = await self._perform_valuation_analysis(state, financial_context)
        
        # Phase 6: Risk Assessment
        state = await self._assess_financial_risks(state, financial_context)
        
        # Phase 7: Peer Comparison
        state = await self._perform_peer_comparison(state, financial_context)
        
        # Phase 8: Financial Synthesis and Recommendations
        state = await self._synthesize_financial_analysis(state, financial_context)
        
        self.logger.info("Financial analysis completed")
        return state
    
    def _extract_financial_context(self, topic: str) -> Dict[str, Any]:
        """Extract financial context from research topic."""
        # Extract company/entity information
        context = {
            "analysis_type": "comprehensive",
            "time_horizon": "5_years",
            "currency": "USD",
            "analysis_date": datetime.utcnow()
        }
        
        # Detect analysis focus
        topic_lower = topic.lower()
        if "valuation" in topic_lower or "估值" in topic_lower:
            context["primary_focus"] = "valuation"
        elif "risk" in topic_lower or "风险" in topic_lower:
            context["primary_focus"] = "risk_assessment"
        elif "performance" in topic_lower or "业绩" in topic_lower:
            context["primary_focus"] = "performance_analysis"
        else:
            context["primary_focus"] = "comprehensive"
        
        return context
    
    async def _collect_financial_data(self, state: ResearchState, 
                                    context: Dict[str, Any]) -> ResearchState:
        """Collect comprehensive financial data."""
        self.logger.info("Collecting financial data")
        
        tool_calls = [
            ToolCall(
                tool_name="financial_data_api",
                parameters={
                    "data_types": ["income_statement", "balance_sheet", "cash_flow"],
                    "periods": ["annual", "quarterly"],
                    "years": 5,
                    "currency": context["currency"]
                }
            ),
            ToolCall(
                tool_name="market_data_feed",
                parameters={
                    "data_types": ["stock_price", "trading_volume", "market_cap"],
                    "time_period": "2_years",
                    "frequency": "daily"
                }
            ),
            ToolCall(
                tool_name="analyst_estimates_api",
                parameters={
                    "estimate_types": ["earnings", "revenue", "growth_rates"],
                    "time_horizon": "3_years",
                    "consensus_metrics": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        # Process and validate financial data
        data_validation = await self.generate_response(
            state,
            """请验证收集到的财务数据的完整性和质量：

1. 检查财务报表数据的完整性和一致性
2. 识别任何异常值或数据缺失
3. 评估数据来源的可靠性
4. 标记需要特别关注的会计项目
5. 确认货币单位和会计准则

请提供数据质量评估报告。"""
        )
        
        state.collected_data["financial_data_collection"] = {
            "validation_report": data_validation,
            "timestamp": datetime.utcnow(),
            "data_sources": [call.tool_name for call in tool_calls]
        }
        
        return state
    
    async def _analyze_financial_statements(self, state: ResearchState, 
                                          context: Dict[str, Any]) -> ResearchState:
        """Analyze financial statements for trends and quality."""
        self.logger.info("Analyzing financial statements")
        
        tool_calls = [
            ToolCall(
                tool_name="financial_data_api",
                parameters={
                    "analysis_type": "trend_analysis",
                    "metrics": ["revenue_growth", "margin_trends", "asset_turnover"],
                    "time_period": "5_years"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        statements_analysis = await self.generate_response(
            state,
            """基于财务报表数据，请进行深入分析：

1. 收入分析：
   - 收入增长趋势和驱动因素
   - 收入质量和可持续性
   - 季节性和周期性模式
   - 收入确认政策影响

2. 盈利能力分析：
   - 毛利率、营业利润率、净利率趋势
   - 成本结构变化和效率改善
   - 一次性项目和调整后盈利
   - 盈利质量评估

3. 资产负债表分析：
   - 资产结构和质量
   - 负债结构和偿债能力
   - 营运资本管理效率
   - 资本结构优化机会

请提供具体的财务指标和趋势分析。"""
        )
        
        state.collected_data["financial_statements_analysis"] = {
            "analysis": statements_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_financial_ratios(self, state: ResearchState, 
                                      context: Dict[str, Any]) -> ResearchState:
        """Perform comprehensive ratio analysis."""
        self.logger.info("Performing ratio analysis")
        
        # Calculate key financial ratios
        ratios_to_calculate = {
            "liquidity": ["current_ratio", "quick_ratio", "cash_ratio"],
            "efficiency": ["asset_turnover", "inventory_turnover", "receivables_turnover"],
            "leverage": ["debt_to_equity", "debt_to_assets", "interest_coverage"],
            "profitability": ["roe", "roa", "roic", "gross_margin", "net_margin"],
            "market": ["pe_ratio", "pb_ratio", "ev_ebitda", "price_to_sales"]
        }
        
        tool_calls = [
            ToolCall(
                tool_name="financial_data_api",
                parameters={
                    "analysis_type": "ratio_calculation",
                    "ratio_categories": list(ratios_to_calculate.keys()),
                    "time_period": "5_years",
                    "benchmark_comparison": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        ratio_analysis = await self.generate_response(
            state,
            f"""基于计算的财务比率，请进行全面分析：

1. 流动性分析：
   - 短期偿债能力评估
   - 营运资本管理效率
   - 现金管理质量

2. 运营效率分析：
   - 资产使用效率
   - 存货和应收账款管理
   - 营运周期优化

3. 财务杠杆分析：
   - 资本结构合理性
   - 偿债能力和财务风险
   - 利息保障倍数

4. 盈利能力分析：
   - 股东回报率（ROE）分解
   - 资产回报率（ROA）趋势
   - 投资资本回报率（ROIC）

5. 市场估值分析：
   - 估值倍数合理性
   - 与行业平均水平比较
   - 估值溢价/折价分析

请识别关键的财务优势和改进领域。"""
        )
        
        state.collected_data["ratio_analysis"] = {
            "analysis": ratio_analysis,
            "ratios_calculated": ratios_to_calculate,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_cash_flows(self, state: ResearchState, 
                                context: Dict[str, Any]) -> ResearchState:
        """Analyze cash flow patterns and quality."""
        self.logger.info("Analyzing cash flows")
        
        tool_calls = [
            ToolCall(
                tool_name="financial_data_api",
                parameters={
                    "analysis_type": "cash_flow_analysis",
                    "cash_flow_types": ["operating", "investing", "financing"],
                    "metrics": ["fcf", "fcf_yield", "cash_conversion_cycle"],
                    "time_period": "5_years"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        cash_flow_analysis = await self.generate_response(
            state,
            """基于现金流数据，请进行深入分析：

1. 经营现金流分析：
   - 经营现金流趋势和稳定性
   - 现金流与净利润的关系
   - 营运资本变化影响
   - 现金流质量评估

2. 自由现金流分析：
   - 自由现金流生成能力
   - 资本支出效率
   - 自由现金流收益率
   - 现金流可持续性

3. 投资现金流分析：
   - 资本支出模式和战略
   - 并购活动影响
   - 资产处置收益
   - 投资回报评估

4. 融资现金流分析：
   - 融资结构变化
   - 股利政策和股票回购
   - 债务管理策略
   - 资本配置效率

5. 现金流预测：
   - 未来现金流生成能力
   - 现金流敏感性分析
   - 流动性需求评估

请评估现金流的质量和可预测性。"""
        )
        
        state.collected_data["cash_flow_analysis"] = {
            "analysis": cash_flow_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _perform_valuation_analysis(self, state: ResearchState, 
                                        context: Dict[str, Any]) -> ResearchState:
        """Perform comprehensive valuation analysis using multiple methods."""
        self.logger.info("Performing valuation analysis")
        
        tool_calls = [
            ToolCall(
                tool_name="valuation_calculator",
                parameters={
                    "valuation_methods": ["dcf", "comparable_companies", "precedent_transactions"],
                    "forecast_period": 5,
                    "terminal_growth_rate": 0.025,
                    "discount_rate_method": "wacc"
                }
            ),
            ToolCall(
                tool_name="peer_comparison_tool",
                parameters={
                    "comparison_metrics": ["ev_ebitda", "pe_ratio", "pb_ratio", "price_to_sales"],
                    "peer_selection_criteria": "industry_and_size",
                    "adjustment_factors": ["growth", "profitability", "risk"]
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        valuation_analysis = await self.generate_response(
            state,
            """基于多种估值方法，请进行综合估值分析：

1. DCF估值分析：
   - 现金流预测假设和合理性
   - WACC计算和敏感性分析
   - 终值计算和增长率假设
   - DCF估值结果和价值区间

2. 相对估值分析：
   - 可比公司选择和调整
   - 关键估值倍数比较
   - 估值溢价/折价分析
   - 相对估值合理性

3. 资产估值分析：
   - 有形资产和无形资产价值
   - 重置成本和清算价值
   - 资产估值vs市场估值

4. 估值综合分析：
   - 不同方法结果比较
   - 估值区间和目标价格
   - 关键估值驱动因素
   - 估值风险和不确定性

5. 投资建议：
   - 基于估值的投资评级
   - 风险调整收益预期
   - 投资时机建议

请提供明确的估值结论和投资建议。"""
        )
        
        state.collected_data["valuation_analysis"] = {
            "analysis": valuation_analysis,
            "methods_used": ["dcf", "comparable_companies", "precedent_transactions"],
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _assess_financial_risks(self, state: ResearchState, 
                                    context: Dict[str, Any]) -> ResearchState:
        """Assess various financial risks."""
        self.logger.info("Assessing financial risks")
        
        tool_calls = [
            ToolCall(
                tool_name="risk_metrics_calculator",
                parameters={
                    "risk_types": ["credit_risk", "liquidity_risk", "market_risk", "operational_risk"],
                    "risk_metrics": ["var", "beta", "volatility", "sharpe_ratio"],
                    "time_horizon": "1_year"
                }
            ),
            ToolCall(
                tool_name="credit_rating_api",
                parameters={
                    "rating_agencies": ["sp", "moody", "fitch"],
                    "rating_history": "5_years",
                    "rating_rationale": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        risk_analysis = await self.generate_response(
            state,
            """基于风险指标和信用评级，请进行全面风险评估：

1. 信用风险分析：
   - 信用评级和变化趋势
   - 违约概率评估
   - 债务偿付能力
   - 信用风险缓释措施

2. 流动性风险分析：
   - 短期流动性需求
   - 融资渠道多样性
   - 现金管理策略
   - 流动性压力测试

3. 市场风险分析：
   - 股价波动性和Beta系数
   - 汇率和利率敏感性
   - 商品价格风险暴露
   - 市场风险对冲策略

4. 运营风险分析：
   - 业务模式风险
   - 关键人员依赖
   - 系统和流程风险
   - 合规和监管风险

5. 综合风险评估：
   - 风险调整收益指标
   - 风险承受能力
   - 风险管理有效性
   - 风险缓释建议

请提供风险等级评估和管理建议。"""
        )
        
        state.collected_data["risk_assessment"] = {
            "analysis": risk_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _perform_peer_comparison(self, state: ResearchState, 
                                     context: Dict[str, Any]) -> ResearchState:
        """Perform detailed peer comparison analysis."""
        self.logger.info("Performing peer comparison")
        
        tool_calls = [
            ToolCall(
                tool_name="peer_comparison_tool",
                parameters={
                    "comparison_dimensions": ["financial_performance", "valuation", "growth", "profitability"],
                    "peer_selection": "industry_leaders",
                    "benchmarking_metrics": ["revenue_growth", "margin_trends", "roic", "debt_levels"],
                    "time_period": "3_years"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        peer_analysis = await self.generate_response(
            state,
            """基于同业比较数据，请进行详细的竞争地位分析：

1. 财务表现比较：
   - 收入增长率vs同业
   - 盈利能力vs同业平均
   - 资产使用效率比较
   - 现金流生成能力对比

2. 估值水平比较：
   - 估值倍数vs同业
   - 估值溢价/折价分析
   - 估值合理性评估
   - 相对投资吸引力

3. 成长性比较：
   - 历史增长率对比
   - 未来增长前景
   - 市场份额变化
   - 创新能力比较

4. 财务质量比较：
   - 资产负债表强度
   - 现金流质量
   - 财务灵活性
   - 风险管理水平

5. 竞争优势分析：
   - 相对竞争地位
   - 关键成功因素
   - 可持续竞争优势
   - 改进机会识别

请评估相对竞争地位和投资价值。"""
        )
        
        state.collected_data["peer_comparison"] = {
            "analysis": peer_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _synthesize_financial_analysis(self, state: ResearchState, 
                                           context: Dict[str, Any]) -> ResearchState:
        """Synthesize all financial analysis into investment recommendations."""
        self.logger.info("Synthesizing financial analysis")
        
        # Gather all analysis components
        analysis_components = []
        for key in ["financial_statements_analysis", "ratio_analysis", "cash_flow_analysis", 
                   "valuation_analysis", "risk_assessment", "peer_comparison"]:
            if key in state.collected_data:
                analysis_components.append(f"{key}: {state.collected_data[key]['analysis'][:400]}...")
        
        synthesis_prompt = f"""基于以下全面的财务分析结果，请提供综合性投资建议：

{chr(10).join(analysis_components)}

请提供：
1. 财务健康度综合评估
   - 财务实力评级
   - 关键财务指标总结
   - 财务趋势判断

2. 投资价值分析
   - 内在价值评估
   - 投资吸引力评级
   - 风险调整收益预期

3. 关键投资要点
   - 主要投资亮点
   - 关键风险因素
   - 催化剂和风险事件

4. 投资建议
   - 明确的投资评级（买入/持有/卖出）
   - 目标价格区间
   - 投资时间框架

5. 敏感性分析
   - 关键假设变化影响
   - 情景分析结果
   - 风险缓释策略

请确保建议具有可操作性和明确性。"""
        
        financial_synthesis = await self.generate_response(state, synthesis_prompt)
        
        # Create comprehensive financial report
        financial_report = {
            "analysis_date": datetime.utcnow(),
            "analysis_scope": context["primary_focus"],
            "executive_summary": financial_synthesis,
            "detailed_analysis": {
                key: state.collected_data.get(key, {}).get("analysis", "")
                for key in ["financial_statements_analysis", "ratio_analysis", 
                           "cash_flow_analysis", "valuation_analysis", "risk_assessment", "peer_comparison"]
            },
            "key_metrics": self._extract_key_metrics(state),
            "investment_rating": self._extract_investment_rating(financial_synthesis),
            "target_price": self._extract_target_price(financial_synthesis),
            "risk_factors": self._extract_financial_risks(financial_synthesis)
        }
        
        state.collected_data["financial_analysis_report"] = financial_report
        
        # Add final summary message
        summary_message = Message(
            role=MessageRole.ASSISTANT,
            content=f"完成了全面的财务分析，包括财务报表分析、比率分析、现金流分析、估值分析、风险评估和同业比较。",
            agent_id=self.agent_id,
            metadata={
                "analysis_type": "financial_comprehensive",
                "components_analyzed": len(analysis_components),
                "investment_focus": context["primary_focus"]
            }
        )
        state.add_message(summary_message)
        
        return state
    
    def _extract_key_metrics(self, state: ResearchState) -> Dict[str, Any]:
        """Extract key financial metrics from analysis."""
        # This would extract specific metrics from the analysis
        # For now, return a placeholder structure
        return {
            "revenue_growth": "N/A",
            "net_margin": "N/A", 
            "roe": "N/A",
            "debt_to_equity": "N/A",
            "free_cash_flow_yield": "N/A",
            "pe_ratio": "N/A"
        }
    
    def _extract_investment_rating(self, analysis_text: str) -> str:
        """Extract investment rating from analysis text."""
        text_lower = analysis_text.lower()
        if "买入" in text_lower or "buy" in text_lower:
            return "买入"
        elif "卖出" in text_lower or "sell" in text_lower:
            return "卖出"
        elif "持有" in text_lower or "hold" in text_lower:
            return "持有"
        else:
            return "未评级"
    
    def _extract_target_price(self, analysis_text: str) -> Optional[str]:
        """Extract target price from analysis text."""
        # Simple extraction - in practice, this would use NLP
        lines = analysis_text.split('\n')
        for line in lines:
            if "目标价" in line or "target price" in line.lower():
                return line.strip()
        return None
    
    def _extract_financial_risks(self, analysis_text: str) -> List[str]:
        """Extract financial risks from analysis text."""
        risks = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['风险', '威胁', '挑战', '不确定', 'risk']):
                risks.append(line.strip())
        
        return risks[:5]  # Return top 5 risks