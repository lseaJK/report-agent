"""Industry analysis agent for sector-specific research and analysis."""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..core.models import AgentType
from ..core.state import ResearchState, Message, MessageRole, ToolCall
from .base import BaseAgent

logger = logging.getLogger(__name__)


class IndustryAgent(BaseAgent):
    """Specialized agent for industry analysis, trends, and competitive landscape."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, **kwargs):
        """Initialize the industry agent."""
        super().__init__(agent_id, agent_type, **kwargs)
        
        # Industry-specific configuration
        self.analysis_areas = [
            "market_size_and_growth",
            "competitive_landscape", 
            "regulatory_environment",
            "technology_trends",
            "supply_chain_analysis",
            "customer_segments",
            "barriers_to_entry",
            "industry_lifecycle"
        ]
        
        # Industry data sources priority
        self.preferred_data_sources = [
            "industry_reports",
            "regulatory_filings", 
            "trade_associations",
            "market_research",
            "news_analysis",
            "patent_databases"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for industry analysis."""
        return """你是一个专业的行业分析师，专门负责深入分析特定行业的各个方面。

你的主要职责包括：
1. 行业规模和增长趋势分析
2. 竞争格局和主要参与者评估
3. 监管环境和政策影响分析
4. 技术发展趋势和创新机会
5. 供应链结构和风险评估
6. 客户细分和需求分析
7. 行业进入壁垒和退出成本
8. 行业生命周期阶段判断

分析时请：
- 使用最新的行业数据和报告
- 关注监管变化和政策影响
- 识别关键成功因素和风险点
- 提供具体的数据支撑和案例
- 考虑宏观经济环境的影响
- 分析技术变革对行业的冲击

请以专业、客观的方式进行分析，并提供可操作的洞察。"""
    
    def get_required_tools(self) -> List[str]:
        """Get list of tools required for industry analysis."""
        return [
            "industry_report_search",
            "regulatory_database_search", 
            "news_sentiment_analysis",
            "patent_search",
            "trade_data_analysis",
            "market_research_api",
            "competitor_analysis_tool",
            "regulatory_tracker"
        ]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform comprehensive industry analysis."""
        self.logger.info(f"Starting industry analysis for topic: {state.topic}")
        
        # Extract industry information from topic
        industry_info = self._extract_industry_context(state.topic)
        
        # Phase 1: Market Size and Growth Analysis
        state = await self._analyze_market_size_and_growth(state, industry_info)
        
        # Phase 2: Competitive Landscape Analysis
        state = await self._analyze_competitive_landscape(state, industry_info)
        
        # Phase 3: Regulatory Environment Analysis
        state = await self._analyze_regulatory_environment(state, industry_info)
        
        # Phase 4: Technology and Innovation Trends
        state = await self._analyze_technology_trends(state, industry_info)
        
        # Phase 5: Supply Chain and Value Chain Analysis
        state = await self._analyze_supply_chain(state, industry_info)
        
        # Phase 6: Synthesize Industry Insights
        state = await self._synthesize_industry_insights(state, industry_info)
        
        self.logger.info("Industry analysis completed")
        return state
    
    def _extract_industry_context(self, topic: str) -> Dict[str, Any]:
        """Extract industry context from research topic."""
        # Simple keyword extraction - in practice, this would be more sophisticated
        industry_keywords = {
            "technology": ["tech", "software", "AI", "cloud", "digital"],
            "healthcare": ["health", "medical", "pharma", "biotech", "hospital"],
            "finance": ["bank", "fintech", "insurance", "payment", "credit"],
            "energy": ["oil", "gas", "renewable", "solar", "wind", "energy"],
            "retail": ["retail", "e-commerce", "consumer", "shopping"],
            "manufacturing": ["manufacturing", "industrial", "automotive", "aerospace"],
            "real_estate": ["real estate", "property", "construction", "housing"]
        }
        
        topic_lower = topic.lower()
        detected_industries = []
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                detected_industries.append(industry)
        
        return {
            "primary_industry": detected_industries[0] if detected_industries else "general",
            "related_industries": detected_industries[1:] if len(detected_industries) > 1 else [],
            "topic": topic,
            "analysis_scope": "comprehensive"
        }
    
    async def _analyze_market_size_and_growth(self, state: ResearchState, 
                                            industry_info: Dict[str, Any]) -> ResearchState:
        """Analyze market size, growth trends, and forecasts."""
        self.logger.info("Analyzing market size and growth trends")
        
        # Prepare tool calls for market data
        tool_calls = [
            ToolCall(
                tool_name="market_research_api",
                parameters={
                    "industry": industry_info["primary_industry"],
                    "metrics": ["market_size", "growth_rate", "forecast"],
                    "time_period": "5_years",
                    "regions": ["global", "north_america", "europe", "asia_pacific"]
                }
            ),
            ToolCall(
                tool_name="industry_report_search",
                parameters={
                    "query": f"{industry_info['primary_industry']} market size growth forecast",
                    "report_types": ["market_research", "industry_analysis"],
                    "date_range": "last_2_years"
                }
            )
        ]
        
        # Execute tool calls
        state = await self.call_tools(state, tool_calls)
        
        # Analyze results and generate insights
        market_analysis = await self.generate_response(
            state,
            f"""基于收集到的市场数据，请分析{industry_info['primary_industry']}行业的：

1. 当前市场规模和主要细分市场
2. 历史增长趋势和驱动因素
3. 未来3-5年增长预测和假设
4. 地区市场差异和机会
5. 市场成熟度和发展阶段

请提供具体的数据支撑和关键洞察。"""
        )
        
        # Store analysis results
        state.collected_data["market_size_analysis"] = {
            "analysis": market_analysis,
            "timestamp": datetime.utcnow(),
            "data_sources": [call.tool_name for call in tool_calls]
        }
        
        return state
    
    async def _analyze_competitive_landscape(self, state: ResearchState, 
                                           industry_info: Dict[str, Any]) -> ResearchState:
        """Analyze competitive landscape and key players."""
        self.logger.info("Analyzing competitive landscape")
        
        tool_calls = [
            ToolCall(
                tool_name="competitor_analysis_tool",
                parameters={
                    "industry": industry_info["primary_industry"],
                    "analysis_type": "market_leaders",
                    "metrics": ["market_share", "revenue", "growth_rate", "profitability"],
                    "top_n": 10
                }
            ),
            ToolCall(
                tool_name="news_sentiment_analysis",
                parameters={
                    "query": f"{industry_info['primary_industry']} competition market share",
                    "time_period": "last_6_months",
                    "sentiment_focus": "competitive_dynamics"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        competitive_analysis = await self.generate_response(
            state,
            f"""基于竞争情报数据，请分析{industry_info['primary_industry']}行业的：

1. 主要竞争者和市场份额分布
2. 竞争格局特征（集中度、竞争强度）
3. 关键成功因素和差异化策略
4. 新进入者和潜在威胁
5. 竞争优势的可持续性
6. 行业整合趋势和并购活动

请识别竞争格局的关键变化和未来趋势。"""
        )
        
        state.collected_data["competitive_landscape"] = {
            "analysis": competitive_analysis,
            "timestamp": datetime.utcnow(),
            "data_sources": [call.tool_name for call in tool_calls]
        }
        
        return state
    
    async def _analyze_regulatory_environment(self, state: ResearchState, 
                                            industry_info: Dict[str, Any]) -> ResearchState:
        """Analyze regulatory environment and policy impacts."""
        self.logger.info("Analyzing regulatory environment")
        
        tool_calls = [
            ToolCall(
                tool_name="regulatory_database_search",
                parameters={
                    "industry": industry_info["primary_industry"],
                    "regulation_types": ["current", "proposed", "international"],
                    "impact_areas": ["market_access", "compliance_costs", "operational_requirements"]
                }
            ),
            ToolCall(
                tool_name="regulatory_tracker",
                parameters={
                    "industry": industry_info["primary_industry"],
                    "tracking_period": "last_12_months",
                    "change_types": ["new_regulations", "amendments", "enforcement_changes"]
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        regulatory_analysis = await self.generate_response(
            state,
            f"""基于监管环境数据，请分析{industry_info['primary_industry']}行业的：

1. 当前主要监管框架和要求
2. 近期监管变化和影响评估
3. 合规成本和运营影响
4. 即将出台的政策和预期影响
5. 国际监管差异和协调趋势
6. 监管风险和机遇识别

请重点关注监管变化对行业竞争格局和商业模式的影响。"""
        )
        
        state.collected_data["regulatory_environment"] = {
            "analysis": regulatory_analysis,
            "timestamp": datetime.utcnow(),
            "data_sources": [call.tool_name for call in tool_calls]
        }
        
        return state
    
    async def _analyze_technology_trends(self, state: ResearchState, 
                                       industry_info: Dict[str, Any]) -> ResearchState:
        """Analyze technology trends and innovation opportunities."""
        self.logger.info("Analyzing technology trends and innovation")
        
        tool_calls = [
            ToolCall(
                tool_name="patent_search",
                parameters={
                    "industry": industry_info["primary_industry"],
                    "technology_areas": ["emerging_tech", "process_innovation", "product_innovation"],
                    "time_period": "last_3_years",
                    "patent_metrics": ["filing_trends", "top_assignees", "technology_clusters"]
                }
            ),
            ToolCall(
                tool_name="news_sentiment_analysis",
                parameters={
                    "query": f"{industry_info['primary_industry']} technology innovation trends",
                    "time_period": "last_12_months",
                    "sentiment_focus": "technology_adoption"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        technology_analysis = await self.generate_response(
            state,
            f"""基于技术趋势数据，请分析{industry_info['primary_industry']}行业的：

1. 关键技术发展趋势和创新方向
2. 技术采用周期和成熟度评估
3. 颠覆性技术的潜在影响
4. 技术投资热点和专利活动
5. 数字化转型进展和挑战
6. 技术标准化和生态系统发展

请评估技术变革对行业价值链和商业模式的重塑作用。"""
        )
        
        state.collected_data["technology_trends"] = {
            "analysis": technology_analysis,
            "timestamp": datetime.utcnow(),
            "data_sources": [call.tool_name for call in tool_calls]
        }
        
        return state
    
    async def _analyze_supply_chain(self, state: ResearchState, 
                                  industry_info: Dict[str, Any]) -> ResearchState:
        """Analyze supply chain structure and risks."""
        self.logger.info("Analyzing supply chain and value chain")
        
        tool_calls = [
            ToolCall(
                tool_name="trade_data_analysis",
                parameters={
                    "industry": industry_info["primary_industry"],
                    "analysis_type": "supply_chain_mapping",
                    "geographic_scope": "global",
                    "trade_flows": ["imports", "exports", "intermediate_goods"]
                }
            ),
            ToolCall(
                tool_name="news_sentiment_analysis",
                parameters={
                    "query": f"{industry_info['primary_industry']} supply chain disruption risk",
                    "time_period": "last_6_months",
                    "sentiment_focus": "supply_chain_resilience"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        supply_chain_analysis = await self.generate_response(
            state,
            f"""基于供应链数据，请分析{industry_info['primary_industry']}行业的：

1. 供应链结构和关键环节
2. 主要供应商和地理分布
3. 供应链风险和脆弱性
4. 供应链效率和成本结构
5. 垂直整合vs外包趋势
6. 供应链数字化和自动化
7. 可持续性和ESG要求影响

请识别供应链优化机会和风险缓解策略。"""
        )
        
        state.collected_data["supply_chain_analysis"] = {
            "analysis": supply_chain_analysis,
            "timestamp": datetime.utcnow(),
            "data_sources": [call.tool_name for call in tool_calls]
        }
        
        return state
    
    async def _synthesize_industry_insights(self, state: ResearchState, 
                                          industry_info: Dict[str, Any]) -> ResearchState:
        """Synthesize all industry analysis into comprehensive insights."""
        self.logger.info("Synthesizing industry insights")
        
        # Gather all analysis results
        analysis_components = []
        for key in ["market_size_analysis", "competitive_landscape", "regulatory_environment", 
                   "technology_trends", "supply_chain_analysis"]:
            if key in state.collected_data:
                analysis_components.append(f"{key}: {state.collected_data[key]['analysis'][:500]}...")
        
        synthesis_prompt = f"""基于以下各个维度的分析结果，请为{industry_info['primary_industry']}行业提供综合性洞察：

{chr(10).join(analysis_components)}

请提供：
1. 行业整体健康度和发展阶段评估
2. 关键驱动因素和制约因素
3. 主要机遇和威胁识别
4. 行业发展趋势预测
5. 投资吸引力评估
6. 风险因素和缓解建议
7. 对相关利益方的影响分析

请确保洞察具有前瞻性和可操作性。"""
        
        industry_synthesis = await self.generate_response(state, synthesis_prompt)
        
        # Create comprehensive industry report
        industry_report = {
            "industry": industry_info["primary_industry"],
            "analysis_date": datetime.utcnow(),
            "executive_summary": industry_synthesis,
            "detailed_analysis": {
                key: state.collected_data.get(key, {}).get("analysis", "")
                for key in ["market_size_analysis", "competitive_landscape", 
                           "regulatory_environment", "technology_trends", "supply_chain_analysis"]
            },
            "key_insights": self._extract_key_insights(industry_synthesis),
            "recommendations": self._extract_recommendations(industry_synthesis),
            "risk_factors": self._extract_risk_factors(industry_synthesis)
        }
        
        state.collected_data["industry_analysis_report"] = industry_report
        
        # Add final summary message
        summary_message = Message(
            role=MessageRole.ASSISTANT,
            content=f"完成了{industry_info['primary_industry']}行业的全面分析，涵盖市场规模、竞争格局、监管环境、技术趋势和供应链等关键维度。",
            agent_id=self.agent_id,
            metadata={
                "analysis_type": "industry_comprehensive",
                "industry": industry_info["primary_industry"],
                "components_analyzed": len(analysis_components)
            }
        )
        state.add_message(summary_message)
        
        return state
    
    def _extract_key_insights(self, analysis_text: str) -> List[str]:
        """Extract key insights from analysis text."""
        # Simple extraction based on keywords - in practice, this would use NLP
        insights = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['关键', '重要', '显著', '主要']):
                insights.append(line.strip())
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract recommendations from analysis text."""
        recommendations = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['建议', '推荐', '应该', '需要']):
                recommendations.append(line.strip())
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _extract_risk_factors(self, analysis_text: str) -> List[str]:
        """Extract risk factors from analysis text."""
        risks = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['风险', '威胁', '挑战', '不确定']):
                risks.append(line.strip())
        
        return risks[:3]  # Return top 3 risks