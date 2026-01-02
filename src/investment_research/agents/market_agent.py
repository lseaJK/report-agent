"""Market analysis agent for market research and consumer behavior analysis."""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..core.models import AgentType
from ..core.state import ResearchState, Message, MessageRole, ToolCall
from .base import BaseAgent

logger = logging.getLogger(__name__)


class MarketAgent(BaseAgent):
    """Specialized agent for market research, consumer analysis, and demand forecasting."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, **kwargs):
        """Initialize the market agent."""
        super().__init__(agent_id, agent_type, **kwargs)
        
        # Market analysis areas
        self.analysis_areas = [
            "market_sizing",
            "consumer_behavior",
            "demand_analysis", 
            "pricing_analysis",
            "channel_analysis",
            "geographic_analysis",
            "trend_analysis",
            "competitive_positioning"
        ]
        
        # Market data sources
        self.preferred_data_sources = [
            "consumer_surveys",
            "market_research_reports",
            "social_media_analytics",
            "e_commerce_data",
            "demographic_data",
            "economic_indicators"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for market analysis."""
        return """你是一个专业的市场研究分析师，专门负责市场调研、消费者行为分析和需求预测。

你的主要职责包括：
1. 市场规模评估和细分分析
2. 消费者行为和偏好研究
3. 需求分析和预测建模
4. 价格敏感性和定价策略分析
5. 销售渠道和分销策略评估
6. 地理市场差异和机会分析
7. 市场趋势识别和影响评估
8. 竞争定位和差异化分析

分析时请：
- 使用多元化的市场数据来源
- 关注消费者需求变化趋势
- 分析人口统计和行为特征
- 评估市场进入和扩张机会
- 识别新兴市场和细分领域
- 考虑技术和社会变革影响
- 提供数据驱动的市场洞察

请以消费者为中心的视角进行分析，并提供可操作的市场策略建议。"""
    
    def get_required_tools(self) -> List[str]:
        """Get list of tools required for market analysis."""
        return [
            "market_research_api",
            "consumer_survey_tool",
            "social_media_analytics",
            "e_commerce_analytics", 
            "demographic_data_api",
            "pricing_analytics_tool",
            "channel_analysis_tool",
            "trend_analysis_api"
        ]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform comprehensive market analysis."""
        self.logger.info(f"Starting market analysis for topic: {state.topic}")
        
        # Extract market context from topic
        market_context = self._extract_market_context(state.topic)
        
        # Phase 1: Market Sizing and Segmentation
        state = await self._analyze_market_size_and_segments(state, market_context)
        
        # Phase 2: Consumer Behavior Analysis
        state = await self._analyze_consumer_behavior(state, market_context)
        
        # Phase 3: Demand Analysis and Forecasting
        state = await self._analyze_demand_patterns(state, market_context)
        
        # Phase 4: Pricing and Value Analysis
        state = await self._analyze_pricing_dynamics(state, market_context)
        
        # Phase 5: Channel and Distribution Analysis
        state = await self._analyze_channels_and_distribution(state, market_context)
        
        # Phase 6: Geographic Market Analysis
        state = await self._analyze_geographic_markets(state, market_context)
        
        # Phase 7: Market Trends and Future Outlook
        state = await self._analyze_market_trends(state, market_context)
        
        # Phase 8: Market Synthesis and Recommendations
        state = await self._synthesize_market_analysis(state, market_context)
        
        self.logger.info("Market analysis completed")
        return state
    
    def _extract_market_context(self, topic: str) -> Dict[str, Any]:
        """Extract market context from research topic."""
        context = {
            "analysis_scope": "comprehensive",
            "geographic_focus": "global",
            "time_horizon": "3_years",
            "analysis_date": datetime.utcnow()
        }
        
        # Detect market type
        topic_lower = topic.lower()
        if "b2b" in topic_lower or "enterprise" in topic_lower:
            context["market_type"] = "b2b"
        elif "b2c" in topic_lower or "consumer" in topic_lower:
            context["market_type"] = "b2c"
        else:
            context["market_type"] = "mixed"
        
        return context
    
    async def _analyze_market_size_and_segments(self, state: ResearchState, 
                                              context: Dict[str, Any]) -> ResearchState:
        """Analyze market size and key segments."""
        self.logger.info("Analyzing market size and segmentation")
        
        tool_calls = [
            ToolCall(
                tool_name="market_research_api",
                parameters={
                    "analysis_type": "market_sizing",
                    "segmentation_criteria": ["demographic", "behavioral", "geographic", "psychographic"],
                    "market_metrics": ["tam", "sam", "som"],
                    "time_period": "5_years"
                }
            ),
            ToolCall(
                tool_name="demographic_data_api",
                parameters={
                    "data_types": ["population", "income", "age_distribution", "education"],
                    "geographic_scope": context["geographic_focus"],
                    "trend_analysis": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        market_sizing_analysis = await self.generate_response(
            state,
            f"""基于市场规模和细分数据，请进行深入分析：

1. 总体市场规模分析：
   - TAM（总可获得市场）评估
   - SAM（可服务市场）分析
   - SOM（可获得市场份额）预测
   - 市场增长率和驱动因素

2. 市场细分分析：
   - 主要细分市场识别
   - 各细分市场规模和特征
   - 细分市场增长潜力
   - 目标细分市场优先级

3. 人口统计分析：
   - 目标人群特征和规模
   - 人口结构变化趋势
   - 收入水平和消费能力
   - 地理分布和密度

4. 市场机会评估：
   - 未开发市场机会
   - 新兴细分领域
   - 市场进入壁垒
   - 增长机会优先级

请提供具体的市场规模数据和细分策略建议。"""
        )
        
        state.collected_data["market_sizing_analysis"] = {
            "analysis": market_sizing_analysis,
            "timestamp": datetime.utcnow(),
            "market_type": context["market_type"]
        }
        
        return state
    
    async def _analyze_consumer_behavior(self, state: ResearchState, 
                                       context: Dict[str, Any]) -> ResearchState:
        """Analyze consumer behavior and preferences."""
        self.logger.info("Analyzing consumer behavior")
        
        tool_calls = [
            ToolCall(
                tool_name="consumer_survey_tool",
                parameters={
                    "survey_types": ["purchase_behavior", "brand_preference", "satisfaction"],
                    "demographic_filters": ["age", "income", "location"],
                    "behavioral_metrics": ["frequency", "loyalty", "switching_patterns"]
                }
            ),
            ToolCall(
                tool_name="social_media_analytics",
                parameters={
                    "analysis_type": "consumer_sentiment",
                    "platforms": ["twitter", "facebook", "instagram", "linkedin"],
                    "sentiment_categories": ["brand_perception", "product_feedback", "trends"],
                    "time_period": "12_months"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        consumer_analysis = await self.generate_response(
            state,
            """基于消费者行为数据，请进行全面分析：

1. 购买行为分析：
   - 购买决策过程和影响因素
   - 购买频率和季节性模式
   - 价格敏感性和支付偏好
   - 渠道选择和购买路径

2. 消费者偏好分析：
   - 产品功能和特性偏好
   - 品牌认知和忠诚度
   - 服务质量期望
   - 创新接受度和早期采用者特征

3. 消费者细分：
   - 基于行为的消费者群体
   - 各群体的独特需求和特征
   - 价值主张匹配度
   - 细分群体增长潜力

4. 消费趋势分析：
   - 新兴消费趋势和模式
   - 代际差异和变化
   - 数字化行为转变
   - 可持续性和社会责任关注

5. 消费者洞察：
   - 未满足的需求和痛点
   - 消费者期望变化
   - 影响购买的关键因素
   - 提升消费者体验的机会

请提供可操作的消费者洞察和策略建议。"""
        )
        
        state.collected_data["consumer_behavior_analysis"] = {
            "analysis": consumer_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_demand_patterns(self, state: ResearchState, 
                                     context: Dict[str, Any]) -> ResearchState:
        """Analyze demand patterns and forecasting."""
        self.logger.info("Analyzing demand patterns")
        
        tool_calls = [
            ToolCall(
                tool_name="market_research_api",
                parameters={
                    "analysis_type": "demand_forecasting",
                    "demand_drivers": ["economic", "demographic", "technological", "social"],
                    "forecast_horizon": "3_years",
                    "scenario_analysis": ["optimistic", "base", "pessimistic"]
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        demand_analysis = await self.generate_response(
            state,
            """基于需求模式数据，请进行预测分析：

1. 历史需求分析：
   - 需求增长趋势和波动性
   - 季节性和周期性模式
   - 需求驱动因素识别
   - 外部冲击对需求的影响

2. 需求预测建模：
   - 基准情景需求预测
   - 乐观和悲观情景分析
   - 关键假设和敏感性分析
   - 预测准确性和置信区间

3. 需求驱动因素：
   - 宏观经济因素影响
   - 人口结构变化影响
   - 技术进步推动需求
   - 政策和监管影响

4. 需求弹性分析：
   - 价格弹性评估
   - 收入弹性分析
   - 替代品和互补品影响
   - 交叉弹性效应

5. 未来需求展望：
   - 新兴需求领域识别
   - 需求结构变化趋势
   - 潜在需求释放因素
   - 需求风险和机遇

请提供需求预测结果和策略含义。"""
        )
        
        state.collected_data["demand_analysis"] = {
            "analysis": demand_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_pricing_dynamics(self, state: ResearchState, 
                                      context: Dict[str, Any]) -> ResearchState:
        """Analyze pricing dynamics and strategies."""
        self.logger.info("Analyzing pricing dynamics")
        
        tool_calls = [
            ToolCall(
                tool_name="pricing_analytics_tool",
                parameters={
                    "analysis_type": "price_optimization",
                    "pricing_factors": ["cost", "competition", "value", "demand"],
                    "pricing_strategies": ["penetration", "skimming", "competitive", "value_based"],
                    "elasticity_analysis": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        pricing_analysis = await self.generate_response(
            state,
            """基于定价分析数据，请进行策略评估：

1. 当前定价分析：
   - 价格水平和结构分析
   - 与竞争对手价格比较
   - 价格-价值关系评估
   - 定价透明度和复杂性

2. 价格弹性分析：
   - 需求价格弹性测算
   - 不同细分市场弹性差异
   - 价格敏感性因素
   - 最优价格点识别

3. 定价策略评估：
   - 现有定价策略有效性
   - 动态定价机会
   - 差异化定价潜力
   - 捆绑和套餐策略

4. 竞争定价分析：
   - 竞争对手定价策略
   - 价格战风险评估
   - 定价跟随vs领导策略
   - 非价格竞争因素

5. 定价优化建议：
   - 定价策略改进机会
   - 收入最大化策略
   - 市场份额vs利润权衡
   - 定价实施建议

请提供定价策略优化建议。"""
        )
        
        state.collected_data["pricing_analysis"] = {
            "analysis": pricing_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_channels_and_distribution(self, state: ResearchState, 
                                               context: Dict[str, Any]) -> ResearchState:
        """Analyze distribution channels and strategies."""
        self.logger.info("Analyzing channels and distribution")
        
        tool_calls = [
            ToolCall(
                tool_name="channel_analysis_tool",
                parameters={
                    "channel_types": ["direct", "retail", "online", "wholesale", "partner"],
                    "channel_metrics": ["reach", "cost", "effectiveness", "customer_satisfaction"],
                    "omnichannel_analysis": True
                }
            ),
            ToolCall(
                tool_name="e_commerce_analytics",
                parameters={
                    "platform_analysis": ["marketplace", "direct_to_consumer", "social_commerce"],
                    "metrics": ["conversion_rates", "customer_acquisition_cost", "lifetime_value"],
                    "trend_analysis": "digital_transformation"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        channel_analysis = await self.generate_response(
            state,
            """基于渠道和分销数据，请进行策略分析：

1. 渠道组合分析：
   - 现有渠道结构和贡献
   - 各渠道成本效益分析
   - 渠道覆盖范围和深度
   - 渠道冲突和协同效应

2. 数字化渠道分析：
   - 电商平台表现和机会
   - 直销vs平台策略
   - 移动商务和社交电商
   - 数字化转型进展

3. 渠道效率分析：
   - 渠道成本结构分析
   - 库存和物流效率
   - 客户获取成本比较
   - 渠道投资回报率

4. 全渠道策略：
   - 线上线下整合机会
   - 客户体验一致性
   - 数据和库存共享
   - 全渠道客户旅程

5. 渠道优化建议：
   - 渠道组合优化策略
   - 新兴渠道机会
   - 渠道伙伴关系管理
   - 分销策略改进

请提供渠道策略优化建议。"""
        )
        
        state.collected_data["channel_analysis"] = {
            "analysis": channel_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_geographic_markets(self, state: ResearchState, 
                                        context: Dict[str, Any]) -> ResearchState:
        """Analyze geographic market opportunities."""
        self.logger.info("Analyzing geographic markets")
        
        tool_calls = [
            ToolCall(
                tool_name="market_research_api",
                parameters={
                    "analysis_type": "geographic_analysis",
                    "regions": ["north_america", "europe", "asia_pacific", "latin_america"],
                    "market_metrics": ["size", "growth", "competition", "barriers"],
                    "localization_factors": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        geographic_analysis = await self.generate_response(
            state,
            """基于地理市场数据，请进行区域分析：

1. 区域市场评估：
   - 各区域市场规模和增长
   - 市场成熟度和发展阶段
   - 竞争强度和主要参与者
   - 进入壁垒和监管环境

2. 消费者差异分析：
   - 区域消费者偏好差异
   - 文化和社会因素影响
   - 购买力和支付习惯
   - 本土化需求和适应性

3. 市场机会排序：
   - 高潜力市场识别
   - 市场进入优先级
   - 投资回报预期
   - 风险调整机会评估

4. 本土化策略：
   - 产品本土化需求
   - 营销和品牌适应
   - 渠道和合作伙伴选择
   - 运营模式调整

5. 扩张策略建议：
   - 市场进入模式选择
   - 时机和节奏规划
   - 资源配置建议
   - 风险缓释措施

请提供地理扩张策略建议。"""
        )
        
        state.collected_data["geographic_analysis"] = {
            "analysis": geographic_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_market_trends(self, state: ResearchState, 
                                   context: Dict[str, Any]) -> ResearchState:
        """Analyze market trends and future outlook."""
        self.logger.info("Analyzing market trends")
        
        tool_calls = [
            ToolCall(
                tool_name="trend_analysis_api",
                parameters={
                    "trend_categories": ["technology", "social", "economic", "environmental"],
                    "trend_horizon": "5_years",
                    "impact_assessment": True,
                    "emerging_trends": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        trend_analysis = await self.generate_response(
            state,
            """基于市场趋势数据，请进行前瞻性分析：

1. 技术趋势影响：
   - 数字化和自动化趋势
   - 人工智能和数据分析应用
   - 新兴技术采用模式
   - 技术颠覆风险和机遇

2. 社会文化趋势：
   - 消费者价值观变化
   - 生活方式和行为模式演变
   - 代际差异和影响
   - 社会责任和可持续性关注

3. 经济环境趋势：
   - 宏观经济发展趋势
   - 收入分配和消费能力变化
   - 全球化vs本土化趋势
   - 经济周期和波动影响

4. 监管和政策趋势：
   - 监管环境变化方向
   - 政策支持和限制因素
   - 国际贸易和合作趋势
   - 环保和ESG要求强化

5. 未来市场展望：
   - 市场发展情景预测
   - 关键转折点识别
   - 新兴机会和威胁
   - 战略适应性建议

请提供市场趋势洞察和战略含义。"""
        )
        
        state.collected_data["trend_analysis"] = {
            "analysis": trend_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _synthesize_market_analysis(self, state: ResearchState, 
                                        context: Dict[str, Any]) -> ResearchState:
        """Synthesize all market analysis into strategic recommendations."""
        self.logger.info("Synthesizing market analysis")
        
        # Gather all analysis components
        analysis_components = []
        for key in ["market_sizing_analysis", "consumer_behavior_analysis", "demand_analysis", 
                   "pricing_analysis", "channel_analysis", "geographic_analysis", "trend_analysis"]:
            if key in state.collected_data:
                analysis_components.append(f"{key}: {state.collected_data[key]['analysis'][:400]}...")
        
        synthesis_prompt = f"""基于以下全面的市场分析结果，请提供综合性市场策略建议：

{chr(10).join(analysis_components)}

请提供：
1. 市场机会评估
   - 最具吸引力的市场机会
   - 市场进入和扩张策略
   - 目标客户群体优先级
   - 价值主张优化建议

2. 竞争定位策略
   - 差异化竞争优势
   - 市场定位建议
   - 品牌和营销策略
   - 竞争响应策略

3. 增长策略建议
   - 市场渗透策略
   - 产品开发机会
   - 市场开发计划
   - 多元化扩张选择

4. 运营策略优化
   - 定价策略优化
   - 渠道策略改进
   - 客户体验提升
   - 运营效率改善

5. 风险管理和应对
   - 主要市场风险识别
   - 风险缓释策略
   - 应急预案建议
   - 监控指标设置

请确保建议具有可操作性和战略价值。"""
        
        market_synthesis = await self.generate_response(state, synthesis_prompt)
        
        # Create comprehensive market report
        market_report = {
            "analysis_date": datetime.utcnow(),
            "market_type": context["market_type"],
            "geographic_focus": context["geographic_focus"],
            "executive_summary": market_synthesis,
            "detailed_analysis": {
                key: state.collected_data.get(key, {}).get("analysis", "")
                for key in ["market_sizing_analysis", "consumer_behavior_analysis", "demand_analysis", 
                           "pricing_analysis", "channel_analysis", "geographic_analysis", "trend_analysis"]
            },
            "key_opportunities": self._extract_opportunities(market_synthesis),
            "strategic_recommendations": self._extract_recommendations(market_synthesis),
            "market_risks": self._extract_market_risks(market_synthesis)
        }
        
        state.collected_data["market_analysis_report"] = market_report
        
        # Add final summary message
        summary_message = Message(
            role=MessageRole.ASSISTANT,
            content=f"完成了全面的市场分析，涵盖市场规模、消费者行为、需求预测、定价策略、渠道分析、地理市场和趋势分析等关键维度。",
            agent_id=self.agent_id,
            metadata={
                "analysis_type": "market_comprehensive",
                "market_type": context["market_type"],
                "components_analyzed": len(analysis_components)
            }
        )
        state.add_message(summary_message)
        
        return state
    
    def _extract_opportunities(self, analysis_text: str) -> List[str]:
        """Extract market opportunities from analysis text."""
        opportunities = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['机会', '潜力', '增长', 'opportunity']):
                opportunities.append(line.strip())
        
        return opportunities[:5]  # Return top 5 opportunities
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract strategic recommendations from analysis text."""
        recommendations = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['建议', '策略', '应该', 'recommend']):
                recommendations.append(line.strip())
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _extract_market_risks(self, analysis_text: str) -> List[str]:
        """Extract market risks from analysis text."""
        risks = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['风险', '威胁', '挑战', 'risk']):
                risks.append(line.strip())
        
        return risks[:3]  # Return top 3 risks