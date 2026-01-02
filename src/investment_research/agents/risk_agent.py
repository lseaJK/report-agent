"""Risk assessment agent for comprehensive risk analysis and management."""

import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..core.models import AgentType
from ..core.state import ResearchState, Message, MessageRole, ToolCall
from .base import BaseAgent

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    """Risk category enumeration."""
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    REGULATORY_RISK = "regulatory_risk"
    REPUTATIONAL_RISK = "reputational_risk"
    STRATEGIC_RISK = "strategic_risk"
    TECHNOLOGY_RISK = "technology_risk"


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAgent(BaseAgent):
    """Specialized agent for risk assessment, analysis, and management recommendations."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, **kwargs):
        """Initialize the risk agent."""
        super().__init__(agent_id, agent_type, **kwargs)
        
        # Risk analysis areas
        self.risk_categories = [category.value for category in RiskCategory]
        
        # Risk assessment frameworks
        self.assessment_frameworks = [
            "quantitative_risk_models",
            "scenario_analysis",
            "stress_testing",
            "monte_carlo_simulation",
            "var_analysis",
            "sensitivity_analysis"
        ]
        
        # Risk data sources
        self.preferred_data_sources = [
            "market_volatility_data",
            "credit_rating_agencies",
            "regulatory_databases",
            "operational_incident_data",
            "compliance_reports",
            "industry_risk_reports"
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for risk analysis."""
        return """你是一个专业的风险管理分析师，专门负责全面的风险识别、评估和管理建议。

你的主要职责包括：
1. 市场风险评估和量化分析
2. 信用风险识别和评级分析
3. 操作风险评估和控制建议
4. 流动性风险分析和管理
5. 监管合规风险评估
6. 声誉风险识别和防范
7. 战略风险分析和应对
8. 技术和网络安全风险评估

分析时请：
- 使用多种风险评估方法和模型
- 进行定量和定性风险分析
- 考虑风险之间的相关性和传染效应
- 评估风险对业务目标的潜在影响
- 提供风险缓释和管理策略
- 建立风险监控和预警机制
- 考虑监管要求和最佳实践

请以系统性、前瞻性的方式进行风险分析，并提供可操作的风险管理建议。"""
    
    def get_required_tools(self) -> List[str]:
        """Get list of tools required for risk analysis."""
        return [
            "risk_assessment_api",
            "market_volatility_calculator",
            "credit_rating_api",
            "regulatory_compliance_checker",
            "operational_risk_database",
            "scenario_analysis_tool",
            "stress_testing_engine",
            "risk_correlation_analyzer"
        ]
    
    async def analyze(self, state: ResearchState) -> ResearchState:
        """Perform comprehensive risk analysis."""
        self.logger.info(f"Starting risk analysis for topic: {state.topic}")
        
        # Extract risk context from topic
        risk_context = self._extract_risk_context(state.topic)
        
        # Phase 1: Risk Identification and Classification
        state = await self._identify_and_classify_risks(state, risk_context)
        
        # Phase 2: Market Risk Analysis
        state = await self._analyze_market_risks(state, risk_context)
        
        # Phase 3: Credit Risk Assessment
        state = await self._assess_credit_risks(state, risk_context)
        
        # Phase 4: Operational Risk Evaluation
        state = await self._evaluate_operational_risks(state, risk_context)
        
        # Phase 5: Regulatory and Compliance Risk Analysis
        state = await self._analyze_regulatory_risks(state, risk_context)
        
        # Phase 6: Scenario Analysis and Stress Testing
        state = await self._perform_scenario_analysis(state, risk_context)
        
        # Phase 7: Risk Correlation and Portfolio Effects
        state = await self._analyze_risk_correlations(state, risk_context)
        
        # Phase 8: Risk Management Recommendations
        state = await self._synthesize_risk_management(state, risk_context)
        
        self.logger.info("Risk analysis completed")
        return state
    
    def _extract_risk_context(self, topic: str) -> Dict[str, Any]:
        """Extract risk context from research topic."""
        context = {
            "analysis_scope": "comprehensive",
            "risk_appetite": "moderate",
            "time_horizon": "1_year",
            "analysis_date": datetime.utcnow()
        }
        
        # Detect primary risk focus
        topic_lower = topic.lower()
        if "market" in topic_lower:
            context["primary_focus"] = RiskCategory.MARKET_RISK
        elif "credit" in topic_lower:
            context["primary_focus"] = RiskCategory.CREDIT_RISK
        elif "operational" in topic_lower or "operation" in topic_lower:
            context["primary_focus"] = RiskCategory.OPERATIONAL_RISK
        elif "regulatory" in topic_lower or "compliance" in topic_lower:
            context["primary_focus"] = RiskCategory.REGULATORY_RISK
        else:
            context["primary_focus"] = "comprehensive"
        
        return context
    
    async def _identify_and_classify_risks(self, state: ResearchState, 
                                         context: Dict[str, Any]) -> ResearchState:
        """Identify and classify all relevant risks."""
        self.logger.info("Identifying and classifying risks")
        
        tool_calls = [
            ToolCall(
                tool_name="risk_assessment_api",
                parameters={
                    "assessment_type": "risk_identification",
                    "risk_categories": self.risk_categories,
                    "industry_specific": True,
                    "emerging_risks": True
                }
            ),
            ToolCall(
                tool_name="operational_risk_database",
                parameters={
                    "query_type": "risk_taxonomy",
                    "risk_sources": ["internal", "external", "systemic"],
                    "time_period": "last_5_years"
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        risk_identification = await self.generate_response(
            state,
            """基于风险识别数据，请进行全面的风险分类和评估：

1. 风险清单和分类：
   - 市场风险因素识别
   - 信用风险暴露分析
   - 操作风险事件类型
   - 流动性风险来源
   - 监管合规风险点
   - 声誉风险触发因素
   - 战略风险要素
   - 技术和网络风险

2. 风险优先级排序：
   - 风险影响程度评估
   - 风险发生概率分析
   - 风险暴露度量化
   - 关键风险因素识别

3. 新兴风险识别：
   - 行业特定新兴风险
   - 技术变革带来的风险
   - 监管环境变化风险
   - 地缘政治风险因素

4. 风险相互关系：
   - 风险之间的关联性
   - 风险传染路径分析
   - 系统性风险因素
   - 风险放大效应

请提供结构化的风险清单和初步评估。"""
        )
        
        state.collected_data["risk_identification"] = {
            "analysis": risk_identification,
            "timestamp": datetime.utcnow(),
            "risk_categories": self.risk_categories
        }
        
        return state
    
    async def _analyze_market_risks(self, state: ResearchState, 
                                  context: Dict[str, Any]) -> ResearchState:
        """Analyze market risks and volatility."""
        self.logger.info("Analyzing market risks")
        
        tool_calls = [
            ToolCall(
                tool_name="market_volatility_calculator",
                parameters={
                    "risk_factors": ["equity", "interest_rate", "currency", "commodity"],
                    "volatility_models": ["historical", "implied", "garch"],
                    "time_horizons": ["1_day", "1_week", "1_month", "1_year"],
                    "confidence_levels": [0.95, 0.99]
                }
            ),
            ToolCall(
                tool_name="scenario_analysis_tool",
                parameters={
                    "scenario_types": ["historical", "hypothetical", "monte_carlo"],
                    "market_shocks": ["equity_crash", "interest_rate_spike", "currency_crisis"],
                    "correlation_analysis": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        market_risk_analysis = await self.generate_response(
            state,
            """基于市场风险数据，请进行深入分析：

1. 市场风险量化：
   - VaR（风险价值）计算和分析
   - 预期损失和尾部风险
   - 波动率分析和预测
   - 敏感性分析结果

2. 各类市场风险评估：
   - 股票市场风险暴露
   - 利率风险影响分析
   - 汇率风险评估
   - 商品价格风险

3. 市场风险因子分析：
   - 主要风险驱动因素
   - 风险因子相关性
   - 系统性vs特异性风险
   - 风险集中度分析

4. 压力测试结果：
   - 极端市场情景影响
   - 历史危机情景重现
   - 多因子压力测试
   - 反向压力测试

5. 市场风险管理：
   - 风险限额设置建议
   - 对冲策略评估
   - 风险监控指标
   - 风险报告机制

请提供市场风险的量化评估和管理建议。"""
        )
        
        state.collected_data["market_risk_analysis"] = {
            "analysis": market_risk_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _assess_credit_risks(self, state: ResearchState, 
                                 context: Dict[str, Any]) -> ResearchState:
        """Assess credit risks and counterparty exposures."""
        self.logger.info("Assessing credit risks")
        
        tool_calls = [
            ToolCall(
                tool_name="credit_rating_api",
                parameters={
                    "analysis_type": "credit_assessment",
                    "rating_agencies": ["sp", "moody", "fitch"],
                    "credit_metrics": ["pd", "lgd", "ead", "expected_loss"],
                    "portfolio_analysis": True
                }
            ),
            ToolCall(
                tool_name="risk_assessment_api",
                parameters={
                    "risk_type": "credit_risk",
                    "assessment_methods": ["fundamental", "market_based", "hybrid"],
                    "concentration_analysis": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        credit_risk_analysis = await self.generate_response(
            state,
            """基于信用风险数据，请进行全面评估：

1. 信用风险量化：
   - 违约概率（PD）评估
   - 违约损失率（LGD）分析
   - 违约暴露（EAD）计算
   - 预期损失和非预期损失

2. 信用评级分析：
   - 当前信用评级状况
   - 评级变化趋势和驱动因素
   - 评级迁移概率矩阵
   - 评级下调风险评估

3. 对手方风险评估：
   - 主要对手方信用质量
   - 对手方集中度风险
   - 担保和抵押品分析
   - 净额结算效果

4. 信用风险集中度：
   - 行业集中度分析
   - 地理集中度风险
   - 单一对手方风险
   - 关联方风险暴露

5. 信用风险缓释：
   - 担保和抵押品价值
   - 信用衍生品使用
   - 保险和其他缓释工具
   - 风险转移效果评估

请提供信用风险评估结果和管理建议。"""
        )
        
        state.collected_data["credit_risk_analysis"] = {
            "analysis": credit_risk_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _evaluate_operational_risks(self, state: ResearchState, 
                                        context: Dict[str, Any]) -> ResearchState:
        """Evaluate operational risks and control effectiveness."""
        self.logger.info("Evaluating operational risks")
        
        tool_calls = [
            ToolCall(
                tool_name="operational_risk_database",
                parameters={
                    "risk_categories": ["people", "process", "systems", "external"],
                    "loss_data_analysis": True,
                    "control_assessment": True,
                    "time_period": "5_years"
                }
            ),
            ToolCall(
                tool_name="regulatory_compliance_checker",
                parameters={
                    "compliance_areas": ["operational_risk", "internal_controls", "governance"],
                    "regulatory_frameworks": ["basel", "sox", "local_regulations"],
                    "gap_analysis": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        operational_risk_analysis = await self.generate_response(
            state,
            """基于操作风险数据，请进行详细评估：

1. 操作风险分类分析：
   - 人员风险因素和控制
   - 流程风险识别和改进
   - 系统风险评估和备份
   - 外部事件风险应对

2. 历史损失分析：
   - 操作风险损失事件统计
   - 损失频率和严重程度
   - 损失趋势和模式识别
   - 根本原因分析

3. 内控体系评估：
   - 内控设计有效性
   - 内控执行情况检查
   - 控制缺陷识别
   - 内控改进建议

4. 业务连续性风险：
   - 关键业务流程识别
   - 单点故障风险分析
   - 灾难恢复能力评估
   - 业务连续性计划

5. 技术和网络风险：
   - IT系统安全风险
   - 网络安全威胁评估
   - 数据保护和隐私风险
   - 技术更新和维护风险

请提供操作风险管理改进建议。"""
        )
        
        state.collected_data["operational_risk_analysis"] = {
            "analysis": operational_risk_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_regulatory_risks(self, state: ResearchState, 
                                      context: Dict[str, Any]) -> ResearchState:
        """Analyze regulatory and compliance risks."""
        self.logger.info("Analyzing regulatory risks")
        
        tool_calls = [
            ToolCall(
                tool_name="regulatory_compliance_checker",
                parameters={
                    "regulatory_areas": ["financial", "operational", "environmental", "data_protection"],
                    "jurisdictions": ["domestic", "international"],
                    "compliance_status": "full_assessment",
                    "upcoming_regulations": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        regulatory_risk_analysis = await self.generate_response(
            state,
            """基于监管合规数据，请进行风险评估：

1. 当前合规状况：
   - 主要监管要求遵循情况
   - 合规缺陷和整改需求
   - 监管检查和处罚历史
   - 合规成本分析

2. 监管变化风险：
   - 即将生效的新法规
   - 监管趋势和方向
   - 合规要求变化影响
   - 适应新规的挑战

3. 跨境监管风险：
   - 多司法管辖区要求
   - 监管冲突和协调
   - 国际制裁风险
   - 跨境数据传输合规

4. 行业特定监管：
   - 行业监管特殊要求
   - 专业资质和许可
   - 行业标准和最佳实践
   - 自律组织要求

5. 合规风险管理：
   - 合规管理体系建设
   - 合规培训和文化
   - 合规监控和报告
   - 违规风险缓释

请提供监管合规风险管理建议。"""
        )
        
        state.collected_data["regulatory_risk_analysis"] = {
            "analysis": regulatory_risk_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _perform_scenario_analysis(self, state: ResearchState, 
                                       context: Dict[str, Any]) -> ResearchState:
        """Perform comprehensive scenario analysis and stress testing."""
        self.logger.info("Performing scenario analysis")
        
        tool_calls = [
            ToolCall(
                tool_name="scenario_analysis_tool",
                parameters={
                    "scenario_types": ["base", "adverse", "severely_adverse"],
                    "risk_factors": ["economic", "market", "operational", "regulatory"],
                    "time_horizons": ["1_year", "3_years", "5_years"],
                    "monte_carlo_simulations": 10000
                }
            ),
            ToolCall(
                tool_name="stress_testing_engine",
                parameters={
                    "stress_types": ["single_factor", "multi_factor", "reverse"],
                    "severity_levels": ["moderate", "severe", "extreme"],
                    "recovery_scenarios": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        scenario_analysis = await self.generate_response(
            state,
            """基于情景分析和压力测试结果，请进行综合评估：

1. 基准情景分析：
   - 正常经营环境下的风险水平
   - 预期损失和收益分布
   - 关键风险指标表现
   - 资本充足性评估

2. 不利情景分析：
   - 经济下行情景影响
   - 市场波动加剧影响
   - 监管环境恶化影响
   - 多重风险叠加效应

3. 极端情景分析：
   - 系统性危机情景
   - 黑天鹅事件影响
   - 尾部风险评估
   - 生存能力测试

4. 压力测试结果：
   - 单一因子压力测试
   - 多因子综合压力测试
   - 反向压力测试
   - 动态压力测试

5. 情景应对策略：
   - 风险缓释措施
   - 应急响应计划
   - 资本和流动性管理
   - 业务调整策略

请提供情景分析结论和应对建议。"""
        )
        
        state.collected_data["scenario_analysis"] = {
            "analysis": scenario_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _analyze_risk_correlations(self, state: ResearchState, 
                                       context: Dict[str, Any]) -> ResearchState:
        """Analyze risk correlations and portfolio effects."""
        self.logger.info("Analyzing risk correlations")
        
        tool_calls = [
            ToolCall(
                tool_name="risk_correlation_analyzer",
                parameters={
                    "correlation_types": ["linear", "nonlinear", "tail_dependence"],
                    "risk_categories": self.risk_categories,
                    "time_varying_correlations": True,
                    "portfolio_effects": True
                }
            )
        ]
        
        state = await self.call_tools(state, tool_calls)
        
        correlation_analysis = await self.generate_response(
            state,
            """基于风险相关性分析，请评估组合风险效应：

1. 风险相关性分析：
   - 不同风险类别间相关性
   - 相关性的时变特征
   - 尾部相关性和极端事件
   - 相关性破裂风险

2. 风险集中度评估：
   - 风险暴露集中程度
   - 多元化效果评估
   - 集中度风险指标
   - 风险分散策略

3. 组合风险效应：
   - 风险叠加和放大效应
   - 风险对冲和抵消效应
   - 组合风险vs单项风险
   - 边际风险贡献分析

4. 系统性风险评估：
   - 系统重要性评估
   - 风险传染路径
   - 系统性风险指标
   - 宏观审慎要求

5. 风险优化建议：
   - 风险配置优化
   - 相关性管理策略
   - 多元化改进建议
   - 风险预算分配

请提供风险组合管理建议。"""
        )
        
        state.collected_data["risk_correlation_analysis"] = {
            "analysis": correlation_analysis,
            "timestamp": datetime.utcnow()
        }
        
        return state
    
    async def _synthesize_risk_management(self, state: ResearchState, 
                                        context: Dict[str, Any]) -> ResearchState:
        """Synthesize all risk analysis into comprehensive risk management recommendations."""
        self.logger.info("Synthesizing risk management recommendations")
        
        # Gather all analysis components
        analysis_components = []
        for key in ["risk_identification", "market_risk_analysis", "credit_risk_analysis", 
                   "operational_risk_analysis", "regulatory_risk_analysis", "scenario_analysis", 
                   "risk_correlation_analysis"]:
            if key in state.collected_data:
                analysis_components.append(f"{key}: {state.collected_data[key]['analysis'][:400]}...")
        
        synthesis_prompt = f"""基于以下全面的风险分析结果，请提供综合性风险管理建议：

{chr(10).join(analysis_components)}

请提供：
1. 风险状况综合评估
   - 整体风险水平评级
   - 关键风险因素排序
   - 风险趋势判断
   - 风险承受能力评估

2. 风险管理策略
   - 风险偏好和容忍度设定
   - 风险限额和指标体系
   - 风险缓释和转移策略
   - 风险监控和报告机制

3. 风险治理建设
   - 风险管理组织架构
   - 风险管理政策制度
   - 风险文化建设
   - 风险管理能力提升

4. 应急预案和危机管理
   - 风险事件应急响应
   - 业务连续性保障
   - 危机沟通和声誉管理
   - 恢复和重建计划

5. 监管合规和最佳实践
   - 监管要求遵循
   - 行业最佳实践借鉴
   - 风险管理创新
   - 持续改进机制

请确保建议具有系统性和可操作性。"""
        
        risk_synthesis = await self.generate_response(state, synthesis_prompt)
        
        # Create comprehensive risk management report
        risk_report = {
            "analysis_date": datetime.utcnow(),
            "risk_focus": context["primary_focus"],
            "executive_summary": risk_synthesis,
            "detailed_analysis": {
                key: state.collected_data.get(key, {}).get("analysis", "")
                for key in ["risk_identification", "market_risk_analysis", "credit_risk_analysis", 
                           "operational_risk_analysis", "regulatory_risk_analysis", "scenario_analysis", 
                           "risk_correlation_analysis"]
            },
            "risk_rating": self._calculate_overall_risk_rating(state),
            "key_risks": self._extract_key_risks(risk_synthesis),
            "risk_recommendations": self._extract_risk_recommendations(risk_synthesis),
            "monitoring_indicators": self._extract_monitoring_indicators(risk_synthesis)
        }
        
        state.collected_data["risk_management_report"] = risk_report
        
        # Add final summary message
        summary_message = Message(
            role=MessageRole.ASSISTANT,
            content=f"完成了全面的风险分析，涵盖市场风险、信用风险、操作风险、监管风险等各个维度，并提供了系统性的风险管理建议。",
            agent_id=self.agent_id,
            metadata={
                "analysis_type": "risk_comprehensive",
                "risk_focus": context["primary_focus"],
                "components_analyzed": len(analysis_components)
            }
        )
        state.add_message(summary_message)
        
        return state
    
    def _calculate_overall_risk_rating(self, state: ResearchState) -> str:
        """Calculate overall risk rating based on analysis."""
        # This would use a sophisticated risk scoring model
        # For now, return a placeholder
        return "中等风险"
    
    def _extract_key_risks(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Extract key risks from analysis text."""
        # Simple extraction - in practice, this would use NLP
        risks = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['关键风险', '主要风险', '重要风险']):
                risks.append({
                    "description": line.strip(),
                    "category": "未分类",
                    "level": "中等"
                })
        
        return risks[:5]  # Return top 5 risks
    
    def _extract_risk_recommendations(self, analysis_text: str) -> List[str]:
        """Extract risk management recommendations."""
        recommendations = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['建议', '应该', '需要', '措施']):
                recommendations.append(line.strip())
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _extract_monitoring_indicators(self, analysis_text: str) -> List[str]:
        """Extract risk monitoring indicators."""
        indicators = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['指标', '监控', '预警', '测量']):
                indicators.append(line.strip())
        
        return indicators[:5]  # Return top 5 indicators