#!/usr/bin/env python3
"""Quick verification of core components that don't require external services."""

import os
import sys
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def verify_core_components():
    """Verify core components that can be tested without external dependencies."""
    print("🔍 快速验证核心组件")
    print("=" * 50)
    
    results = {}
    
    # 1. Configuration loading
    print("\n📋 验证配置加载...")
    try:
        from src.investment_research.config.settings import settings
        print(f"  ✅ 配置加载成功")
        print(f"  📊 数据库类型: {'MySQL' if 'mysql' in settings.database.url else 'Other'}")
        print(f"  🤖 AI模型: {settings.ai_service.model}")
        results["配置加载"] = True
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        results["配置加载"] = False
    
    # 2. Database models
    print("\n🗄️  验证数据库模型...")
    try:
        from src.investment_research.core.models import (
            ResearchTask, AnalysisResult, TaskStatus, AgentType
        )
        print(f"  ✅ 数据库模型导入成功")
        print(f"  📋 任务状态: {[s.value for s in TaskStatus]}")
        print(f"  🤖 智能体类型: {[a.value for a in AgentType]}")
        results["数据库模型"] = True
    except Exception as e:
        print(f"  ❌ 数据库模型导入失败: {e}")
        results["数据库模型"] = False
    
    # 3. LLM wrapper
    print("\n🤖 验证LLM包装器...")
    try:
        from src.investment_research.core.siliconcloud_llm import SiliconCloudLLM
        llm = SiliconCloudLLM(model="deepseek-ai/DeepSeek-V3.2", temperature=0.7)
        print(f"  ✅ SiliconCloud LLM创建成功")
        print(f"  📋 模型: {llm.model}")
        print(f"  🌡️  温度: {llm.temperature}")
        print(f"  🔧 LLM类型: {llm._llm_type}")
        results["LLM包装器"] = True
    except Exception as e:
        print(f"  ❌ LLM包装器创建失败: {e}")
        results["LLM包装器"] = False
    
    # 4. LangChain integration
    print("\n🦜 验证LangChain集成...")
    try:
        from src.investment_research.core.langchain_setup import create_llm, create_agent_llm
        
        # Test general LLM
        llm = create_llm()
        print(f"  ✅ 通用LLM创建成功: {type(llm).__name__}")
        
        # Test agent LLMs
        agent_types = ["industry", "financial", "market", "risk"]
        for agent_type in agent_types:
            agent_llm = create_agent_llm(agent_type)
            print(f"  ✅ {agent_type}智能体LLM: 温度={agent_llm.temperature}")
        
        results["LangChain集成"] = True
    except Exception as e:
        print(f"  ❌ LangChain集成失败: {e}")
        results["LangChain集成"] = False
    
    # 5. State management
    print("\n📊 验证状态管理...")
    try:
        from src.investment_research.core.state import (
            ResearchState, Message, MessageRole, ToolCall, state_manager
        )
        
        # Create state
        state = await state_manager.create_state(
            topic="测试研究",
            parameters={"test": True}
        )
        print(f"  ✅ 研究状态创建成功: {state.task_id[:8]}...")
        
        # Test state operations
        from src.investment_research.core.state import ToolResult
        tool_call = ToolCall(tool_name="test_tool", parameters={"test": "value"})
        call_id = state.add_tool_call(tool_call)
        
        result = ToolResult(call_id=call_id, tool_name="test_tool", success=True)
        state.add_tool_result(result)
        
        print(f"  ✅ 工具调用管理: {len(state.tool_calls)} 调用, {len(state.tool_results)} 结果")
        
        # Cleanup
        await state_manager.delete_state(state.task_id)
        print(f"  ✅ 状态清理完成")
        
        results["状态管理"] = True
    except Exception as e:
        print(f"  ❌ 状态管理失败: {e}")
        results["状态管理"] = False
    
    # 6. Tool system
    print("\n🔧 验证工具系统...")
    try:
        from src.investment_research.core.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        tools = executor.list_tools()
        print(f"  ✅ 工具执行器创建成功")
        print(f"  📋 可用工具: {', '.join(tools)}")
        
        # Test data validation tool (doesn't need external APIs)
        validation_tool = executor.get_tool("data_validation")
        if validation_tool:
            test_call = ToolCall(
                tool_name="data_validation",
                parameters={"data": {"revenue": 1000, "profit": 200}}
            )
            
            result = await executor.execute_tool(test_call)
            print(f"  ✅ 工具执行测试: 成功={result.success}")
            if result.result:
                print(f"  📊 数据质量评分: {result.result.get('quality_score', 'N/A')}")
        
        results["工具系统"] = True
    except Exception as e:
        print(f"  ❌ 工具系统失败: {e}")
        results["工具系统"] = False
    
    # 7. Agent base class
    print("\n🤖 验证智能体基类...")
    try:
        from src.investment_research.agents.base import BaseAgent
        from src.investment_research.core.models import AgentType
        
        # Create a simple test agent
        class TestAgent(BaseAgent):
            def __init__(self):
                super().__init__("test_agent", AgentType.INDUSTRY)
            
            def get_system_prompt(self):
                return "测试智能体"
            
            def get_required_tools(self):
                return ["data_validation"]
            
            async def analyze(self, state):
                return state
        
        agent = TestAgent()
        print(f"  ✅ 测试智能体创建成功: {agent.agent_id}")
        print(f"  📋 智能体类型: {agent.agent_type.value}")
        print(f"  🔧 所需工具: {agent.get_required_tools()}")
        
        results["智能体基类"] = True
    except Exception as e:
        print(f"  ❌ 智能体基类失败: {e}")
        results["智能体基类"] = False
    
    # 8. Workflow system
    print("\n🔄 验证工作流系统...")
    try:
        from src.investment_research.core.workflow import research_workflow
        
        # Test task status (without actually running workflow)
        state = await state_manager.create_state("工作流测试", {})
        status = await research_workflow.get_task_status(state.task_id)
        
        if status:
            print(f"  ✅ 工作流状态查询成功")
            print(f"  📈 当前步骤: {status['current_step']}")
            print(f"  📋 任务主题: {status['topic']}")
        
        await state_manager.delete_state(state.task_id)
        results["工作流系统"] = True
    except Exception as e:
        print(f"  ❌ 工作流系统失败: {e}")
        results["工作流系统"] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 验证结果总结:")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} {name}")
    
    print(f"\n📊 核心组件验证: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有核心组件验证通过!")
        print("💡 系统基础架构正常，可以进行下一步配置")
    else:
        print("⚠️  部分核心组件验证失败")
        print("💡 请检查Python环境和依赖安装")
    
    # Additional checks
    print("\n🔍 额外检查:")
    
    # Check if API key is set
    api_key = os.getenv("SILICONCLOUD_API_KEY")
    if api_key:
        print(f"  ✅ SiliconCloud API密钥已设置 ({api_key[:10]}...)")
    else:
        print("  ⚠️  SiliconCloud API密钥未设置")
        print("     设置方法: export SILICONCLOUD_API_KEY='your-api-key'")
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version >= (3, 9):
        print(f"  ✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ⚠️  Python版本过低: {python_version.major}.{python_version.minor}")
        print("     建议使用Python 3.9+")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(verify_core_components())
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 快速验证完成 - 系统核心组件正常!")
        print("📝 下一步:")
        print("   1. 设置 SILICONCLOUD_API_KEY 环境变量")
        print("   2. 配置MySQL数据库连接")
        print("   3. 运行 python scripts/verify_apis.py 进行完整验证")
        print("   4. 运行 python examples/langgraph_approach.py 测试多智能体系统")
    else:
        print("❌ 快速验证失败 - 请检查环境配置")
        print("💡 建议:")
        print("   1. 重新安装依赖: pip install -r requirements.txt")
        print("   2. 检查Python版本是否为3.9+")
        print("   3. 查看错误信息并修复相关问题")
    
    sys.exit(0 if success else 1)