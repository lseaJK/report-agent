# 投资研究报告系统 - 完整设置和验证指南

## 🚀 快速开始

### 1. 环境准备

#### Python环境
```bash
# 确保Python 3.9+
python --version

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 必需的外部服务

1. **MySQL 8.0.16+**
   ```bash
   # 安装MySQL (示例 - Ubuntu)
   sudo apt update
   sudo apt install mysql-server-8.0
   
   # 启动MySQL服务
   sudo systemctl start mysql
   sudo systemctl enable mysql
   
   # 创建数据库
   mysql -u root -p
   CREATE DATABASE investment_research CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   CREATE USER 'research_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON investment_research.* TO 'research_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

2. **SiliconCloud API密钥**
   - 访问 [SiliconCloud](https://siliconflow.cn) 注册账号
   - 获取API密钥
   - 确保账户有足够的额度

### 2. 配置环境变量

创建 `.env` 文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
# 数据库配置
DATABASE_URL=mysql+aiomysql://research_user:your_password@localhost:3306/investment_research
DATABASE_ECHO=false

# SiliconCloud API配置
SILICONCLOUD_API_KEY=your_actual_api_key_here
SILICONCLOUD_MODEL=deepseek-ai/DeepSeek-V3.2
SILICONCLOUD_TEMPERATURE=0.7
SILICONCLOUD_MAX_TOKENS=4096

# MCP搜索服务 (可选)
MCP_SEARCH_ENDPOINT=http://localhost:8080
MCP_SEARCH_API_KEY=your_mcp_api_key

# 应用配置
APP_NAME=Investment Research Reports System
DEBUG=true
LOG_LEVEL=INFO
```

### 3. 数据库初始化

```bash
# 运行数据库迁移
alembic upgrade head
```

### 4. 验证安装

运行完整验证：
```bash
python scripts/verify_apis.py
```

## 🔍 验证检查清单

### ✅ 核心组件验证

运行以下命令验证各组件：

#### 1. 基础配置验证
```bash
python -c "from src.investment_research.config.settings import settings; print('✅ 配置加载成功')"
```

#### 2. 数据库连接验证
```bash
python -c "
from src.investment_research.core.database import get_engine
engine = get_engine()
print('✅ 数据库引擎创建成功')
print(f'数据库URL: {engine.url}')
"
```

#### 3. SiliconCloud API验证
```bash
# 设置API密钥
export SILICONCLOUD_API_KEY="your_api_key_here"

# 验证API调用
python -c "
import asyncio
from src.investment_research.core.siliconcloud_llm import SiliconCloudLLM

async def test():
    llm = SiliconCloudLLM()
    try:
        response = await llm._acall('你好，请简单介绍一下自己')
        print('✅ SiliconCloud API调用成功')
        print(f'响应: {response[:100]}...')
        await llm.aclose()
    except Exception as e:
        print(f'❌ API调用失败: {e}')

asyncio.run(test())
"
```

#### 4. 多智能体系统验证
```bash
python examples/langgraph_approach.py
```

### 🛠️ 故障排除

#### 常见问题及解决方案

1. **SiliconCloud API调用失败**
   ```
   错误: SILICONCLOUD_API_KEY environment variable is required
   解决: 设置正确的API密钥环境变量
   ```

2. **数据库连接失败**
   ```
   错误: No module named 'aiomysql'
   解决: pip install aiomysql
   
   错误: Can't connect to MySQL server
   解决: 检查MySQL服务是否运行，连接字符串是否正确
   ```

3. **MCP服务不可用**
   ```
   错误: 无法连接到MCP端点
   解决: MCP服务是可选的，可以跳过或配置正确的端点
   ```

#### 依赖问题解决

```bash
# 如果遇到依赖冲突，重新安装
pip uninstall -y -r requirements.txt
pip install -r requirements.txt

# 或者使用虚拟环境重新开始
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🧪 测试验证

### 单元测试
```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行集成测试
python tests/test_integration.py
```

### 功能测试
```bash
# 测试多智能体工作流
python examples/langgraph_approach.py

# 验证API和服务
python scripts/verify_apis.py
```

### 性能测试
```bash
# 测试并发工具调用
python -c "
import asyncio
from src.investment_research.core.tool_executor import ToolExecutor
from src.investment_research.core.state import ToolCall

async def test_concurrent():
    executor = ToolExecutor()
    calls = [
        ToolCall(tool_name='data_validation', parameters={'data': {'test': i}})
        for i in range(10)
    ]
    
    import time
    start = time.time()
    results = await executor.execute_tools_concurrent(calls)
    end = time.time()
    
    print(f'✅ 并发执行10个工具调用耗时: {end-start:.2f}秒')
    print(f'成功率: {sum(1 for r in results if r.success)}/{len(results)}')

asyncio.run(test_concurrent())
"
```

## 📊 验证结果解读

### 完全成功的验证结果
```
📊 总体结果: 7/7 项验证通过
🎉 所有验证通过! 系统准备就绪。
```

### 部分成功的验证结果
```
📊 总体结果: 5/7 项验证通过
⚠️  部分验证失败，请检查配置和服务状态。
```

**可接受的失败项**:
- MCP搜索服务 (可选功能)
- 某些外部API (如果没有密钥)

**必须成功的项**:
- LangChain集成
- 工具执行系统  
- 状态管理系统
- SiliconCloud API (如果要使用AI功能)

## 🚀 启动系统

### 开发模式
```bash
# 启动FastAPI开发服务器
python -m src.investment_research.main

# 或使用uvicorn
uvicorn src.investment_research.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产模式
```bash
# 使用gunicorn (需要安装)
pip install gunicorn
gunicorn src.investment_research.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📝 使用示例

### 创建研究任务
```python
import asyncio
from src.investment_research.core.workflow import research_workflow
from examples.langgraph_approach import IndustryAgent, FinancialAgent, MarketAgent, RiskAgent

async def create_research():
    # 注册智能体
    agents = [IndustryAgent(), FinancialAgent(), MarketAgent(), RiskAgent()]
    for agent in agents:
        research_workflow.register_agent(agent)
    
    # 创建研究任务
    state = await research_workflow.create_research_task(
        topic="苹果公司(AAPL)投资分析",
        parameters={"analysis_depth": "comprehensive"}
    )
    
    print(f"任务创建成功: {state.task_id}")
    return state.task_id

# 运行示例
task_id = asyncio.run(create_research())
```

### API访问
```bash
# 启动服务后访问
curl http://localhost:8000/
curl http://localhost:8000/health
```

## 🔧 高级配置

### 自定义智能体
```python
from src.investment_research.agents.base import BaseAgent
from src.investment_research.core.models import AgentType

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="custom_agent",
            agent_type=AgentType.INDUSTRY  # 或其他类型
        )
    
    def get_system_prompt(self) -> str:
        return "你是一个自定义的分析智能体..."
    
    def get_required_tools(self) -> list[str]:
        return ["mcp_search", "rag_retrieval"]
    
    async def analyze(self, state):
        # 自定义分析逻辑
        return state
```

### 自定义工具
```python
from src.investment_research.core.tool_executor import BaseTool, ToolConfig

class CustomTool(BaseTool):
    def __init__(self):
        config = ToolConfig(
            name="custom_tool",
            description="自定义工具",
            required_params=["param1"]
        )
        super().__init__(config)
    
    async def _execute(self, parameters):
        # 自定义工具逻辑
        return {"result": "custom_result"}

# 注册工具
from src.investment_research.core.tool_executor import tool_executor
tool_executor.register_tool(CustomTool())
```

## 📚 更多资源

- [架构文档](docs/architecture.md)
- [API文档](http://localhost:8000/docs) (启动服务后访问)
- [示例代码](examples/)
- [测试用例](tests/)

## 🆘 获取帮助

如果遇到问题：

1. 检查日志输出
2. 运行验证脚本诊断问题
3. 查看故障排除部分
4. 检查环境变量配置
5. 确认所有依赖已正确安装