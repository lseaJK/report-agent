# Investment Research Reports System

A state-driven multi-agent AI system for generating comprehensive investment research reports, inspired by LangGraph architecture.

## 🏗️ Architecture Highlights

- **State-Driven Communication**: Agents communicate through shared state, not HTTP calls
- **Concurrent Execution**: Multiple agents work in parallel for maximum efficiency  
- **Robust Tool Calling**: Asynchronous, concurrent, and fault-tolerant external API calls
- **LangGraph-Style Workflow**: Structured workflow with clear state transitions

## Features

- **Multi-Agent Architecture**: Specialized agents for industry, financial, market, and risk analysis
- **SiliconCloud Integration**: Uses DeepSeek-V3.2 model via SiliconCloud API
- **Concurrent Tool Execution**: Parallel external data retrieval and processing
- **MySQL Database**: Optimized for MySQL 8.0.16+ with async operations
- **State Management**: Centralized state with message history and result tracking
- **Configurable Workflows**: Flexible workflow orchestration with error recovery

## Architecture

The system uses a state-driven multi-agent architecture inspired by LangGraph:

### Core Components

- **State Management**: Centralized `ResearchState` object for agent communication
- **Workflow Orchestrator**: Manages multi-agent execution flow
- **Tool Executor**: Concurrent and robust external API calls
- **Specialized Agents**: Domain-specific analysis agents
- **SiliconCloud LLM**: DeepSeek-V3.2 model integration

### Execution Flow

```
Task Creation → Agent Registration → Parallel Analysis → Result Synthesis → Report Generation
     ↓               ↓                    ↓                ↓                 ↓
  ResearchState → WorkflowOrchestrator → BaseAgent → ToolExecutor → Final Report
```

### Key Advantages

- **High Performance**: Parallel agent execution and concurrent tool calls
- **Fault Tolerance**: Isolated error handling with automatic retry
- **Scalability**: Easy to add new agents and tools
- **Observability**: Complete state tracking and execution logs

## Quick Start

### Prerequisites

- Python 3.9+
- MySQL 8.0.16+ database
- SiliconCloud API key for DeepSeek-V3.2 model
- Optional: Redis for caching

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd investment-research-reports
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
alembic upgrade head
```

6. Run the application:
```bash
python -m src.investment_research.main
```

The API will be available at `http://localhost:8000`

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `DATABASE_URL`: MySQL connection string
- `SILICONCLOUD_API_KEY`: SiliconCloud API key for DeepSeek-V3.2 model
- `MCP_SEARCH_ENDPOINT`: MCP search service endpoint
- `LANGCHAIN_API_KEY`: LangChain tracing API key (optional)

### Agent Configuration

Agents can be configured with:
- Available tools and data sources
- LLM parameters (temperature, max tokens)
- Domain-specific prompts and templates

## Usage

### Creating a Research Task

```python
from src.investment_research.core.workflow import research_workflow
from src.investment_research.agents.base import BaseAgent

# Create research task
state = await research_workflow.create_research_task(
    topic="Apple Inc. Investment Analysis",
    parameters={
        "analysis_depth": "comprehensive",
        "time_horizon": "12_months"
    }
)

# Register specialized agents
agents = [IndustryAgent(), FinancialAgent(), MarketAgent(), RiskAgent()]
for agent in agents:
    research_workflow.register_agent(agent)

# Execute workflow
result = await research_workflow.execute_research(state.task_id)
```

### Agent Communication via Shared State

Unlike traditional HTTP-based communication, agents share state:

```python
class FinancialAgent(BaseAgent):
    async def analyze(self, state: ResearchState) -> ResearchState:
        # Read industry analysis from shared state
        industry_data = state.analysis_results.get("industry", {})
        
        # Call tools concurrently
        tool_calls = [
            ToolCall(tool_name="mcp_search", parameters={"query": "AAPL financials"}),
            ToolCall(tool_name="rag_retrieval", parameters={"query": "Apple revenue"})
        ]
        state = await self.call_tools(state, tool_calls)
        
        # Generate analysis and update state
        analysis = await self.generate_response(state, "Analyze Apple's financials")
        state.analysis_results["financial"] = {"analysis": analysis}
        
        return state
```

## Development

### Project Structure

```
src/investment_research/
├── config/          # Configuration and settings
├── core/            # Core framework (database, models, LangChain)
├── agents/          # Multi-agent system
├── services/        # Data services (MCP, RAG)
└── main.py          # Application entry point

tests/               # Test suite
alembic/             # Database migrations
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue in the repository.