#!/usr/bin/env python3
"""Integration tests to verify all components work correctly."""

import os
import sys
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
import httpx

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.investment_research.core.siliconcloud_llm import SiliconCloudLLM
from src.investment_research.core.langchain_setup import create_llm, create_agent_llm
from src.investment_research.services.mcp_search import MCPSearchService, SearchQuery
from src.investment_research.services.rag_service import RAGService
from src.investment_research.core.tool_executor import ToolExecutor, MCPSearchTool, RAGRetrievalTool
from src.investment_research.core.state import ResearchState, ToolCall
from src.investment_research.config.settings import settings


class TestSiliconCloudIntegration:
    """Test SiliconCloud LLM integration."""
    
    def test_siliconcloud_llm_creation(self):
        """Test SiliconCloud LLM can be created."""
        llm = SiliconCloudLLM(
            model="deepseek-ai/DeepSeek-V3.2",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert llm.model == "deepseek-ai/DeepSeek-V3.2"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1000
        assert llm._llm_type == "siliconcloud"
    
    def test_api_key_validation(self):
        """Test API key validation."""
        llm = SiliconCloudLLM()
        
        # Without API key, should raise error
        with pytest.raises(ValueError, match="SILICONCLOUD_API_KEY"):
            llm._get_api_key()
    
    @patch.dict(os.environ, {"SILICONCLOUD_API_KEY": "test-key-123"})
    def test_api_key_found(self):
        """Test API key is found when set."""
        llm = SiliconCloudLLM()
        api_key = llm._get_api_key()
        assert api_key == "test-key-123"
    
    @patch.dict(os.environ, {"SILICONCLOUD_API_KEY": "test-key-123"})
    async def test_siliconcloud_api_call_structure(self):
        """Test SiliconCloud API call structure (mocked)."""
        llm = SiliconCloudLLM()
        
        # Mock the HTTP response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "这是一个测试响应"
                    }
                }
            ]
        }
        
        with patch.object(llm.client, 'post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = await llm._acall("测试提示")
            
            # Verify API call was made with correct structure
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL
            assert call_args[0][0] == "https://api.siliconflow.cn/v1/chat/completions"
            
            # Check headers
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key-123"
            assert headers["Content-Type"] == "application/json"
            
            # Check payload structure
            payload = call_args[1]["json"]
            assert payload["model"] == "deepseek-ai/DeepSeek-V3.2"
            assert payload["messages"] == [{"role": "user", "content": "测试提示"}]
            assert payload["stream"] is False
            assert "temperature" in payload
            assert "max_tokens" in payload
            
            assert result == "这是一个测试响应"


class TestLangChainIntegration:
    """Test LangChain integration."""
    
    def test_create_llm_without_api_key(self):
        """Test LLM creation works without API key (for testing)."""
        # This should not raise an error during creation
        llm = create_llm()
        assert llm is not None
        assert isinstance(llm, SiliconCloudLLM)
    
    def test_create_agent_llms(self):
        """Test agent-specific LLM creation."""
        agent_types = ["industry", "financial", "market", "risk"]
        
        for agent_type in agent_types:
            llm = create_agent_llm(agent_type)
            assert llm is not None
            assert isinstance(llm, SiliconCloudLLM)
            
            # Check agent-specific configurations
            if agent_type == "financial":
                assert llm.temperature == 0.3  # More deterministic
            elif agent_type == "risk":
                assert llm.temperature == 0.3  # Conservative
            elif agent_type == "market":
                assert llm.temperature == 0.7  # Default
    
    @patch.dict(os.environ, {"SILICONCLOUD_API_KEY": "test-key-123"})
    async def test_langchain_llm_call(self):
        """Test LangChain LLM call integration."""
        llm = create_llm()
        
        # Mock the API response
        mock_response = {
            "choices": [{"message": {"content": "LangChain集成测试成功"}}]
        }
        
        with patch.object(llm.client, 'post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = await llm._acall("测试LangChain集成")
            assert result == "LangChain集成测试成功"


class TestMCPIntegration:
    """Test MCP search integration."""
    
    async def test_mcp_service_creation(self):
        """Test MCP service can be created."""
        service = MCPSearchService()
        assert service.endpoint == settings.mcp_search.endpoint
        assert service.timeout == settings.mcp_search.timeout
        await service.close()
    
    async def test_mcp_search_structure(self):
        """Test MCP search request structure (mocked)."""
        service = MCPSearchService()
        
        query = SearchQuery(
            query="AAPL financial data",
            domain="financial",
            limit=5
        )
        
        mock_response = {
            "symbol": "AAPL",
            "data": {"revenue": "100B", "profit": "20B"},
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "mcp",
            "quality_score": 0.9
        }
        
        with patch.object(service.client, 'post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = await service.search_market_data(query)
            
            # Verify request structure
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL
            expected_url = f"{service.endpoint}/search/market"
            assert call_args[0][0] == expected_url
            
            # Check payload
            payload = call_args[1]["json"]
            assert payload["query"] == "AAPL financial data"
            assert payload["domain"] == "financial"
            assert payload["limit"] == 5
            
            # Check result
            assert result.symbol == "AAPL"
            assert result.quality_score == 0.9
        
        await service.close()
    
    async def test_mcp_error_handling(self):
        """Test MCP service error handling."""
        service = MCPSearchService()
        
        query = SearchQuery(query="test")
        
        # Test HTTP error
        with patch.object(service.client, 'post') as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "API Error", request=None, response=None
            )
            
            with pytest.raises(Exception, match="MCP search HTTP error"):
                await service.search_market_data(query)
        
        await service.close()


class TestRAGIntegration:
    """Test RAG service integration."""
    
    async def test_rag_service_creation(self):
        """Test RAG service can be created."""
        service = RAGService()
        assert service.embedding_dimension == 1536
    
    async def test_rag_relevance_calculation(self):
        """Test RAG relevance calculation."""
        service = RAGService()
        
        # Test relevance calculation
        query = "Apple financial performance"
        content = "Apple reported strong financial performance with revenue growth"
        
        relevance = service._calculate_relevance(query, content)
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0  # Should have some relevance
    
    async def test_rag_quality_assessment(self):
        """Test RAG quality assessment."""
        service = RAGService()
        
        # Test different content lengths
        short_content = "Short"
        medium_content = "This is a medium length content for testing quality assessment."
        long_content = "This is a very long content " * 100
        
        short_quality = service._assess_quality(short_content)
        medium_quality = service._assess_quality(medium_content)
        long_quality = service._assess_quality(long_content)
        
        assert short_quality < medium_quality < long_quality
        assert all(0.0 <= q <= 1.0 for q in [short_quality, medium_quality, long_quality])


class TestToolExecutorIntegration:
    """Test tool executor integration."""
    
    async def test_tool_executor_creation(self):
        """Test tool executor can be created."""
        executor = ToolExecutor()
        
        # Check default tools are registered
        tools = executor.list_tools()
        expected_tools = ["mcp_search", "rag_retrieval", "data_validation"]
        
        for tool in expected_tools:
            assert tool in tools
    
    async def test_tool_parameter_validation(self):
        """Test tool parameter validation."""
        executor = ToolExecutor()
        
        # Test MCP search tool
        mcp_tool = executor.get_tool("mcp_search")
        assert mcp_tool is not None
        
        # Valid parameters
        valid_params = {"query": "test query", "call_id": "test-123"}
        assert mcp_tool.validate_parameters(valid_params) is True
        
        # Missing required parameter
        invalid_params = {"call_id": "test-123"}
        with pytest.raises(ValueError, match="Required parameter 'query' missing"):
            mcp_tool.validate_parameters(invalid_params)
    
    async def test_concurrent_tool_execution(self):
        """Test concurrent tool execution."""
        executor = ToolExecutor()
        
        # Create multiple tool calls
        tool_calls = [
            ToolCall(
                tool_name="data_validation",
                parameters={"data": {"test": "data1"}}
            ),
            ToolCall(
                tool_name="data_validation", 
                parameters={"data": {"test": "data2"}}
            )
        ]
        
        # Execute concurrently
        results = await executor.execute_tools_concurrent(tool_calls)
        
        assert len(results) == 2
        for result in results:
            assert result.tool_name == "data_validation"
            assert result.success is True  # Data validation should succeed


class TestStateManagement:
    """Test state management."""
    
    async def test_research_state_creation(self):
        """Test research state creation."""
        state = ResearchState(
            topic="Test Research",
            parameters={"depth": "basic"}
        )
        
        assert state.topic == "Test Research"
        assert state.parameters["depth"] == "basic"
        assert len(state.messages) == 0
        assert len(state.agent_results) == 0
        assert len(state.tool_calls) == 0
    
    async def test_state_tool_call_management(self):
        """Test state tool call management."""
        state = ResearchState(topic="Test")
        
        # Add tool call
        tool_call = ToolCall(
            tool_name="test_tool",
            parameters={"param": "value"}
        )
        
        call_id = state.add_tool_call(tool_call)
        assert call_id == tool_call.id
        assert len(state.tool_calls) == 1
        
        # Add tool result
        from src.investment_research.core.state import ToolResult
        result = ToolResult(
            call_id=call_id,
            tool_name="test_tool",
            success=True,
            result="test result"
        )
        
        state.add_tool_result(result)
        assert len(state.tool_results) == 1
        assert state.tool_results[call_id].success is True


class TestDatabaseConfiguration:
    """Test database configuration."""
    
    def test_mysql_configuration(self):
        """Test MySQL configuration is correct."""
        db_url = settings.database.url
        assert "mysql+aiomysql://" in db_url
        assert "investment_research" in db_url
    
    def test_database_models_import(self):
        """Test database models can be imported."""
        from src.investment_research.core.models import (
            ResearchTask, AnalysisResult, Source, ReportTemplate,
            KBDocument, Tool, DataSource
        )
        
        # Test enum imports
        from src.investment_research.core.models import (
            TaskStatus, SourceType, AgentType, OutputFormat
        )
        
        # Verify enums have expected values
        assert TaskStatus.PENDING == "pending"
        assert AgentType.FINANCIAL == "financial"
        assert SourceType.MCP == "mcp"


async def run_integration_tests():
    """Run all integration tests."""
    print("🧪 运行集成测试...")
    print("=" * 60)
    
    test_classes = [
        TestSiliconCloudIntegration,
        TestLangChainIntegration,
        TestMCPIntegration,
        TestRAGIntegration,
        TestToolExecutorIntegration,
        TestStateManagement,
        TestDatabaseConfiguration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 测试类: {test_class.__name__}")
        
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            method = getattr(instance, method_name)
            
            try:
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{method_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed_tests}/{total_tests} 通过")
    
    if failed_tests:
        print(f"\n❌ 失败的测试:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print("\n🎉 所有测试通过!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)