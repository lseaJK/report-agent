"""Tests for RAG service functionality."""

import pytest
from datetime import datetime
from uuid import uuid4
from typing import List

from src.investment_research.services.rag_service import (
    RAGService, Document, Source, RetrievalContext, IndexingResult
)
from src.investment_research.core.models import KBDocument, Source as SourceModel, SourceType


class TestRAGService:
    """Test cases for RAG service."""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAG service instance."""
        return RAGService()
    
    @pytest.fixture
    async def sample_documents(self, test_db_session):
        """Create sample documents in the test database."""
        documents = []
        
        # Document 1: High quality financial analysis
        doc1 = KBDocument(
            document_id=uuid4(),
            title="Financial Analysis of Tech Sector",
            content="This comprehensive financial analysis examines the technology sector's performance in Q4 2023. The analysis includes revenue growth, profit margins, and market share data. Key findings indicate strong growth in cloud computing and AI-related services.",
            document_type="financial_analysis",
            domain="technology",
            tags=["finance", "technology", "Q4", "analysis"],
            quality_score=0.8,
            document_metadata={"author": "Financial Analyst", "department": "Research"}
        )
        
        # Document 2: Market research report
        doc2 = KBDocument(
            document_id=uuid4(),
            title="Market Research: Consumer Electronics",
            content="Market research report on consumer electronics trends. The report covers smartphone adoption rates, tablet market dynamics, and emerging wearable technology trends. Consumer preferences show increasing demand for sustainable and eco-friendly products.",
            document_type="market_research",
            domain="consumer_electronics",
            tags=["market", "consumer", "electronics", "trends"],
            quality_score=0.7,
            document_metadata={"source": "Market Research Firm", "year": "2023"}
        )
        
        # Document 3: Low quality document
        doc3 = KBDocument(
            document_id=uuid4(),
            title="Brief Note",
            content="Short note.",
            document_type="note",
            domain="general",
            tags=["note"],
            quality_score=0.2,
            document_metadata={}
        )
        
        documents = [doc1, doc2, doc3]
        
        for doc in documents:
            test_db_session.add(doc)
        
        await test_db_session.commit()
        return documents
    
    @pytest.mark.asyncio
    async def test_retrieve_context_basic(self, rag_service, sample_documents):
        """Test basic context retrieval."""
        # Test retrieving documents with a relevant query
        documents = await rag_service.retrieve_context("financial analysis technology")
        
        assert len(documents) >= 1
        assert any("Financial Analysis" in doc.title for doc in documents)
        
        # Check document structure
        for doc in documents:
            assert isinstance(doc, Document)
            assert doc.document_id
            assert doc.title
            assert doc.content
            assert doc.relevance_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_retrieve_context_with_domain_filter(self, rag_service, sample_documents):
        """Test context retrieval with domain filtering."""
        # Test domain filtering
        tech_docs = await rag_service.retrieve_context("analysis", domain="technology")
        
        assert len(tech_docs) >= 1
        assert all(doc.domain == "technology" for doc in tech_docs)
    
    @pytest.mark.asyncio
    async def test_retrieve_context_empty_query(self, rag_service, sample_documents):
        """Test context retrieval with empty query."""
        documents = await rag_service.retrieve_context("")
        
        # Should still return documents but with low relevance scores
        assert isinstance(documents, list)
    
    @pytest.mark.asyncio
    async def test_index_document_success(self, rag_service):
        """Test successful document indexing."""
        result = await rag_service.index_document(
            title="Test Investment Report",
            content="This is a comprehensive investment report analyzing market trends and providing investment recommendations. The report includes detailed financial analysis, risk assessment, and market outlook for the next quarter.",
            document_type="investment_report",
            domain="finance",
            tags=["investment", "analysis", "Q1"],
            metadata={"author": "Investment Analyst"}
        )
        
        assert isinstance(result, IndexingResult)
        assert result.success is True
        assert result.document_id
        assert result.quality_score > 0.3
        assert result.embedding_generated is True
        assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_index_document_low_quality(self, rag_service):
        """Test indexing of low quality document."""
        result = await rag_service.index_document(
            title="Short",
            content="Too short.",
            document_type="note",
            domain="general"
        )
        
        assert isinstance(result, IndexingResult)
        assert result.success is False
        assert result.quality_score < 0.3
        assert "quality too low" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, rag_service, sample_documents):
        """Test similarity search functionality."""
        # Generate a test embedding
        test_embedding = [0.1] * rag_service.embedding_dimension
        
        # Perform similarity search
        documents = await rag_service.similarity_search(
            query_embedding=test_embedding,
            top_k=5
        )
        
        assert isinstance(documents, list)
        assert len(documents) <= 5
        
        # Check that results are sorted by relevance
        if len(documents) > 1:
            for i in range(len(documents) - 1):
                assert documents[i].relevance_score >= documents[i + 1].relevance_score
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_domain(self, rag_service, sample_documents):
        """Test similarity search with domain filtering."""
        test_embedding = [0.1] * rag_service.embedding_dimension
        
        documents = await rag_service.similarity_search(
            query_embedding=test_embedding,
            domain="technology",
            top_k=3
        )
        
        assert isinstance(documents, list)
        # All returned documents should be from the specified domain
        for doc in documents:
            assert doc.domain == "technology"
    
    @pytest.mark.asyncio
    async def test_track_sources(self, rag_service, sample_documents):
        """Test source tracking functionality."""
        # First retrieve some documents
        documents = await rag_service.retrieve_context("financial analysis")
        
        # Track sources
        sources = await rag_service.track_sources(documents)
        
        assert isinstance(sources, list)
        assert len(sources) == len(documents)
        
        for source in sources:
            assert isinstance(source, Source)
            assert source.source_id
            assert source.source_type == "rag"
            assert source.title
            assert source.reliability_score >= 0.0
            assert isinstance(source.access_date, datetime)
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_with_context(self, rag_service, sample_documents):
        """Test prompt enhancement with context."""
        base_prompt = "Analyze the technology sector performance."
        
        # Retrieve context
        documents = await rag_service.retrieve_context("technology financial")
        
        # Enhance prompt
        enhanced_prompt = await rag_service.enhance_prompt(base_prompt, documents)
        
        assert base_prompt in enhanced_prompt
        assert "相关背景信息" in enhanced_prompt
        assert len(enhanced_prompt) > len(base_prompt)
        
        # Test with empty context
        empty_enhanced = await rag_service.enhance_prompt(base_prompt, [])
        assert empty_enhanced == base_prompt
    
    @pytest.mark.asyncio
    async def test_get_document_by_id(self, rag_service, sample_documents):
        """Test retrieving document by ID."""
        # Get a document ID from sample documents
        doc_id = str(sample_documents[0].document_id)
        
        # Retrieve document
        document = await rag_service.get_document_by_id(doc_id)
        
        assert document is not None
        assert isinstance(document, Document)
        assert document.document_id == doc_id
        assert document.title == sample_documents[0].title
        
        # Test with non-existent ID
        non_existent = await rag_service.get_document_by_id(str(uuid4()))
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_update_document_quality(self, rag_service, sample_documents):
        """Test updating document quality score."""
        doc_id = str(sample_documents[0].document_id)
        new_quality = 0.9
        
        # Update quality
        success = await rag_service.update_document_quality(doc_id, new_quality)
        assert success is True
        
        # Verify update
        document = await rag_service.get_document_by_id(doc_id)
        assert document.quality_score == new_quality
        
        # Test with non-existent document
        success = await rag_service.update_document_quality(str(uuid4()), 0.5)
        assert success is False
    
    def test_calculate_relevance(self, rag_service):
        """Test relevance calculation."""
        # Test exact match
        relevance = asyncio.run(rag_service._calculate_relevance(
            "financial analysis", 
            "This is a financial analysis report"
        ))
        assert relevance > 0.0
        
        # Test no match
        relevance = asyncio.run(rag_service._calculate_relevance(
            "technology", 
            "This is about cooking recipes"
        ))
        assert relevance >= 0.0  # Should be low but not necessarily 0
        
        # Test empty query
        relevance = asyncio.run(rag_service._calculate_relevance("", "content"))
        assert relevance == 0.0
    
    def test_assess_quality(self, rag_service):
        """Test content quality assessment."""
        # High quality content
        high_quality = asyncio.run(rag_service._assess_quality(
            "This is a comprehensive analysis of market trends. "
            "The analysis includes detailed examination of various factors. "
            "In conclusion, the market shows positive indicators."
        ))
        assert high_quality > 0.5
        
        # Low quality content
        low_quality = asyncio.run(rag_service._assess_quality("Short."))
        assert low_quality < 0.3
        
        # Empty content
        empty_quality = asyncio.run(rag_service._assess_quality(""))
        assert empty_quality == 0.1
    
    def test_generate_embedding(self, rag_service):
        """Test embedding generation."""
        content = "This is test content for embedding generation."
        
        embedding = asyncio.run(rag_service._generate_embedding(content))
        
        assert embedding is not None
        assert len(embedding) == rag_service.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)
        
        # Test reproducibility
        embedding2 = asyncio.run(rag_service._generate_embedding(content))
        assert embedding == embedding2
    
    def test_cosine_similarity(self, rag_service):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        # Identical vectors
        similarity = asyncio.run(rag_service._cosine_similarity(vec1, vec2))
        assert abs(similarity - 1.0) < 1e-6
        
        # Orthogonal vectors
        similarity = asyncio.run(rag_service._cosine_similarity(vec1, vec3))
        assert abs(similarity - 0.0) < 1e-6
        
        # Different length vectors
        similarity = asyncio.run(rag_service._cosine_similarity([1.0], [1.0, 0.0]))
        assert similarity == 0.0
        
        # Zero vectors
        similarity = asyncio.run(rag_service._cosine_similarity([0.0], [0.0]))
        assert similarity == 0.0


class TestRAGServiceIntegration:
    """Integration tests for RAG service."""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAG service instance."""
        return RAGService()
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, rag_service):
        """Test complete RAG workflow: index -> retrieve -> track."""
        # Step 1: Index a document
        index_result = await rag_service.index_document(
            title="Investment Strategy Report 2024",
            content="This comprehensive investment strategy report provides detailed analysis of market opportunities for 2024. The report covers equity markets, bond markets, and alternative investments. Key recommendations include diversification strategies and risk management approaches.",
            document_type="strategy_report",
            domain="investment",
            tags=["strategy", "2024", "investment", "analysis"],
            metadata={"author": "Strategy Team", "version": "1.0"}
        )
        
        assert index_result.success is True
        
        # Step 2: Retrieve the document
        documents = await rag_service.retrieve_context("investment strategy 2024")
        
        assert len(documents) >= 1
        retrieved_doc = next(
            (doc for doc in documents if "Investment Strategy" in doc.title), 
            None
        )
        assert retrieved_doc is not None
        
        # Step 3: Track sources
        sources = await rag_service.track_sources([retrieved_doc])
        
        assert len(sources) == 1
        assert sources[0].title == retrieved_doc.title
        
        # Step 4: Enhance prompt
        base_prompt = "What are the key investment recommendations for 2024?"
        enhanced_prompt = await rag_service.enhance_prompt(base_prompt, documents)
        
        assert base_prompt in enhanced_prompt
        assert "Investment Strategy" in enhanced_prompt