"""RAG (Retrieval-Augmented Generation) service implementation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..core.models import KBDocument, Source as SourceModel
from ..core.database import get_session_factory


@dataclass
class Document:
    """Document retrieved from knowledge base."""
    
    document_id: str
    title: str
    content: str
    document_type: str
    domain: str
    tags: List[str]
    relevance_score: float
    source_info: Dict[str, Any]


@dataclass
class Source:
    """Source information for retrieved documents."""
    
    source_id: str
    source_type: str
    url: Optional[str]
    title: str
    reliability_score: float
    access_date: datetime


class RAGService:
    """RAG service for knowledge base retrieval and context enhancement."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.embedding_dimension = 1536  # OpenAI embedding dimension
    
    async def retrieve_context(
        self, 
        query: str, 
        domain: str, 
        limit: int = 5,
        min_relevance: float = 0.3
    ) -> List[Document]:
        """Retrieve relevant context from knowledge base.
        
        Args:
            query: Search query
            domain: Domain to search within
            limit: Maximum number of documents to retrieve
            min_relevance: Minimum relevance score threshold
        
        Returns:
            List of relevant documents
        """
        async with get_session_factory()() as session:
            # For now, implement simple text-based search
            # In production, this would use vector similarity search
            stmt = select(KBDocument).where(
                KBDocument.domain == domain,
                func.lower(KBDocument.content).contains(query.lower())
            ).limit(limit)
            
            result = await session.execute(stmt)
            kb_documents = result.scalars().all()
            
            documents = []
            for kb_doc in kb_documents:
                # Calculate relevance score (simplified)
                relevance_score = self._calculate_relevance(query, kb_doc.content)
                
                if relevance_score >= min_relevance:
                    documents.append(Document(
                        document_id=str(kb_doc.document_id),
                        title=kb_doc.title,
                        content=kb_doc.content,
                        document_type=kb_doc.document_type,
                        domain=kb_doc.domain,
                        tags=kb_doc.tags,
                        relevance_score=relevance_score,
                        source_info={
                            "quality_score": kb_doc.quality_score,
                            "last_updated": kb_doc.last_updated.isoformat(),
                            "document_metadata": kb_doc.document_metadata
                        }
                    ))
            
            # Sort by relevance score
            documents.sort(key=lambda x: x.relevance_score, reverse=True)
            return documents
    
    def enhance_prompt(
        self, 
        base_prompt: str, 
        context: List[Document]
    ) -> str:
        """Enhance a prompt with retrieved context.
        
        Args:
            base_prompt: Original prompt
            context: Retrieved context documents
        
        Returns:
            Enhanced prompt with context
        """
        if not context:
            return base_prompt
        
        context_text = "\n\n".join([
            f"**{doc.title}** (Relevance: {doc.relevance_score:.2f})\n{doc.content[:500]}..."
            for doc in context[:3]  # Use top 3 most relevant documents
        ])
        
        enhanced_prompt = f"""
{base_prompt}

## Relevant Context:
{context_text}

Please use the above context to inform your analysis, but also draw upon your general knowledge. 
Cite specific information from the context when relevant.
"""
        
        return enhanced_prompt
    
    async def track_sources(self, documents: List[Document]) -> List[Source]:
        """Track and return source information for documents.
        
        Args:
            documents: List of documents to track sources for
        
        Returns:
            List of source information
        """
        sources = []
        
        for doc in documents:
            # Create source tracking entry
            source = Source(
                source_id=doc.document_id,
                source_type="rag",
                url=None,  # Knowledge base documents don't have URLs
                title=doc.title,
                reliability_score=doc.source_info.get("quality_score", 0.5),
                access_date=datetime.utcnow()
            )
            sources.append(source)
        
        return sources
    
    async def add_document(
        self,
        title: str,
        content: str,
        document_type: str,
        domain: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a new document to the knowledge base.
        
        Args:
            title: Document title
            content: Document content
            document_type: Type of document
            domain: Domain category
            tags: List of tags
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        async with get_session_factory()() as session:
            kb_doc = KBDocument(
                title=title,
                content=content,
                document_type=document_type,
                domain=domain,
                tags=tags or [],
                quality_score=self._assess_quality(content),
                document_metadata=metadata or {}
            )
            
            session.add(kb_doc)
            await session.commit()
            await session.refresh(kb_doc)
            
            return str(kb_doc.document_id)
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content.
        
        This is a simplified implementation. In production, this would use
        vector embeddings and cosine similarity.
        
        Args:
            query: Search query
            content: Document content
        
        Returns:
            Relevance score between 0 and 1
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Simple word overlap calculation
        overlap = len(query_words.intersection(content_words))
        relevance = overlap / len(query_words)
        
        return min(relevance, 1.0)
    
    def _assess_quality(self, content: str) -> float:
        """Assess the quality of document content.
        
        Args:
            content: Document content
        
        Returns:
            Quality score between 0 and 1
        """
        # Simple quality assessment based on content length and structure
        if len(content) < 100:
            return 0.3
        elif len(content) < 500:
            return 0.6
        elif len(content) < 2000:
            return 0.8
        else:
            return 0.9