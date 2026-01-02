"""Citation and source traceability management service."""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse
import hashlib
import json

logger = logging.getLogger(__name__)


class CitationStyle(str, Enum):
    """Citation style enumeration."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    CUSTOM = "custom"


class SourceType(str, Enum):
    """Source type enumeration."""
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    FINANCIAL_REPORT = "financial_report"
    GOVERNMENT_DATA = "government_data"
    COMPANY_FILING = "company_filing"
    MARKET_DATA = "market_data"
    INTERVIEW = "interview"
    SURVEY = "survey"
    DATABASE = "database"
    WEBSITE = "website"
    BOOK = "book"
    OTHER = "other"


class AccessibilityStatus(str, Enum):
    """Source accessibility status."""
    ACCESSIBLE = "accessible"
    RESTRICTED = "restricted"
    BROKEN = "broken"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


@dataclass
class SourceMetadata:
    """Metadata for a source."""
    title: str
    authors: List[str]
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    access_date: Optional[datetime] = None
    language: str = "en"
    
    def __post_init__(self):
        """Initialize access date if URL is provided."""
        if self.url and not self.access_date:
            self.access_date = datetime.utcnow()


@dataclass
class Source:
    """Represents a source with full metadata and traceability."""
    source_id: str
    source_type: SourceType
    metadata: SourceMetadata
    reliability_score: float
    accessibility_status: AccessibilityStatus
    content_hash: Optional[str] = None
    extraction_method: Optional[str] = None
    last_verified: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps and source ID."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        
        if not self.source_id:
            # Generate source ID from metadata
            id_content = f"{self.metadata.title}_{self.metadata.authors}_{self.metadata.publication_date}"
            self.source_id = hashlib.md5(id_content.encode()).hexdigest()[:12]


@dataclass
class Citation:
    """Represents a citation with context and formatting."""
    citation_id: str
    source_id: str
    page_numbers: Optional[str] = None
    quote: Optional[str] = None
    context: Optional[str] = None
    citation_style: CitationStyle = CitationStyle.APA
    formatted_citation: Optional[str] = None
    in_text_citation: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps and citation ID."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if not self.citation_id:
            # Generate citation ID
            id_content = f"{self.source_id}_{self.page_numbers}_{self.quote}"
            self.citation_id = hashlib.md5(id_content.encode()).hexdigest()[:8]


@dataclass
class TraceabilityRecord:
    """Records the traceability chain for a piece of information."""
    record_id: str
    original_source_id: str
    processing_chain: List[Dict[str, Any]]
    final_content: str
    confidence_score: float
    verification_status: str
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps and record ID."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if not self.record_id:
            # Generate record ID
            chain_hash = hashlib.md5(json.dumps(self.processing_chain, sort_keys=True).encode()).hexdigest()[:8]
            self.record_id = f"{self.original_source_id}_{chain_hash}"


class CitationManager:
    """Service for managing citations and source traceability."""
    
    def __init__(self, default_style: CitationStyle = CitationStyle.APA):
        """Initialize citation manager."""
        self.default_style = default_style
        self._sources: Dict[str, Source] = {}
        self._citations: Dict[str, Citation] = {}
        self._traceability_records: Dict[str, TraceabilityRecord] = {}
        self._citation_counters: Dict[str, int] = {}  # For numbering citations
        
    async def register_source(self, source_type: SourceType, metadata: SourceMetadata,
                             reliability_score: float = 0.8,
                             extraction_method: Optional[str] = None) -> str:
        """Register a new source."""
        try:
            # Create source
            source = Source(
                source_id="",  # Will be auto-generated
                source_type=source_type,
                metadata=metadata,
                reliability_score=reliability_score,
                accessibility_status=AccessibilityStatus.UNKNOWN,
                extraction_method=extraction_method
            )
            
            # Verify accessibility if URL is provided
            if metadata.url:
                source.accessibility_status = await self._verify_url_accessibility(metadata.url)
            else:
                source.accessibility_status = AccessibilityStatus.ACCESSIBLE
            
            # Store source
            self._sources[source.source_id] = source
            
            logger.info(f"Registered source: {source.source_id}")
            return source.source_id
            
        except Exception as e:
            logger.error(f"Error registering source: {e}")
            raise
    
    async def create_citation(self, source_id: str, page_numbers: Optional[str] = None,
                             quote: Optional[str] = None, context: Optional[str] = None,
                             citation_style: Optional[CitationStyle] = None) -> str:
        """Create a citation for a source."""
        try:
            if source_id not in self._sources:
                raise ValueError(f"Source {source_id} not found")
            
            style = citation_style or self.default_style
            
            # Create citation
            citation = Citation(
                citation_id="",  # Will be auto-generated
                source_id=source_id,
                page_numbers=page_numbers,
                quote=quote,
                context=context,
                citation_style=style
            )
            
            # Generate formatted citations
            source = self._sources[source_id]
            citation.formatted_citation = self._format_citation(source, citation, style)
            citation.in_text_citation = self._format_in_text_citation(source, citation, style)
            
            # Store citation
            self._citations[citation.citation_id] = citation
            
            # Update citation counter for this source
            if source_id not in self._citation_counters:
                self._citation_counters[source_id] = 0
            self._citation_counters[source_id] += 1
            
            logger.info(f"Created citation: {citation.citation_id}")
            return citation.citation_id
            
        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            raise
    
    async def create_traceability_record(self, original_source_id: str, 
                                        processing_chain: List[Dict[str, Any]],
                                        final_content: str, confidence_score: float) -> str:
        """Create a traceability record for processed information."""
        try:
            if original_source_id not in self._sources:
                raise ValueError(f"Source {original_source_id} not found")
            
            # Verify processing chain
            verification_status = await self._verify_processing_chain(processing_chain)
            
            # Create traceability record
            record = TraceabilityRecord(
                record_id="",  # Will be auto-generated
                original_source_id=original_source_id,
                processing_chain=processing_chain,
                final_content=final_content,
                confidence_score=confidence_score,
                verification_status=verification_status
            )
            
            # Store record
            self._traceability_records[record.record_id] = record
            
            logger.info(f"Created traceability record: {record.record_id}")
            return record.record_id
            
        except Exception as e:
            logger.error(f"Error creating traceability record: {e}")
            raise
    
    def _format_citation(self, source: Source, citation: Citation, style: CitationStyle) -> str:
        """Format a full citation according to the specified style."""
        metadata = source.metadata
        
        if style == CitationStyle.APA:
            return self._format_apa_citation(metadata, citation)
        elif style == CitationStyle.MLA:
            return self._format_mla_citation(metadata, citation)
        elif style == CitationStyle.CHICAGO:
            return self._format_chicago_citation(metadata, citation)
        elif style == CitationStyle.IEEE:
            return self._format_ieee_citation(metadata, citation)
        else:
            return self._format_custom_citation(metadata, citation)
    
    def _format_apa_citation(self, metadata: SourceMetadata, citation: Citation) -> str:
        """Format citation in APA style."""
        parts = []
        
        # Authors
        if metadata.authors:
            if len(metadata.authors) == 1:
                parts.append(f"{metadata.authors[0]}")
            elif len(metadata.authors) <= 7:
                authors_str = ", ".join(metadata.authors[:-1]) + f", & {metadata.authors[-1]}"
                parts.append(authors_str)
            else:
                authors_str = ", ".join(metadata.authors[:6]) + ", ... " + metadata.authors[-1]
                parts.append(authors_str)
        
        # Publication date
        if metadata.publication_date:
            parts.append(f"({metadata.publication_date.year})")
        
        # Title
        if metadata.title:
            parts.append(f"{metadata.title}")
        
        # Publisher
        if metadata.publisher:
            parts.append(f"{metadata.publisher}")
        
        # URL and access date
        if metadata.url:
            url_part = f"Retrieved from {metadata.url}"
            if metadata.access_date:
                url_part = f"Retrieved {metadata.access_date.strftime('%B %d, %Y')}, from {metadata.url}"
            parts.append(url_part)
        
        return ". ".join(parts) + "."
    
    def _format_mla_citation(self, metadata: SourceMetadata, citation: Citation) -> str:
        """Format citation in MLA style."""
        parts = []
        
        # Authors
        if metadata.authors:
            if len(metadata.authors) == 1:
                parts.append(f"{metadata.authors[0]}")
            else:
                parts.append(f"{metadata.authors[0]}, et al")
        
        # Title
        if metadata.title:
            parts.append(f'"{metadata.title}"')
        
        # Publisher
        if metadata.publisher:
            parts.append(f"{metadata.publisher}")
        
        # Publication date
        if metadata.publication_date:
            parts.append(f"{metadata.publication_date.strftime('%d %b %Y')}")
        
        # URL
        if metadata.url:
            parts.append(f"{metadata.url}")
        
        return ", ".join(parts) + "."
    
    def _format_chicago_citation(self, metadata: SourceMetadata, citation: Citation) -> str:
        """Format citation in Chicago style."""
        parts = []
        
        # Authors
        if metadata.authors:
            parts.append(f"{metadata.authors[0]}")
        
        # Title
        if metadata.title:
            parts.append(f'"{metadata.title}"')
        
        # Publisher and date
        if metadata.publisher and metadata.publication_date:
            parts.append(f"{metadata.publisher}, {metadata.publication_date.year}")
        
        # URL
        if metadata.url:
            parts.append(f"{metadata.url}")
        
        return ". ".join(parts) + "."
    
    def _format_ieee_citation(self, metadata: SourceMetadata, citation: Citation) -> str:
        """Format citation in IEEE style."""
        parts = []
        
        # Authors
        if metadata.authors:
            if len(metadata.authors) == 1:
                parts.append(f"{metadata.authors[0]}")
            else:
                parts.append(f"{metadata.authors[0]} et al.")
        
        # Title
        if metadata.title:
            parts.append(f'"{metadata.title}"')
        
        # Publication info
        if metadata.publisher:
            parts.append(f"{metadata.publisher}")
        
        if metadata.publication_date:
            parts.append(f"{metadata.publication_date.year}")
        
        # URL
        if metadata.url:
            parts.append(f"[Online]. Available: {metadata.url}")
        
        return ", ".join(parts) + "."
    
    def _format_custom_citation(self, metadata: SourceMetadata, citation: Citation) -> str:
        """Format citation in custom style."""
        # Simple custom format
        parts = []
        
        if metadata.authors:
            parts.append(f"Authors: {', '.join(metadata.authors)}")
        
        if metadata.title:
            parts.append(f"Title: {metadata.title}")
        
        if metadata.publication_date:
            parts.append(f"Date: {metadata.publication_date.strftime('%Y-%m-%d')}")
        
        if metadata.url:
            parts.append(f"URL: {metadata.url}")
        
        return " | ".join(parts)
    
    def _format_in_text_citation(self, source: Source, citation: Citation, style: CitationStyle) -> str:
        """Format in-text citation."""
        metadata = source.metadata
        
        if style == CitationStyle.APA:
            if metadata.authors and metadata.publication_date:
                author = metadata.authors[0].split()[-1] if metadata.authors else "Unknown"
                year = metadata.publication_date.year
                page_info = f", p. {citation.page_numbers}" if citation.page_numbers else ""
                return f"({author}, {year}{page_info})"
            
        elif style == CitationStyle.MLA:
            if metadata.authors:
                author = metadata.authors[0].split()[-1]
                page_info = f" {citation.page_numbers}" if citation.page_numbers else ""
                return f"({author}{page_info})"
        
        elif style == CitationStyle.IEEE:
            # Use citation counter as reference number
            ref_num = self._citation_counters.get(source.source_id, 1)
            return f"[{ref_num}]"
        
        # Default format
        return f"[{source.source_id[:8]}]"
    
    async def _verify_url_accessibility(self, url: str) -> AccessibilityStatus:
        """Verify if a URL is accessible."""
        try:
            # Parse URL to check if it's valid
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return AccessibilityStatus.BROKEN
            
            # In a real implementation, this would make an HTTP request
            # For now, we'll assume URLs are accessible
            return AccessibilityStatus.ACCESSIBLE
            
        except Exception:
            return AccessibilityStatus.BROKEN
    
    async def _verify_processing_chain(self, processing_chain: List[Dict[str, Any]]) -> str:
        """Verify the integrity of a processing chain."""
        try:
            # Check that each step in the chain has required fields
            required_fields = ["step", "method", "timestamp"]
            
            for i, step in enumerate(processing_chain):
                for field in required_fields:
                    if field not in step:
                        return f"invalid_step_{i}_missing_{field}"
                
                # Verify timestamp format
                if not isinstance(step.get("timestamp"), (str, datetime)):
                    return f"invalid_timestamp_step_{i}"
            
            return "verified"
            
        except Exception as e:
            return f"verification_error_{str(e)}"
    
    async def verify_all_sources(self) -> Dict[str, AccessibilityStatus]:
        """Verify accessibility of all sources with URLs."""
        results = {}
        
        for source_id, source in self._sources.items():
            if source.metadata.url:
                status = await self._verify_url_accessibility(source.metadata.url)
                source.accessibility_status = status
                source.last_verified = datetime.utcnow()
                results[source_id] = status
            else:
                results[source_id] = AccessibilityStatus.ACCESSIBLE
        
        return results
    
    def generate_bibliography(self, citation_ids: Optional[List[str]] = None,
                             style: Optional[CitationStyle] = None) -> str:
        """Generate a bibliography from citations."""
        try:
            style = style or self.default_style
            
            # Get citations to include
            if citation_ids:
                citations = [self._citations[cid] for cid in citation_ids if cid in self._citations]
            else:
                citations = list(self._citations.values())
            
            # Group by source and get unique sources
            unique_sources = {}
            for citation in citations:
                if citation.source_id not in unique_sources:
                    unique_sources[citation.source_id] = self._sources[citation.source_id]
            
            # Sort sources by author or title
            sorted_sources = sorted(
                unique_sources.values(),
                key=lambda s: (s.metadata.authors[0] if s.metadata.authors else s.metadata.title)
            )
            
            # Generate bibliography entries
            bibliography_entries = []
            for source in sorted_sources:
                # Create a dummy citation for formatting
                dummy_citation = Citation(
                    citation_id="temp",
                    source_id=source.source_id,
                    citation_style=style
                )
                
                formatted = self._format_citation(source, dummy_citation, style)
                bibliography_entries.append(formatted)
            
            # Combine into bibliography
            if style == CitationStyle.IEEE:
                # Number the references
                numbered_entries = [f"[{i+1}] {entry}" for i, entry in enumerate(bibliography_entries)]
                return "\n".join(numbered_entries)
            else:
                return "\n\n".join(bibliography_entries)
            
        except Exception as e:
            logger.error(f"Error generating bibliography: {e}")
            return ""
    
    def get_source_usage_report(self) -> Dict[str, Any]:
        """Generate a report on source usage."""
        total_sources = len(self._sources)
        total_citations = len(self._citations)
        
        # Count by source type
        sources_by_type = {}
        for source in self._sources.values():
            source_type = source.source_type.value
            sources_by_type[source_type] = sources_by_type.get(source_type, 0) + 1
        
        # Count by accessibility status
        sources_by_accessibility = {}
        for source in self._sources.values():
            status = source.accessibility_status.value
            sources_by_accessibility[status] = sources_by_accessibility.get(status, 0) + 1
        
        # Most cited sources
        most_cited = sorted(
            self._citation_counters.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Reliability distribution
        reliability_scores = [source.reliability_score for source in self._sources.values()]
        avg_reliability = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0
        
        return {
            "total_sources": total_sources,
            "total_citations": total_citations,
            "sources_by_type": sources_by_type,
            "sources_by_accessibility": sources_by_accessibility,
            "most_cited_sources": most_cited,
            "average_reliability_score": avg_reliability,
            "traceability_records": len(self._traceability_records),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def get_traceability_chain(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get the full traceability chain for a record."""
        if record_id not in self._traceability_records:
            return None
        
        record = self._traceability_records[record_id]
        source = self._sources.get(record.original_source_id)
        
        return {
            "record_id": record.record_id,
            "original_source": asdict(source) if source else None,
            "processing_chain": record.processing_chain,
            "final_content": record.final_content,
            "confidence_score": record.confidence_score,
            "verification_status": record.verification_status,
            "created_at": record.created_at.isoformat()
        }
    
    def export_citations(self, format_type: str = "json") -> str:
        """Export all citations in the specified format."""
        if format_type == "json":
            export_data = {
                "sources": {sid: asdict(source) for sid, source in self._sources.items()},
                "citations": {cid: asdict(citation) for cid, citation in self._citations.items()},
                "traceability_records": {rid: asdict(record) for rid, record in self._traceability_records.items()},
                "exported_at": datetime.utcnow().isoformat()
            }
            return json.dumps(export_data, indent=2, default=str)
        
        elif format_type == "bibtex":
            # Generate BibTeX format
            bibtex_entries = []
            for source in self._sources.values():
                entry = self._generate_bibtex_entry(source)
                bibtex_entries.append(entry)
            return "\n\n".join(bibtex_entries)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _generate_bibtex_entry(self, source: Source) -> str:
        """Generate a BibTeX entry for a source."""
        metadata = source.metadata
        entry_type = "article"  # Default
        
        # Determine entry type based on source type
        if source.source_type == SourceType.BOOK:
            entry_type = "book"
        elif source.source_type == SourceType.ACADEMIC_PAPER:
            entry_type = "article"
        elif source.source_type == SourceType.WEBSITE:
            entry_type = "misc"
        
        # Build BibTeX entry
        lines = [f"@{entry_type}{{{source.source_id},"]
        
        if metadata.title:
            lines.append(f'  title = "{metadata.title}",')
        
        if metadata.authors:
            authors_str = " and ".join(metadata.authors)
            lines.append(f'  author = "{authors_str}",')
        
        if metadata.publication_date:
            lines.append(f'  year = "{metadata.publication_date.year}",')
        
        if metadata.publisher:
            lines.append(f'  publisher = "{metadata.publisher}",')
        
        if metadata.url:
            lines.append(f'  url = "{metadata.url}",')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def clear_all_data(self):
        """Clear all citation data."""
        self._sources.clear()
        self._citations.clear()
        self._traceability_records.clear()
        self._citation_counters.clear()
        logger.info("All citation data cleared")