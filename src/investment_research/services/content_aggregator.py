"""Content aggregation and coherence management service."""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Content type enumeration."""
    EXECUTIVE_SUMMARY = "executive_summary"
    INDUSTRY_ANALYSIS = "industry_analysis"
    FINANCIAL_ANALYSIS = "financial_analysis"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    CONCLUSION = "conclusion"


class CoherenceLevel(str, Enum):
    """Content coherence level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class ContentFragment:
    """Represents a fragment of content from an agent."""
    fragment_id: str
    agent_type: str
    content_type: ContentType
    content: str
    data_references: List[str]
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize fragment ID if not provided."""
        if not self.fragment_id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.fragment_id = f"{self.agent_type}_{self.content_type}_{content_hash}"


@dataclass
class CoherenceIssue:
    """Represents a coherence issue found in content."""
    issue_id: str
    issue_type: str
    severity: CoherenceLevel
    description: str
    affected_fragments: List[str]
    suggested_resolution: Optional[str] = None
    resolved: bool = False


@dataclass
class DataReference:
    """Represents a data reference with consistency tracking."""
    reference_id: str
    source: str
    data_point: str
    value: Any
    unit: Optional[str] = None
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Initialize reference ID if not provided."""
        if not self.reference_id:
            ref_content = f"{self.source}_{self.data_point}_{str(self.value)}"
            self.reference_id = hashlib.md5(ref_content.encode()).hexdigest()[:12]


class ContentAggregator:
    """Service for aggregating agent results and ensuring content coherence."""
    
    def __init__(self):
        """Initialize content aggregator."""
        self._fragments: Dict[str, ContentFragment] = {}
        self._data_references: Dict[str, DataReference] = {}
        self._coherence_issues: List[CoherenceIssue] = []
        self._aggregated_content: Dict[ContentType, str] = {}
        
    async def add_agent_content(self, agent_type: str, content_type: ContentType, 
                               content: str, data_references: List[Dict[str, Any]], 
                               confidence_score: float = 1.0, 
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add content from an agent."""
        try:
            # Create content fragment
            fragment = ContentFragment(
                fragment_id="",  # Will be auto-generated
                agent_type=agent_type,
                content_type=content_type,
                content=content,
                data_references=[],
                confidence_score=confidence_score,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Process data references
            reference_ids = []
            for ref_data in data_references:
                data_ref = DataReference(
                    reference_id="",  # Will be auto-generated
                    source=ref_data.get("source", "unknown"),
                    data_point=ref_data.get("data_point", ""),
                    value=ref_data.get("value"),
                    unit=ref_data.get("unit"),
                    timestamp=ref_data.get("timestamp"),
                    confidence=ref_data.get("confidence", 1.0)
                )
                
                self._data_references[data_ref.reference_id] = data_ref
                reference_ids.append(data_ref.reference_id)
            
            fragment.data_references = reference_ids
            self._fragments[fragment.fragment_id] = fragment
            
            logger.info(f"Added content fragment: {fragment.fragment_id}")
            return fragment.fragment_id
            
        except Exception as e:
            logger.error(f"Error adding agent content: {e}")
            raise
    
    async def aggregate_content_by_type(self, content_type: ContentType) -> str:
        """Aggregate all content fragments of a specific type."""
        try:
            # Get all fragments of the specified type
            fragments = [
                fragment for fragment in self._fragments.values()
                if fragment.content_type == content_type
            ]
            
            if not fragments:
                return ""
            
            # Sort fragments by confidence score (highest first)
            fragments.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Aggregate content with coherence checks
            aggregated_parts = []
            
            for fragment in fragments:
                # Check for coherence issues before adding
                issues = await self._check_fragment_coherence(fragment, aggregated_parts)
                
                if not any(issue.severity == CoherenceLevel.CRITICAL for issue in issues):
                    # Add fragment with agent attribution
                    agent_attribution = f"[{fragment.agent_type.upper()}]"
                    content_with_attribution = f"{agent_attribution} {fragment.content}"
                    aggregated_parts.append(content_with_attribution)
                    
                    # Store any non-critical issues
                    self._coherence_issues.extend(issues)
                else:
                    logger.warning(f"Skipping fragment {fragment.fragment_id} due to critical coherence issues")
                    self._coherence_issues.extend(issues)
            
            # Combine parts with proper formatting
            aggregated_content = self._format_aggregated_content(content_type, aggregated_parts)
            self._aggregated_content[content_type] = aggregated_content
            
            return aggregated_content
            
        except Exception as e:
            logger.error(f"Error aggregating content: {e}")
            raise
    
    async def _check_fragment_coherence(self, fragment: ContentFragment, 
                                       existing_parts: List[str]) -> List[CoherenceIssue]:
        """Check coherence of a fragment against existing content."""
        issues = []
        
        try:
            # Check for data consistency
            data_issues = await self._check_data_consistency(fragment)
            issues.extend(data_issues)
            
            # Check for content contradictions
            contradiction_issues = await self._check_contradictions(fragment, existing_parts)
            issues.extend(contradiction_issues)
            
            # Check for redundancy
            redundancy_issues = await self._check_redundancy(fragment, existing_parts)
            issues.extend(redundancy_issues)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking fragment coherence: {e}")
            return []
    
    async def _check_data_consistency(self, fragment: ContentFragment) -> List[CoherenceIssue]:
        """Check data consistency within and across fragments."""
        issues = []
        
        try:
            # Check references in this fragment
            for ref_id in fragment.data_references:
                if ref_id not in self._data_references:
                    continue
                
                current_ref = self._data_references[ref_id]
                
                # Find similar references from other fragments
                similar_refs = [
                    ref for ref in self._data_references.values()
                    if (ref.data_point == current_ref.data_point and 
                        ref.reference_id != current_ref.reference_id)
                ]
                
                # Check for inconsistent values
                for similar_ref in similar_refs:
                    if self._values_inconsistent(current_ref.value, similar_ref.value):
                        issue = CoherenceIssue(
                            issue_id=f"data_inconsistency_{current_ref.reference_id}_{similar_ref.reference_id}",
                            issue_type="data_inconsistency",
                            severity=CoherenceLevel.HIGH,
                            description=f"Inconsistent values for {current_ref.data_point}: "
                                      f"{current_ref.value} vs {similar_ref.value}",
                            affected_fragments=[fragment.fragment_id],
                            suggested_resolution=f"Verify data sources and use most reliable value"
                        )
                        issues.append(issue)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking data consistency: {e}")
            return []
    
    async def _check_contradictions(self, fragment: ContentFragment, 
                                   existing_parts: List[str]) -> List[CoherenceIssue]:
        """Check for logical contradictions in content."""
        issues = []
        
        try:
            # Simple contradiction detection using keyword analysis
            fragment_content = fragment.content.lower()
            
            # Define contradiction patterns
            positive_indicators = ["increase", "growth", "positive", "strong", "high", "good"]
            negative_indicators = ["decrease", "decline", "negative", "weak", "low", "poor"]
            
            fragment_sentiment = self._analyze_sentiment(fragment_content, positive_indicators, negative_indicators)
            
            for existing_part in existing_parts:
                existing_sentiment = self._analyze_sentiment(existing_part.lower(), positive_indicators, negative_indicators)
                
                # Check for contradictory sentiments on similar topics
                if self._sentiments_contradictory(fragment_sentiment, existing_sentiment):
                    issue = CoherenceIssue(
                        issue_id=f"contradiction_{fragment.fragment_id}_{len(issues)}",
                        issue_type="content_contradiction",
                        severity=CoherenceLevel.MEDIUM,
                        description=f"Potential contradiction detected between {fragment.agent_type} analysis and existing content",
                        affected_fragments=[fragment.fragment_id],
                        suggested_resolution="Review and reconcile conflicting analyses"
                    )
                    issues.append(issue)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking contradictions: {e}")
            return []
    
    async def _check_redundancy(self, fragment: ContentFragment, 
                               existing_parts: List[str]) -> List[CoherenceIssue]:
        """Check for redundant content."""
        issues = []
        
        try:
            fragment_content = fragment.content.lower()
            
            for i, existing_part in enumerate(existing_parts):
                similarity = self._calculate_content_similarity(fragment_content, existing_part.lower())
                
                if similarity > 0.8:  # High similarity threshold
                    issue = CoherenceIssue(
                        issue_id=f"redundancy_{fragment.fragment_id}_{i}",
                        issue_type="content_redundancy",
                        severity=CoherenceLevel.LOW,
                        description=f"High content similarity ({similarity:.2f}) detected",
                        affected_fragments=[fragment.fragment_id],
                        suggested_resolution="Consider merging or removing redundant content"
                    )
                    issues.append(issue)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking redundancy: {e}")
            return []
    
    def _values_inconsistent(self, value1: Any, value2: Any) -> bool:
        """Check if two values are inconsistent."""
        try:
            # Handle numeric values
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                # Consider values inconsistent if they differ by more than 10%
                if value1 == 0 and value2 == 0:
                    return False
                if value1 == 0 or value2 == 0:
                    return abs(value1 - value2) > 0.1
                
                relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
                return relative_diff > 0.1
            
            # Handle string values
            if isinstance(value1, str) and isinstance(value2, str):
                return value1.lower().strip() != value2.lower().strip()
            
            # Handle other types
            return value1 != value2
            
        except Exception:
            return True  # Assume inconsistent if comparison fails
    
    def _analyze_sentiment(self, content: str, positive_indicators: List[str], 
                          negative_indicators: List[str]) -> str:
        """Analyze sentiment of content."""
        positive_count = sum(1 for indicator in positive_indicators if indicator in content)
        negative_count = sum(1 for indicator in negative_indicators if indicator in content)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _sentiments_contradictory(self, sentiment1: str, sentiment2: str) -> bool:
        """Check if two sentiments are contradictory."""
        contradictory_pairs = [
            ("positive", "negative"),
            ("negative", "positive")
        ]
        return (sentiment1, sentiment2) in contradictory_pairs
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        try:
            # Simple word-based similarity
            words1 = set(re.findall(r'\w+', content1.lower()))
            words2 = set(re.findall(r'\w+', content2.lower()))
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0
    
    def _format_aggregated_content(self, content_type: ContentType, parts: List[str]) -> str:
        """Format aggregated content with proper structure."""
        if not parts:
            return ""
        
        # Add section header
        section_headers = {
            ContentType.EXECUTIVE_SUMMARY: "## 执行摘要",
            ContentType.INDUSTRY_ANALYSIS: "## 行业分析",
            ContentType.FINANCIAL_ANALYSIS: "## 财务分析",
            ContentType.MARKET_ANALYSIS: "## 市场分析",
            ContentType.RISK_ASSESSMENT: "## 风险评估",
            ContentType.CONCLUSION: "## 结论与建议"
        }
        
        header = section_headers.get(content_type, f"## {content_type.value.title()}")
        
        # Combine parts with proper spacing
        formatted_content = f"{header}\n\n"
        
        for i, part in enumerate(parts):
            if i > 0:
                formatted_content += "\n\n"
            formatted_content += part
        
        return formatted_content
    
    async def resolve_coherence_issues(self, auto_resolve: bool = False) -> List[CoherenceIssue]:
        """Resolve coherence issues in the aggregated content."""
        resolved_issues = []
        
        try:
            for issue in self._coherence_issues:
                if issue.resolved:
                    continue
                
                if auto_resolve and issue.severity in [CoherenceLevel.LOW, CoherenceLevel.MEDIUM]:
                    # Attempt automatic resolution
                    success = await self._auto_resolve_issue(issue)
                    if success:
                        issue.resolved = True
                        resolved_issues.append(issue)
                        logger.info(f"Auto-resolved issue: {issue.issue_id}")
            
            return resolved_issues
            
        except Exception as e:
            logger.error(f"Error resolving coherence issues: {e}")
            return []
    
    async def _auto_resolve_issue(self, issue: CoherenceIssue) -> bool:
        """Attempt to automatically resolve a coherence issue."""
        try:
            if issue.issue_type == "content_redundancy":
                # Remove redundant fragments
                for fragment_id in issue.affected_fragments:
                    if fragment_id in self._fragments:
                        fragment = self._fragments[fragment_id]
                        if fragment.confidence_score < 0.8:  # Only remove low-confidence redundant content
                            del self._fragments[fragment_id]
                            return True
            
            elif issue.issue_type == "data_inconsistency":
                # Use the most confident data reference
                # This would require more sophisticated logic
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error auto-resolving issue {issue.issue_id}: {e}")
            return False
    
    def get_coherence_report(self) -> Dict[str, Any]:
        """Generate a coherence report."""
        total_issues = len(self._coherence_issues)
        resolved_issues = len([issue for issue in self._coherence_issues if issue.resolved])
        
        issues_by_severity = {}
        for severity in CoherenceLevel:
            issues_by_severity[severity.value] = len([
                issue for issue in self._coherence_issues 
                if issue.severity == severity and not issue.resolved
            ])
        
        issues_by_type = {}
        for issue in self._coherence_issues:
            if not issue.resolved:
                if issue.issue_type not in issues_by_type:
                    issues_by_type[issue.issue_type] = 0
                issues_by_type[issue.issue_type] += 1
        
        return {
            "total_fragments": len(self._fragments),
            "total_data_references": len(self._data_references),
            "total_issues": total_issues,
            "resolved_issues": resolved_issues,
            "pending_issues": total_issues - resolved_issues,
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
            "aggregated_sections": list(self._aggregated_content.keys()),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def get_data_consistency_report(self) -> Dict[str, Any]:
        """Generate a data consistency report."""
        # Group references by data point
        references_by_point = {}
        for ref in self._data_references.values():
            if ref.data_point not in references_by_point:
                references_by_point[ref.data_point] = []
            references_by_point[ref.data_point].append(ref)
        
        # Check consistency for each data point
        consistency_report = {}
        for data_point, refs in references_by_point.items():
            if len(refs) > 1:
                values = [ref.value for ref in refs]
                sources = [ref.source for ref in refs]
                
                # Check if all values are consistent
                consistent = all(not self._values_inconsistent(values[0], val) for val in values[1:])
                
                consistency_report[data_point] = {
                    "consistent": consistent,
                    "reference_count": len(refs),
                    "unique_values": len(set(str(val) for val in values)),
                    "sources": list(set(sources)),
                    "values": values
                }
        
        return {
            "total_data_points": len(references_by_point),
            "multi_reference_points": len(consistency_report),
            "consistent_points": len([
                point for point, info in consistency_report.items() 
                if info["consistent"]
            ]),
            "inconsistent_points": len([
                point for point, info in consistency_report.items() 
                if not info["consistent"]
            ]),
            "details": consistency_report,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def export_aggregated_content(self) -> Dict[str, str]:
        """Export all aggregated content."""
        return dict(self._aggregated_content)
    
    def clear_content(self):
        """Clear all content and reset the aggregator."""
        self._fragments.clear()
        self._data_references.clear()
        self._coherence_issues.clear()
        self._aggregated_content.clear()
        logger.info("Content aggregator cleared")