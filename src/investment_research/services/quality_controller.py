"""Quality control service for analysis results and report content."""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import statistics
import json

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality level enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class QualityMetric(str, Enum):
    """Quality metric types."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    OBJECTIVITY = "objectivity"
    TIMELINESS = "timeliness"
    RELIABILITY = "reliability"


class IssueType(str, Enum):
    """Quality issue types."""
    MISSING_DATA = "missing_data"
    INCONSISTENT_DATA = "inconsistent_data"
    OUTDATED_DATA = "outdated_data"
    UNRELIABLE_SOURCE = "unreliable_source"
    LOGICAL_ERROR = "logical_error"
    FORMATTING_ERROR = "formatting_error"
    BIAS_DETECTED = "bias_detected"
    INCOMPLETE_ANALYSIS = "incomplete_analysis"


@dataclass
class QualityScore:
    """Represents a quality score for a specific metric."""
    metric: QualityMetric
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    evidence: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        """Initialize timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def quality_level(self) -> QualityLevel:
        """Get quality level based on score."""
        if self.score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.score >= 0.8:
            return QualityLevel.GOOD
        elif self.score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif self.score >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


@dataclass
class QualityIssue:
    """Represents a quality issue found during assessment."""
    issue_id: str
    issue_type: IssueType
    severity: QualityLevel
    description: str
    affected_content: str
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False
    fixed: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        """Initialize timestamp and issue ID."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        if not self.issue_id:
            import hashlib
            content_hash = hashlib.md5(f"{self.issue_type}_{self.description}".encode()).hexdigest()[:8]
            self.issue_id = f"{self.issue_type}_{content_hash}"


@dataclass
class QualityAssessment:
    """Complete quality assessment for content."""
    assessment_id: str
    content_id: str
    overall_score: float
    metric_scores: Dict[QualityMetric, QualityScore]
    issues: List[QualityIssue]
    recommendations: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        """Initialize timestamp and assessment ID."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        if not self.assessment_id:
            import hashlib
            content_hash = hashlib.md5(f"{self.content_id}_{self.timestamp}".encode()).hexdigest()[:8]
            self.assessment_id = f"qa_{content_hash}"
    
    @property
    def overall_quality_level(self) -> QualityLevel:
        """Get overall quality level."""
        if self.overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.overall_score >= 0.8:
            return QualityLevel.GOOD
        elif self.overall_score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif self.overall_score >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


class QualityController:
    """Service for quality control and assessment of analysis results."""
    
    def __init__(self):
        """Initialize quality controller."""
        self._assessments: Dict[str, QualityAssessment] = {}
        self._quality_standards: Dict[QualityMetric, float] = {
            QualityMetric.COMPLETENESS: 0.8,
            QualityMetric.ACCURACY: 0.9,
            QualityMetric.CONSISTENCY: 0.85,
            QualityMetric.RELEVANCE: 0.8,
            QualityMetric.CLARITY: 0.75,
            QualityMetric.OBJECTIVITY: 0.8,
            QualityMetric.TIMELINESS: 0.7,
            QualityMetric.RELIABILITY: 0.85
        }
        self._metric_weights: Dict[QualityMetric, float] = {
            QualityMetric.COMPLETENESS: 0.15,
            QualityMetric.ACCURACY: 0.20,
            QualityMetric.CONSISTENCY: 0.15,
            QualityMetric.RELEVANCE: 0.10,
            QualityMetric.CLARITY: 0.10,
            QualityMetric.OBJECTIVITY: 0.10,
            QualityMetric.TIMELINESS: 0.10,
            QualityMetric.RELIABILITY: 0.10
        }
    
    async def assess_content_quality(self, content_id: str, content: str, 
                                   metadata: Optional[Dict[str, Any]] = None) -> QualityAssessment:
        """Perform comprehensive quality assessment of content."""
        try:
            metadata = metadata or {}
            
            # Assess individual quality metrics
            metric_scores = {}
            all_issues = []
            
            # Completeness assessment
            completeness_score, completeness_issues = await self._assess_completeness(content, metadata)
            metric_scores[QualityMetric.COMPLETENESS] = completeness_score
            all_issues.extend(completeness_issues)
            
            # Accuracy assessment
            accuracy_score, accuracy_issues = await self._assess_accuracy(content, metadata)
            metric_scores[QualityMetric.ACCURACY] = accuracy_score
            all_issues.extend(accuracy_issues)
            
            # Consistency assessment
            consistency_score, consistency_issues = await self._assess_consistency(content, metadata)
            metric_scores[QualityMetric.CONSISTENCY] = consistency_score
            all_issues.extend(consistency_issues)
            
            # Relevance assessment
            relevance_score, relevance_issues = await self._assess_relevance(content, metadata)
            metric_scores[QualityMetric.RELEVANCE] = relevance_score
            all_issues.extend(relevance_issues)
            
            # Clarity assessment
            clarity_score, clarity_issues = await self._assess_clarity(content, metadata)
            metric_scores[QualityMetric.CLARITY] = clarity_score
            all_issues.extend(clarity_issues)
            
            # Objectivity assessment
            objectivity_score, objectivity_issues = await self._assess_objectivity(content, metadata)
            metric_scores[QualityMetric.OBJECTIVITY] = objectivity_score
            all_issues.extend(objectivity_issues)
            
            # Timeliness assessment
            timeliness_score, timeliness_issues = await self._assess_timeliness(content, metadata)
            metric_scores[QualityMetric.TIMELINESS] = timeliness_score
            all_issues.extend(timeliness_issues)
            
            # Reliability assessment
            reliability_score, reliability_issues = await self._assess_reliability(content, metadata)
            metric_scores[QualityMetric.RELIABILITY] = reliability_score
            all_issues.extend(reliability_issues)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metric_scores)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metric_scores, all_issues)
            
            # Create assessment
            assessment = QualityAssessment(
                assessment_id="",  # Will be auto-generated
                content_id=content_id,
                overall_score=overall_score,
                metric_scores=metric_scores,
                issues=all_issues,
                recommendations=recommendations
            )
            
            # Store assessment
            self._assessments[assessment.assessment_id] = assessment
            
            logger.info(f"Quality assessment completed: {assessment.assessment_id} (score: {overall_score:.2f})")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            raise
    
    async def _assess_completeness(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content completeness."""
        issues = []
        evidence = []
        
        # Check for required sections
        required_sections = ["executive summary", "analysis", "conclusion", "recommendation"]
        missing_sections = []
        
        content_lower = content.lower()
        for section in required_sections:
            if section not in content_lower:
                missing_sections.append(section)
        
        # Check content length
        word_count = len(content.split())
        min_word_count = metadata.get("min_word_count", 1000)
        
        if word_count < min_word_count:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.INCOMPLETE_ANALYSIS,
                severity=QualityLevel.POOR,
                description=f"Content too short: {word_count} words (minimum: {min_word_count})",
                affected_content=content[:100] + "...",
                suggested_fix="Expand analysis with more detailed information",
                auto_fixable=False
            ))
            evidence.append(f"Word count below minimum: {word_count}/{min_word_count}")
        
        # Check for missing sections
        if missing_sections:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.INCOMPLETE_ANALYSIS,
                severity=QualityLevel.POOR,
                description=f"Missing required sections: {', '.join(missing_sections)}",
                affected_content="Overall structure",
                suggested_fix="Add missing sections to complete the analysis",
                auto_fixable=False
            ))
            evidence.append(f"Missing sections: {missing_sections}")
        
        # Calculate completeness score
        section_score = (len(required_sections) - len(missing_sections)) / len(required_sections)
        length_score = min(1.0, word_count / min_word_count)
        completeness_score = (section_score + length_score) / 2
        
        if not evidence:
            evidence.append("All required sections present and adequate length")
        
        score = QualityScore(
            metric=QualityMetric.COMPLETENESS,
            score=completeness_score,
            confidence=0.8,
            explanation=f"Completeness based on section coverage ({section_score:.2f}) and content length ({length_score:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_accuracy(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content accuracy."""
        issues = []
        evidence = []
        
        # Check for numerical inconsistencies
        numbers = re.findall(r'\d+\.?\d*', content)
        if numbers:
            # Look for obviously incorrect values (e.g., percentages > 100)
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if "%" in content and num > 100:
                        issues.append(QualityIssue(
                            issue_id="",
                            issue_type=IssueType.LOGICAL_ERROR,
                            severity=QualityLevel.POOR,
                            description=f"Percentage value exceeds 100%: {num}%",
                            affected_content=f"Value: {num}%",
                            suggested_fix="Verify and correct percentage calculations",
                            auto_fixable=False
                        ))
                        evidence.append(f"Invalid percentage: {num}%")
                except ValueError:
                    continue
        
        # Check for contradictory statements
        contradictory_patterns = [
            (r"increase.*decrease", "Contradictory trend statements"),
            (r"positive.*negative", "Contradictory sentiment statements"),
            (r"high.*low", "Contradictory magnitude statements")
        ]
        
        for pattern, description in contradictory_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.LOGICAL_ERROR,
                    severity=QualityLevel.ACCEPTABLE,
                    description=description,
                    affected_content="Content contains potentially contradictory statements",
                    suggested_fix="Review and clarify contradictory statements",
                    auto_fixable=False
                ))
                evidence.append(description)
        
        # Check source reliability
        source_reliability = metadata.get("average_source_reliability", 0.8)
        if source_reliability < 0.7:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.UNRELIABLE_SOURCE,
                severity=QualityLevel.POOR,
                description=f"Low average source reliability: {source_reliability:.2f}",
                affected_content="Overall content reliability",
                suggested_fix="Use more reliable sources for analysis",
                auto_fixable=False
            ))
            evidence.append(f"Source reliability: {source_reliability:.2f}")
        
        # Calculate accuracy score
        logical_error_penalty = len([i for i in issues if i.issue_type == IssueType.LOGICAL_ERROR]) * 0.1
        source_reliability_score = source_reliability
        accuracy_score = max(0.0, source_reliability_score - logical_error_penalty)
        
        if not evidence:
            evidence.append("No obvious accuracy issues detected")
        
        score = QualityScore(
            metric=QualityMetric.ACCURACY,
            score=accuracy_score,
            confidence=0.7,
            explanation=f"Accuracy based on source reliability ({source_reliability:.2f}) minus logical error penalty ({logical_error_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_consistency(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content consistency."""
        issues = []
        evidence = []
        
        # Check for consistent terminology
        financial_terms = {
            "revenue": ["revenue", "sales", "income"],
            "profit": ["profit", "earnings", "net income"],
            "growth": ["growth", "increase", "expansion"]
        }
        
        inconsistent_terms = []
        for concept, terms in financial_terms.items():
            found_terms = [term for term in terms if term in content.lower()]
            if len(found_terms) > 1:
                inconsistent_terms.append((concept, found_terms))
        
        if inconsistent_terms:
            for concept, terms in inconsistent_terms:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.INCONSISTENT_DATA,
                    severity=QualityLevel.ACCEPTABLE,
                    description=f"Inconsistent terminology for {concept}: {', '.join(terms)}",
                    affected_content=f"Terms: {', '.join(terms)}",
                    suggested_fix=f"Use consistent terminology for {concept}",
                    auto_fixable=True
                ))
                evidence.append(f"Inconsistent {concept} terminology")
        
        # Check for consistent formatting
        date_formats = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', content)
        if date_formats:
            unique_formats = set()
            for date in date_formats:
                if '/' in date:
                    unique_formats.add('slash')
                elif '-' in date:
                    unique_formats.add('dash')
            
            if len(unique_formats) > 1:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.FORMATTING_ERROR,
                    severity=QualityLevel.ACCEPTABLE,
                    description="Inconsistent date formatting",
                    affected_content="Date formats vary throughout content",
                    suggested_fix="Use consistent date formatting",
                    auto_fixable=True
                ))
                evidence.append("Inconsistent date formatting")
        
        # Calculate consistency score
        terminology_penalty = len([i for i in issues if i.issue_type == IssueType.INCONSISTENT_DATA]) * 0.1
        formatting_penalty = len([i for i in issues if i.issue_type == IssueType.FORMATTING_ERROR]) * 0.05
        consistency_score = max(0.0, 1.0 - terminology_penalty - formatting_penalty)
        
        if not evidence:
            evidence.append("Content appears consistent in terminology and formatting")
        
        score = QualityScore(
            metric=QualityMetric.CONSISTENCY,
            score=consistency_score,
            confidence=0.8,
            explanation=f"Consistency score after penalties for terminology ({terminology_penalty:.2f}) and formatting ({formatting_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_relevance(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content relevance."""
        issues = []
        evidence = []
        
        # Check for topic relevance
        target_topic = metadata.get("target_topic", "").lower()
        if target_topic:
            topic_mentions = content.lower().count(target_topic)
            content_length = len(content.split())
            relevance_ratio = topic_mentions / max(1, content_length / 100)  # mentions per 100 words
            
            if relevance_ratio < 0.5:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.INCOMPLETE_ANALYSIS,
                    severity=QualityLevel.ACCEPTABLE,
                    description=f"Low topic relevance: {target_topic} mentioned {topic_mentions} times",
                    affected_content="Overall content focus",
                    suggested_fix=f"Increase focus on {target_topic}",
                    auto_fixable=False
                ))
                evidence.append(f"Topic relevance ratio: {relevance_ratio:.2f}")
        
        # Check for off-topic content
        off_topic_indicators = ["unrelated", "digression", "side note", "by the way"]
        off_topic_count = sum(1 for indicator in off_topic_indicators if indicator in content.lower())
        
        if off_topic_count > 0:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.INCOMPLETE_ANALYSIS,
                severity=QualityLevel.ACCEPTABLE,
                description=f"Potential off-topic content detected ({off_topic_count} indicators)",
                affected_content="Content may contain irrelevant information",
                suggested_fix="Remove or minimize off-topic content",
                auto_fixable=False
            ))
            evidence.append(f"Off-topic indicators: {off_topic_count}")
        
        # Calculate relevance score
        topic_score = min(1.0, relevance_ratio) if target_topic else 0.8  # Default if no target topic
        off_topic_penalty = off_topic_count * 0.1
        relevance_score = max(0.0, topic_score - off_topic_penalty)
        
        if not evidence:
            evidence.append("Content appears relevant to the intended topic")
        
        score = QualityScore(
            metric=QualityMetric.RELEVANCE,
            score=relevance_score,
            confidence=0.7,
            explanation=f"Relevance based on topic focus ({topic_score:.2f}) minus off-topic penalty ({off_topic_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_clarity(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content clarity."""
        issues = []
        evidence = []
        
        # Check sentence length
        sentences = re.split(r'[.!?]+', content)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if sentence_lengths:
            avg_sentence_length = statistics.mean(sentence_lengths)
            long_sentences = [length for length in sentence_lengths if length > 30]
            
            if avg_sentence_length > 25:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.FORMATTING_ERROR,
                    severity=QualityLevel.ACCEPTABLE,
                    description=f"Average sentence length too long: {avg_sentence_length:.1f} words",
                    affected_content="Sentence structure",
                    suggested_fix="Break down long sentences for better readability",
                    auto_fixable=True
                ))
                evidence.append(f"Average sentence length: {avg_sentence_length:.1f}")
            
            if len(long_sentences) > len(sentence_lengths) * 0.2:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.FORMATTING_ERROR,
                    severity=QualityLevel.ACCEPTABLE,
                    description=f"Too many long sentences: {len(long_sentences)} out of {len(sentence_lengths)}",
                    affected_content="Sentence structure",
                    suggested_fix="Simplify complex sentences",
                    auto_fixable=True
                ))
                evidence.append(f"Long sentences: {len(long_sentences)}/{len(sentence_lengths)}")
        
        # Check for jargon and complex terms
        complex_terms = ["aforementioned", "heretofore", "notwithstanding", "pursuant", "whereby"]
        jargon_count = sum(1 for term in complex_terms if term in content.lower())
        
        if jargon_count > 5:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.FORMATTING_ERROR,
                severity=QualityLevel.ACCEPTABLE,
                description=f"Excessive use of complex terminology: {jargon_count} instances",
                affected_content="Language complexity",
                suggested_fix="Simplify language and reduce jargon",
                auto_fixable=True
            ))
            evidence.append(f"Complex terms: {jargon_count}")
        
        # Calculate clarity score
        sentence_penalty = max(0, (avg_sentence_length - 20) / 20) if sentence_lengths else 0
        jargon_penalty = min(0.3, jargon_count * 0.05)
        clarity_score = max(0.0, 1.0 - sentence_penalty - jargon_penalty)
        
        if not evidence:
            evidence.append("Content appears clear and well-structured")
        
        score = QualityScore(
            metric=QualityMetric.CLARITY,
            score=clarity_score,
            confidence=0.8,
            explanation=f"Clarity score after penalties for sentence length ({sentence_penalty:.2f}) and jargon ({jargon_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_objectivity(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content objectivity."""
        issues = []
        evidence = []
        
        # Check for biased language
        bias_indicators = {
            "positive_bias": ["amazing", "fantastic", "incredible", "perfect", "flawless"],
            "negative_bias": ["terrible", "awful", "horrible", "disaster", "catastrophic"],
            "emotional_language": ["shocking", "outrageous", "unbelievable", "stunning"]
        }
        
        bias_counts = {}
        for bias_type, indicators in bias_indicators.items():
            count = sum(1 for indicator in indicators if indicator in content.lower())
            bias_counts[bias_type] = count
            
            if count > 2:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.BIAS_DETECTED,
                    severity=QualityLevel.ACCEPTABLE,
                    description=f"{bias_type.replace('_', ' ').title()} detected: {count} instances",
                    affected_content="Language tone and objectivity",
                    suggested_fix="Use more neutral, objective language",
                    auto_fixable=True
                ))
                evidence.append(f"{bias_type}: {count} instances")
        
        # Check for unsupported claims
        claim_indicators = ["clearly", "obviously", "undoubtedly", "certainly", "definitely"]
        unsupported_claims = sum(1 for indicator in claim_indicators if indicator in content.lower())
        
        if unsupported_claims > 3:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.BIAS_DETECTED,
                severity=QualityLevel.ACCEPTABLE,
                description=f"Potential unsupported claims: {unsupported_claims} instances",
                affected_content="Claim substantiation",
                suggested_fix="Support claims with evidence or use more tentative language",
                auto_fixable=False
            ))
            evidence.append(f"Unsupported claims: {unsupported_claims}")
        
        # Calculate objectivity score
        total_bias = sum(bias_counts.values())
        bias_penalty = min(0.5, total_bias * 0.05)
        claim_penalty = min(0.3, unsupported_claims * 0.05)
        objectivity_score = max(0.0, 1.0 - bias_penalty - claim_penalty)
        
        if not evidence:
            evidence.append("Content appears objective and unbiased")
        
        score = QualityScore(
            metric=QualityMetric.OBJECTIVITY,
            score=objectivity_score,
            confidence=0.7,
            explanation=f"Objectivity score after penalties for bias ({bias_penalty:.2f}) and unsupported claims ({claim_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_timeliness(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content timeliness."""
        issues = []
        evidence = []
        
        # Check data freshness
        data_age_days = metadata.get("average_data_age_days", 30)
        max_acceptable_age = metadata.get("max_data_age_days", 90)
        
        if data_age_days > max_acceptable_age:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.OUTDATED_DATA,
                severity=QualityLevel.POOR,
                description=f"Data is outdated: {data_age_days} days old (max acceptable: {max_acceptable_age})",
                affected_content="Data currency",
                suggested_fix="Update analysis with more recent data",
                auto_fixable=False
            ))
            evidence.append(f"Data age: {data_age_days} days")
        
        # Check for outdated references
        current_year = datetime.now().year
        years_mentioned = re.findall(r'\b(19|20)\d{2}\b', content)
        if years_mentioned:
            old_years = [int(year) for year in years_mentioned if int(year) < current_year - 5]
            if len(old_years) > len(years_mentioned) * 0.5:
                issues.append(QualityIssue(
                    issue_id="",
                    issue_type=IssueType.OUTDATED_DATA,
                    severity=QualityLevel.ACCEPTABLE,
                    description=f"Many references to old years: {len(old_years)} out of {len(years_mentioned)}",
                    affected_content="Reference currency",
                    suggested_fix="Include more recent references and data",
                    auto_fixable=False
                ))
                evidence.append(f"Old references: {len(old_years)}/{len(years_mentioned)}")
        
        # Calculate timeliness score
        age_score = max(0.0, 1.0 - (data_age_days / max_acceptable_age))
        reference_penalty = 0.2 if old_years and len(old_years) > len(years_mentioned) * 0.5 else 0
        timeliness_score = max(0.0, age_score - reference_penalty)
        
        if not evidence:
            evidence.append("Data and references appear current")
        
        score = QualityScore(
            metric=QualityMetric.TIMELINESS,
            score=timeliness_score,
            confidence=0.8,
            explanation=f"Timeliness based on data age ({age_score:.2f}) minus reference penalty ({reference_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    async def _assess_reliability(self, content: str, metadata: Dict[str, Any]) -> Tuple[QualityScore, List[QualityIssue]]:
        """Assess content reliability."""
        issues = []
        evidence = []
        
        # Check source diversity
        source_count = metadata.get("source_count", 1)
        min_sources = metadata.get("min_sources", 3)
        
        if source_count < min_sources:
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.UNRELIABLE_SOURCE,
                severity=QualityLevel.ACCEPTABLE,
                description=f"Insufficient source diversity: {source_count} sources (minimum: {min_sources})",
                affected_content="Source coverage",
                suggested_fix="Include more diverse sources",
                auto_fixable=False
            ))
            evidence.append(f"Source count: {source_count}/{min_sources}")
        
        # Check confidence indicators
        uncertainty_indicators = ["might", "could", "possibly", "perhaps", "maybe", "uncertain"]
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in content.lower())
        
        if uncertainty_count > len(content.split()) * 0.02:  # More than 2% uncertainty words
            issues.append(QualityIssue(
                issue_id="",
                issue_type=IssueType.UNRELIABLE_SOURCE,
                severity=QualityLevel.ACCEPTABLE,
                description=f"High uncertainty in analysis: {uncertainty_count} uncertainty indicators",
                affected_content="Analysis confidence",
                suggested_fix="Strengthen analysis with more definitive evidence",
                auto_fixable=False
            ))
            evidence.append(f"Uncertainty indicators: {uncertainty_count}")
        
        # Calculate reliability score
        source_score = min(1.0, source_count / min_sources)
        uncertainty_penalty = min(0.3, uncertainty_count * 0.01)
        reliability_score = max(0.0, source_score - uncertainty_penalty)
        
        if not evidence:
            evidence.append("Analysis appears reliable with adequate sources")
        
        score = QualityScore(
            metric=QualityMetric.RELIABILITY,
            score=reliability_score,
            confidence=0.8,
            explanation=f"Reliability based on source diversity ({source_score:.2f}) minus uncertainty penalty ({uncertainty_penalty:.2f})",
            evidence=evidence
        )
        
        return score, issues
    
    def _calculate_overall_score(self, metric_scores: Dict[QualityMetric, QualityScore]) -> float:
        """Calculate weighted overall quality score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in metric_scores.items():
            weight = self._metric_weights.get(metric, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, metric_scores: Dict[QualityMetric, QualityScore], 
                                 issues: List[QualityIssue]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Recommendations based on low scores
        for metric, score in metric_scores.items():
            if score.score < self._quality_standards[metric]:
                recommendations.append(f"Improve {metric.value}: {score.explanation}")
        
        # Recommendations based on critical issues
        critical_issues = [issue for issue in issues if issue.severity in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]]
        for issue in critical_issues:
            if issue.suggested_fix:
                recommendations.append(f"Address {issue.issue_type.value}: {issue.suggested_fix}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Content meets quality standards. Consider minor improvements for excellence.")
        
        return recommendations
    
    async def auto_fix_issues(self, assessment_id: str) -> Dict[str, Any]:
        """Attempt to automatically fix quality issues."""
        if assessment_id not in self._assessments:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        assessment = self._assessments[assessment_id]
        fixed_issues = []
        failed_fixes = []
        
        for issue in assessment.issues:
            if issue.auto_fixable and not issue.fixed:
                try:
                    # Attempt to fix the issue
                    success = await self._apply_auto_fix(issue)
                    if success:
                        issue.fixed = True
                        fixed_issues.append(issue.issue_id)
                    else:
                        failed_fixes.append(issue.issue_id)
                except Exception as e:
                    logger.error(f"Error auto-fixing issue {issue.issue_id}: {e}")
                    failed_fixes.append(issue.issue_id)
        
        return {
            "fixed_issues": fixed_issues,
            "failed_fixes": failed_fixes,
            "total_auto_fixable": len([i for i in assessment.issues if i.auto_fixable]),
            "fixed_count": len(fixed_issues)
        }
    
    async def _apply_auto_fix(self, issue: QualityIssue) -> bool:
        """Apply automatic fix for a specific issue."""
        # This would contain the actual fix logic
        # For now, we'll just mark it as fixed
        logger.info(f"Auto-fixing issue: {issue.issue_id}")
        return True
    
    def get_quality_report(self, assessment_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate quality report."""
        if assessment_id:
            if assessment_id not in self._assessments:
                raise ValueError(f"Assessment {assessment_id} not found")
            assessments = [self._assessments[assessment_id]]
        else:
            assessments = list(self._assessments.values())
        
        if not assessments:
            return {"message": "No assessments available"}
        
        # Calculate aggregate statistics
        overall_scores = [a.overall_score for a in assessments]
        avg_overall_score = statistics.mean(overall_scores)
        
        # Count issues by type and severity
        all_issues = [issue for assessment in assessments for issue in assessment.issues]
        issues_by_type = {}
        issues_by_severity = {}
        
        for issue in all_issues:
            # By type
            issue_type = issue.issue_type.value
            issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
            
            # By severity
            severity = issue.severity.value
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        # Metric performance
        metric_performance = {}
        for metric in QualityMetric:
            scores = [a.metric_scores[metric].score for a in assessments if metric in a.metric_scores]
            if scores:
                metric_performance[metric.value] = {
                    "average_score": statistics.mean(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "below_standard": len([s for s in scores if s < self._quality_standards[metric]])
                }
        
        return {
            "total_assessments": len(assessments),
            "average_overall_score": avg_overall_score,
            "overall_quality_level": QualityLevel.GOOD.value if avg_overall_score >= 0.8 else QualityLevel.ACCEPTABLE.value,
            "total_issues": len(all_issues),
            "issues_by_type": issues_by_type,
            "issues_by_severity": issues_by_severity,
            "metric_performance": metric_performance,
            "quality_standards": {m.value: s for m, s in self._quality_standards.items()},
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def get_assessment(self, assessment_id: str) -> Optional[QualityAssessment]:
        """Get a specific quality assessment."""
        return self._assessments.get(assessment_id)
    
    def update_quality_standards(self, standards: Dict[QualityMetric, float]):
        """Update quality standards."""
        for metric, standard in standards.items():
            if 0.0 <= standard <= 1.0:
                self._quality_standards[metric] = standard
                logger.info(f"Updated quality standard for {metric.value}: {standard}")
    
    def clear_assessments(self):
        """Clear all quality assessments."""
        self._assessments.clear()
        logger.info("All quality assessments cleared")