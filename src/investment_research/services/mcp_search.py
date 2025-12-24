"""MCP (Model Context Protocol) search service implementation."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx
import asyncio

from ..config.settings import settings


@dataclass
class SearchQuery:
    """Search query for MCP service."""
    
    query: str
    domain: Optional[str] = None
    filters: Dict[str, Any] = None
    limit: int = 10
    offset: int = 0


@dataclass
class MarketData:
    """Market data response from MCP service."""
    
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    quality_score: float


@dataclass
class RealTimeData:
    """Real-time data response from MCP service."""
    
    symbols: List[str]
    data: Dict[str, Any]
    timestamp: datetime
    latency_ms: float


@dataclass
class QualityScore:
    """Data quality assessment."""
    
    score: float
    factors: Dict[str, float]
    issues: List[str]
    recommendations: List[str]


class MCPSearchService:
    """MCP search service for external data retrieval."""
    
    def __init__(self):
        """Initialize MCP search service."""
        self.endpoint = settings.mcp_search.endpoint
        self.api_key = settings.mcp_search.api_key
        self.timeout = settings.mcp_search.timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def search_market_data(self, query: SearchQuery) -> MarketData:
        """Search for market data using MCP protocol.
        
        Args:
            query: Search query parameters
        
        Returns:
            Market data response
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "query": query.query,
            "domain": query.domain,
            "filters": query.filters or {},
            "limit": query.limit,
            "offset": query.offset
        }
        
        try:
            response = await self.client.post(
                f"{self.endpoint}/search/market",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            
            return MarketData(
                symbol=data.get("symbol", ""),
                data=data.get("data", {}),
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat())
                ),
                source=data.get("source", "mcp"),
                quality_score=data.get("quality_score", 0.5)
            )
        
        except httpx.RequestError as e:
            raise Exception(f"MCP search request failed: {e}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"MCP search HTTP error: {e.response.status_code}")
    
    async def get_real_time_data(self, symbols: List[str]) -> RealTimeData:
        """Get real-time data for specified symbols.
        
        Args:
            symbols: List of symbols to retrieve data for
        
        Returns:
            Real-time data response
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {"symbols": symbols}
        
        start_time = datetime.utcnow()
        
        try:
            response = await self.client.post(
                f"{self.endpoint}/realtime",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            data = response.json()
            
            return RealTimeData(
                symbols=symbols,
                data=data.get("data", {}),
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat())
                ),
                latency_ms=latency_ms
            )
        
        except httpx.RequestError as e:
            raise Exception(f"Real-time data request failed: {e}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Real-time data HTTP error: {e.response.status_code}")
    
    def validate_data_quality(self, data: Any) -> QualityScore:
        """Validate the quality of retrieved data.
        
        Args:
            data: Data to validate
        
        Returns:
            Quality score assessment
        """
        factors = {}
        issues = []
        recommendations = []
        
        # Completeness check
        if isinstance(data, dict):
            completeness = len([v for v in data.values() if v is not None]) / len(data)
            factors["completeness"] = completeness
            
            if completeness < 0.8:
                issues.append("Data has missing values")
                recommendations.append("Consider using alternative data sources")
        
        # Freshness check (placeholder - would need timestamp info)
        factors["freshness"] = 0.8  # Default assumption
        
        # Consistency check (placeholder - would need historical data)
        factors["consistency"] = 0.7  # Default assumption
        
        # Calculate overall score
        score = sum(factors.values()) / len(factors) if factors else 0.0
        
        return QualityScore(
            score=score,
            factors=factors,
            issues=issues,
            recommendations=recommendations
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()