"""MCP (Model Context Protocol) search service implementation."""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import asyncio
import json
import hashlib
from enum import Enum
import logging
import aioredis
from contextlib import asynccontextmanager

from ..config.settings import settings


logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """External data source types."""
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    SEC_EDGAR = "sec_edgar"
    CUSTOM_API = "custom_api"


@dataclass
class ExternalDataSource:
    """Configuration for external data sources."""
    
    source_type: DataSourceType
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class CacheEntry:
    """Cache entry for storing search results."""
    
    data: Any
    timestamp: datetime
    ttl_seconds: int
    quality_score: float
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl_seconds)


@dataclass
class DataValidationResult:
    """Result of data validation process."""
    
    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Search query for MCP service."""
    
    query: str
    domain: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    data_sources: List[DataSourceType] = field(default_factory=list)
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour default


@dataclass
class MarketData:
    """Market data response from MCP service."""
    
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    quality_score: float
    validation_result: Optional[DataValidationResult] = None


@dataclass
class RealTimeData:
    """Real-time data response from MCP service."""
    
    symbols: List[str]
    data: Dict[str, Any]
    timestamp: datetime
    latency_ms: float
    source: str
    quality_score: float


@dataclass
class QualityScore:
    """Data quality assessment."""
    
    score: float
    factors: Dict[str, float]
    issues: List[str]
    recommendations: List[str]


class MCPSearchService:
    """Enhanced MCP search service for external data retrieval."""
    
    def __init__(self):
        """Initialize MCP search service."""
        self.endpoint = settings.mcp_search.endpoint
        self.api_key = settings.mcp_search.api_key
        self.timeout = settings.mcp_search.timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        # Initialize cache (Redis if available, fallback to in-memory)
        self._cache: Dict[str, CacheEntry] = {}
        self._redis_client: Optional[aioredis.Redis] = None
        
        # Initialize external data sources
        self._data_sources = self._initialize_data_sources()
        
        # Rate limiting
        self._rate_limits: Dict[str, List[datetime]] = {}
        
        # Circuit breaker for external services
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    async def initialize_redis(self):
        """Initialize Redis connection for caching."""
        try:
            self._redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}. Using in-memory cache.")
            self._redis_client = None
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache (Redis or in-memory)."""
        if self._redis_client:
            try:
                cached_data = await self._redis_client.get(f"mcp_search:{cache_key}")
                if cached_data:
                    cache_entry = json.loads(cached_data)
                    # Check if expired
                    timestamp = datetime.fromisoformat(cache_entry["timestamp"])
                    if datetime.utcnow() < timestamp + timedelta(seconds=cache_entry["ttl_seconds"]):
                        return cache_entry["data"]
                    else:
                        # Remove expired entry
                        await self._redis_client.delete(f"mcp_search:{cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        
        # Fallback to in-memory cache
        if cache_key in self._cache and not self._cache[cache_key].is_expired():
            return self._cache[cache_key].data
        
        return None
    
    async def _set_cache(self, cache_key: str, data: Any, ttl_seconds: int, quality_score: float):
        """Set data in cache (Redis or in-memory)."""
        cache_entry_data = {
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "ttl_seconds": ttl_seconds,
            "quality_score": quality_score
        }
        
        if self._redis_client:
            try:
                await self._redis_client.setex(
                    f"mcp_search:{cache_key}",
                    ttl_seconds,
                    json.dumps(cache_entry_data, default=str)
                )
                return
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
        
        # Fallback to in-memory cache
        cache_entry = CacheEntry(
            data=data,
            timestamp=datetime.utcnow(),
            ttl_seconds=ttl_seconds,
            quality_score=quality_score
        )
        self._cache[cache_key] = cache_entry
    
    def _get_circuit_breaker_state(self, source_type: DataSourceType) -> Dict[str, Any]:
        """Get circuit breaker state for a data source."""
        if source_type.value not in self._circuit_breakers:
            self._circuit_breakers[source_type.value] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure_time": None,
                "success_count": 0,
                "failure_threshold": 5,
                "recovery_timeout": 60,  # seconds
                "success_threshold": 3  # for half_open -> closed
            }
        return self._circuit_breakers[source_type.value]
    
    def _record_success(self, source_type: DataSourceType):
        """Record successful request for circuit breaker."""
        cb_state = self._get_circuit_breaker_state(source_type)
        cb_state["failure_count"] = 0
        cb_state["success_count"] += 1
        
        if cb_state["state"] == "half_open" and cb_state["success_count"] >= cb_state["success_threshold"]:
            cb_state["state"] = "closed"
            cb_state["success_count"] = 0
            logger.info(f"Circuit breaker for {source_type} closed (recovered)")
    
    def _record_failure(self, source_type: DataSourceType):
        """Record failed request for circuit breaker."""
        cb_state = self._get_circuit_breaker_state(source_type)
        cb_state["failure_count"] += 1
        cb_state["last_failure_time"] = datetime.utcnow()
        cb_state["success_count"] = 0
        
        if cb_state["failure_count"] >= cb_state["failure_threshold"]:
            cb_state["state"] = "open"
            logger.warning(f"Circuit breaker for {source_type} opened due to failures")
    
    def _can_make_request(self, source_type: DataSourceType) -> bool:
        """Check if request can be made based on circuit breaker state."""
        cb_state = self._get_circuit_breaker_state(source_type)
        
        if cb_state["state"] == "closed":
            return True
        elif cb_state["state"] == "open":
            # Check if recovery timeout has passed
            if cb_state["last_failure_time"]:
                time_since_failure = (datetime.utcnow() - cb_state["last_failure_time"]).total_seconds()
                if time_since_failure >= cb_state["recovery_timeout"]:
                    cb_state["state"] = "half_open"
                    logger.info(f"Circuit breaker for {source_type} moved to half-open")
                    return True
            return False
        elif cb_state["state"] == "half_open":
            return True
        
        return False
    
    def _initialize_data_sources(self) -> Dict[DataSourceType, ExternalDataSource]:
        """Initialize external data source configurations."""
        sources = {}
        
        # Bloomberg configuration
        if settings.bloomberg_api_key:
            sources[DataSourceType.BLOOMBERG] = ExternalDataSource(
                source_type=DataSourceType.BLOOMBERG,
                api_key=settings.bloomberg_api_key,
                endpoint="https://api.bloomberg.com/v1",
                rate_limit_per_minute=100,
                headers={"Content-Type": "application/json"}
            )
        
        # Reuters configuration
        if settings.reuters_api_key:
            sources[DataSourceType.REUTERS] = ExternalDataSource(
                source_type=DataSourceType.REUTERS,
                api_key=settings.reuters_api_key,
                endpoint="https://api.reuters.com/v1",
                rate_limit_per_minute=60,
                headers={"Content-Type": "application/json"}
            )
        
        # Alpha Vantage configuration
        if settings.alpha_vantage_api_key:
            sources[DataSourceType.ALPHA_VANTAGE] = ExternalDataSource(
                source_type=DataSourceType.ALPHA_VANTAGE,
                api_key=settings.alpha_vantage_api_key,
                endpoint="https://www.alphavantage.co/query",
                rate_limit_per_minute=5,  # Free tier limit
                headers={"Content-Type": "application/json"}
            )
        
        # Yahoo Finance (no API key required)
        sources[DataSourceType.YAHOO_FINANCE] = ExternalDataSource(
            source_type=DataSourceType.YAHOO_FINANCE,
            endpoint="https://query1.finance.yahoo.com/v8/finance/chart",
            rate_limit_per_minute=100,
            headers={"User-Agent": "Investment Research System"}
        )
        
        return sources
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        query_str = json.dumps({
            "query": query.query,
            "domain": query.domain,
            "filters": query.filters,
            "limit": query.limit,
            "offset": query.offset,
            "data_sources": [ds.value for ds in query.data_sources]
        }, sort_keys=True)
        
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _check_rate_limit(self, source_type: DataSourceType) -> bool:
        """Check if rate limit allows request for given source."""
        if source_type not in self._data_sources:
            return False
        
        source = self._data_sources[source_type]
        now = datetime.utcnow()
        
        # Clean old requests (older than 1 minute)
        if source_type.value in self._rate_limits:
            self._rate_limits[source_type.value] = [
                req_time for req_time in self._rate_limits[source_type.value]
                if now - req_time < timedelta(minutes=1)
            ]
        else:
            self._rate_limits[source_type.value] = []
        
        # Check if under rate limit
        return len(self._rate_limits[source_type.value]) < source.rate_limit_per_minute
    
    def _record_request(self, source_type: DataSourceType):
        """Record a request for rate limiting."""
        if source_type.value not in self._rate_limits:
            self._rate_limits[source_type.value] = []
        
        self._rate_limits[source_type.value].append(datetime.utcnow())
    
    async def search_market_data(self, query: SearchQuery) -> MarketData:
        """Search for market data using MCP protocol and external sources.
        
        Args:
            query: Search query parameters
        
        Returns:
            Market data response
        """
        # Check cache first
        if query.use_cache:
            cache_key = self._generate_cache_key(query)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Returning cached result for query: {query.query}")
                return cached_result
        
        # Try external data sources first
        for source_type in query.data_sources:
            if (source_type in self._data_sources and 
                self._check_rate_limit(source_type) and 
                self._can_make_request(source_type)):
                try:
                    result = await self._search_external_source(query, source_type)
                    if result:
                        self._record_success(source_type)
                        
                        # Cache the result
                        if query.use_cache:
                            await self._set_cache(
                                cache_key, 
                                result, 
                                query.cache_ttl, 
                                result.quality_score
                            )
                        
                        return result
                except Exception as e:
                    logger.warning(f"Failed to search {source_type}: {e}")
                    self._record_failure(source_type)
                    continue
        
        # Fallback to MCP search
        try:
            result = await self._search_mcp_fallback(query)
            
            # Cache the fallback result
            if query.use_cache:
                await self._set_cache(
                    cache_key, 
                    result, 
                    query.cache_ttl, 
                    result.quality_score
                )
            
            return result
        except Exception as e:
            logger.error(f"MCP fallback search failed: {e}")
            raise
    
    async def _search_external_source(
        self, 
        query: SearchQuery, 
        source_type: DataSourceType
    ) -> Optional[MarketData]:
        """Search specific external data source."""
        source = self._data_sources[source_type]
        
        if not source.is_active or not self._check_rate_limit(source_type):
            return None
        
        self._record_request(source_type)
        
        try:
            if source_type == DataSourceType.ALPHA_VANTAGE:
                return await self._search_alpha_vantage(query, source)
            elif source_type == DataSourceType.YAHOO_FINANCE:
                return await self._search_yahoo_finance(query, source)
            # Add more source implementations as needed
            
        except Exception as e:
            logger.error(f"Error searching {source_type}: {e}")
            return None
    
    async def _search_alpha_vantage(
        self, 
        query: SearchQuery, 
        source: ExternalDataSource
    ) -> Optional[MarketData]:
        """Search Alpha Vantage API."""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": query.query,
            "apikey": source.api_key
        }
        
        try:
            response = await self.client.get(
                source.endpoint,
                params=params,
                headers=source.headers,
                timeout=source.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Validate and process Alpha Vantage response
            if "Global Quote" in data:
                quote_data = data["Global Quote"]
                
                market_data = MarketData(
                    symbol=quote_data.get("01. symbol", query.query),
                    data=quote_data,
                    timestamp=datetime.utcnow(),
                    source="alpha_vantage",
                    quality_score=0.8  # Alpha Vantage is generally reliable
                )
                
                # Validate data quality
                validation_result = self.validate_data_quality(quote_data)
                market_data.validation_result = validation_result
                market_data.quality_score = validation_result.quality_score
                
                return market_data
            
        except Exception as e:
            logger.error(f"Alpha Vantage search failed: {e}")
            return None
    
    async def _search_yahoo_finance(
        self, 
        query: SearchQuery, 
        source: ExternalDataSource
    ) -> Optional[MarketData]:
        """Search Yahoo Finance API."""
        url = f"{source.endpoint}/{query.query}"
        
        try:
            response = await self.client.get(
                url,
                headers=source.headers,
                timeout=source.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Process Yahoo Finance response
            if "chart" in data and "result" in data["chart"]:
                chart_data = data["chart"]["result"][0]
                
                market_data = MarketData(
                    symbol=chart_data.get("meta", {}).get("symbol", query.query),
                    data=chart_data,
                    timestamp=datetime.utcnow(),
                    source="yahoo_finance",
                    quality_score=0.7  # Yahoo Finance is moderately reliable
                )
                
                # Validate data quality
                validation_result = self.validate_data_quality(chart_data)
                market_data.validation_result = validation_result
                market_data.quality_score = validation_result.quality_score
                
                return market_data
            
        except Exception as e:
            logger.error(f"Yahoo Finance search failed: {e}")
            return None
    
    async def _search_mcp_fallback(self, query: SearchQuery) -> MarketData:
        """Fallback to original MCP search implementation."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "query": query.query,
            "domain": query.domain,
            "filters": query.filters,
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
            
            market_data = MarketData(
                symbol=data.get("symbol", ""),
                data=data.get("data", {}),
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat())
                ),
                source="mcp",
                quality_score=data.get("quality_score", 0.5)
            )
            
            # Validate data quality
            validation_result = self.validate_data_quality(market_data.data)
            market_data.validation_result = validation_result
            market_data.quality_score = validation_result.quality_score
            
            return market_data
        
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
            
            real_time_data = RealTimeData(
                symbols=symbols,
                data=data.get("data", {}),
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat())
                ),
                latency_ms=latency_ms,
                source="mcp",
                quality_score=0.5
            )
            
            # Validate data quality
            validation_result = self.validate_data_quality(real_time_data.data)
            real_time_data.quality_score = validation_result.quality_score
            
            return real_time_data
        
        except httpx.RequestError as e:
            raise Exception(f"Real-time data request failed: {e}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Real-time data HTTP error: {e.response.status_code}")
    
    async def batch_search(self, queries: List[SearchQuery]) -> List[MarketData]:
        """Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
        
        Returns:
            List of market data responses
        """
        # Use asyncio.gather for concurrent execution
        tasks = [self.search_market_data(query) for query in queries]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch search failed for query {i}: {result}")
                    # Create a placeholder result with error info
                    error_result = MarketData(
                        symbol=queries[i].query,
                        data={"error": str(result)},
                        timestamp=datetime.utcnow(),
                        source="error",
                        quality_score=0.0
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise
    
    async def get_data_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all configured data sources.
        
        Returns:
            Dictionary with health status for each data source
        """
        health_status = {}
        
        for source_type, source_config in self._data_sources.items():
            cb_state = self._get_circuit_breaker_state(source_type)
            rate_limit_status = self._check_rate_limit(source_type)
            
            health_status[source_type.value] = {
                "is_active": source_config.is_active,
                "circuit_breaker_state": cb_state["state"],
                "failure_count": cb_state["failure_count"],
                "rate_limit_available": rate_limit_status,
                "endpoint": source_config.endpoint,
                "last_health_check": datetime.utcnow().isoformat()
            }
            
            # Perform basic health check
            if source_config.is_active and cb_state["state"] != "open":
                try:
                    # Simple ping to check if service is responsive
                    health_check_response = await self._perform_health_check(source_type)
                    health_status[source_type.value]["health_check"] = health_check_response
                except Exception as e:
                    health_status[source_type.value]["health_check"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
        
        return health_status
    
    async def _perform_health_check(self, source_type: DataSourceType) -> Dict[str, Any]:
        """Perform health check for a specific data source."""
        source = self._data_sources[source_type]
        
        try:
            # Create a simple test request
            response = await self.client.get(
                source.endpoint,
                headers=source.headers,
                timeout=5  # Short timeout for health check
            )
            
            return {
                "status": "healthy" if response.status_code < 400 else "degraded",
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def validate_data_quality(self, data: Any) -> DataValidationResult:
        """Enhanced data quality validation.
        
        Args:
            data: Data to validate
        
        Returns:
            Data validation result with quality score and issues
        """
        issues = []
        warnings = []
        factors = {}
        metadata = {}
        
        # Basic validation
        if data is None:
            return DataValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Data is None"],
                metadata={"validation_timestamp": datetime.utcnow().isoformat()}
            )
        
        # Completeness check
        if isinstance(data, dict):
            total_fields = len(data)
            non_null_fields = len([v for v in data.values() if v is not None])
            completeness = non_null_fields / total_fields if total_fields > 0 else 0
            factors["completeness"] = completeness
            
            if completeness < 0.5:
                issues.append("Data has significant missing values")
            elif completeness < 0.8:
                warnings.append("Data has some missing values")
            
            # Check for required financial fields
            required_fields = ["price", "volume", "timestamp", "symbol"]
            missing_required = [field for field in required_fields if field not in data]
            if missing_required:
                issues.append(f"Missing required fields: {missing_required}")
                factors["required_fields"] = (len(required_fields) - len(missing_required)) / len(required_fields)
            else:
                factors["required_fields"] = 1.0
        
        elif isinstance(data, list):
            if len(data) == 0:
                issues.append("Data list is empty")
                factors["completeness"] = 0.0
            else:
                factors["completeness"] = 1.0
        else:
            factors["completeness"] = 1.0
        
        # Freshness check (if timestamp available)
        timestamp_field = None
        if isinstance(data, dict):
            for key in ["timestamp", "time", "date", "updated_at"]:
                if key in data:
                    timestamp_field = data[key]
                    break
        
        if timestamp_field:
            try:
                if isinstance(timestamp_field, str):
                    data_time = datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                elif isinstance(timestamp_field, (int, float)):
                    data_time = datetime.fromtimestamp(timestamp_field)
                else:
                    data_time = timestamp_field
                
                age_hours = (datetime.utcnow() - data_time).total_seconds() / 3600
                
                if age_hours < 1:
                    factors["freshness"] = 1.0
                elif age_hours < 24:
                    factors["freshness"] = 0.8
                elif age_hours < 168:  # 1 week
                    factors["freshness"] = 0.6
                else:
                    factors["freshness"] = 0.3
                    warnings.append(f"Data is {age_hours:.1f} hours old")
                
                metadata["data_age_hours"] = age_hours
            except Exception:
                factors["freshness"] = 0.5
                warnings.append("Could not parse timestamp for freshness check")
        else:
            factors["freshness"] = 0.5
            warnings.append("No timestamp found for freshness validation")
        
        # Consistency check (basic range validation for numeric fields)
        if isinstance(data, dict):
            numeric_issues = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if key.lower() in ["price", "close", "open", "high", "low"] and value < 0:
                        numeric_issues.append(f"Negative price value for {key}: {value}")
                    elif key.lower() == "volume" and value < 0:
                        numeric_issues.append(f"Negative volume: {value}")
            
            if numeric_issues:
                issues.extend(numeric_issues)
                factors["consistency"] = 0.5
            else:
                factors["consistency"] = 0.9
        else:
            factors["consistency"] = 0.7
        
        # Calculate overall score
        if factors:
            score = sum(factors.values()) / len(factors)
        else:
            score = 0.0
        
        # Adjust score based on issues
        if issues:
            score *= 0.5  # Significant penalty for issues
        elif warnings:
            score *= 0.8  # Minor penalty for warnings
        
        is_valid = score >= 0.3 and len(issues) == 0
        
        metadata.update({
            "validation_timestamp": datetime.utcnow().isoformat(),
            "total_factors": len(factors),
            "issue_count": len(issues),
            "warning_count": len(warnings)
        })
        
        return DataValidationResult(
            is_valid=is_valid,
            quality_score=min(score, 1.0),
            issues=issues,
            warnings=warnings,
            metadata=metadata
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "cache_hit_potential": (total_entries - expired_entries) / max(total_entries, 1)
        }
    
    def clear_expired_cache(self):
        """Clear expired cache entries."""
        expired_keys = [
            key for key, entry in self._cache.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        await self.client.aclose()
        
        if self._redis_client:
            await self._redis_client.close()
        
        self._cache.clear()
        self._rate_limits.clear()
        self._circuit_breakers.clear()
    
    @asynccontextmanager
    async def get_service(self):
        """Context manager for MCP search service."""
        try:
            await self.initialize_redis()
            yield self
        finally:
            await self.close()