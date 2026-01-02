"""Data standardization and validation service implementation."""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from enum import Enum
import re
import logging
import json
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Supported data types for standardization."""
    FINANCIAL = "financial"
    MARKET = "market"
    COMPANY = "company"
    INDUSTRY = "industry"
    ECONOMIC = "economic"
    NEWS = "news"
    REGULATORY = "regulatory"


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    
    field_name: str
    severity: ValidationSeverity
    message: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of data validation process."""
    
    is_valid: bool
    quality_score: float
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        if issue.severity == ValidationSeverity.ERROR:
            self.issues.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        
        # Recalculate quality score
        self._recalculate_quality_score()
    
    def _recalculate_quality_score(self):
        """Recalculate quality score based on issues."""
        base_score = 1.0
        
        # Penalty for errors (more severe)
        error_penalty = len(self.issues) * 0.2
        
        # Penalty for warnings (less severe)
        warning_penalty = len(self.warnings) * 0.1
        
        self.quality_score = max(0.0, base_score - error_penalty - warning_penalty)


@dataclass
class StandardizedData:
    """Standardized data container."""
    
    data_id: str
    data_type: DataType
    original_data: Dict[str, Any]
    standardized_data: Dict[str, Any]
    validation_result: ValidationResult
    standardization_timestamp: datetime
    source_info: Dict[str, Any] = field(default_factory=dict)


class DataStandardizer:
    """Service for standardizing and validating different types of data."""
    
    def __init__(self):
        """Initialize data standardizer."""
        self._field_mappings = self._initialize_field_mappings()
        self._validation_rules = self._initialize_validation_rules()
        self._currency_codes = self._load_currency_codes()
        self._country_codes = self._load_country_codes()
    
    def _initialize_field_mappings(self) -> Dict[DataType, Dict[str, str]]:
        """Initialize field mapping configurations for different data types."""
        return {
            DataType.FINANCIAL: {
                # Common financial field mappings
                "price": ["price", "close", "closing_price", "last_price", "current_price"],
                "volume": ["volume", "vol", "trading_volume", "shares_traded"],
                "market_cap": ["market_cap", "marketcap", "market_capitalization"],
                "revenue": ["revenue", "sales", "total_revenue", "net_sales"],
                "profit": ["profit", "net_income", "earnings", "net_profit"],
                "assets": ["total_assets", "assets", "total_asset"],
                "liabilities": ["total_liabilities", "liabilities", "total_liability"],
                "equity": ["shareholders_equity", "equity", "stockholders_equity"],
                "eps": ["eps", "earnings_per_share", "basic_eps"],
                "pe_ratio": ["pe_ratio", "p_e_ratio", "price_earnings_ratio"],
                "dividend": ["dividend", "dividend_per_share", "annual_dividend"]
            },
            DataType.MARKET: {
                "symbol": ["symbol", "ticker", "stock_symbol", "security_id"],
                "exchange": ["exchange", "market", "trading_venue"],
                "sector": ["sector", "industry_sector", "gics_sector"],
                "industry": ["industry", "sub_industry", "industry_group"],
                "country": ["country", "domicile", "headquarters_country"],
                "currency": ["currency", "base_currency", "trading_currency"]
            },
            DataType.COMPANY: {
                "name": ["company_name", "name", "legal_name", "entity_name"],
                "description": ["description", "business_description", "company_overview"],
                "employees": ["employees", "employee_count", "total_employees"],
                "founded": ["founded", "incorporation_date", "established"],
                "headquarters": ["headquarters", "hq", "main_office"],
                "website": ["website", "url", "company_website"],
                "ceo": ["ceo", "chief_executive", "chief_executive_officer"]
            }
        }
    
    def _initialize_validation_rules(self) -> Dict[DataType, Dict[str, Any]]:
        """Initialize validation rules for different data types."""
        return {
            DataType.FINANCIAL: {
                "price": {"type": "numeric", "min": 0, "required": True},
                "volume": {"type": "integer", "min": 0, "required": False},
                "market_cap": {"type": "numeric", "min": 0, "required": False},
                "revenue": {"type": "numeric", "required": False},
                "profit": {"type": "numeric", "required": False},
                "assets": {"type": "numeric", "min": 0, "required": False},
                "liabilities": {"type": "numeric", "min": 0, "required": False},
                "equity": {"type": "numeric", "required": False},
                "eps": {"type": "numeric", "required": False},
                "pe_ratio": {"type": "numeric", "min": 0, "required": False},
                "dividend": {"type": "numeric", "min": 0, "required": False}
            },
            DataType.MARKET: {
                "symbol": {"type": "string", "pattern": r"^[A-Z0-9]{1,10}$", "required": True},
                "exchange": {"type": "string", "required": False},
                "sector": {"type": "string", "required": False},
                "industry": {"type": "string", "required": False},
                "country": {"type": "country_code", "required": False},
                "currency": {"type": "currency_code", "required": False}
            },
            DataType.COMPANY: {
                "name": {"type": "string", "min_length": 1, "max_length": 200, "required": True},
                "description": {"type": "string", "max_length": 5000, "required": False},
                "employees": {"type": "integer", "min": 0, "required": False},
                "founded": {"type": "date", "required": False},
                "headquarters": {"type": "string", "required": False},
                "website": {"type": "url", "required": False},
                "ceo": {"type": "string", "required": False}
            }
        }
    
    def _load_currency_codes(self) -> set:
        """Load valid ISO currency codes."""
        # Common currency codes - in production, load from a comprehensive source
        return {
            "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
            "CNY", "HKD", "SGD", "KRW", "INR", "BRL", "MXN", "RUB",
            "ZAR", "TRY", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF"
        }
    
    def _load_country_codes(self) -> set:
        """Load valid ISO country codes."""
        # Common country codes - in production, load from a comprehensive source
        return {
            "US", "GB", "DE", "FR", "JP", "CA", "AU", "CH", "NL", "SE",
            "CN", "HK", "SG", "KR", "IN", "BR", "MX", "RU", "ZA", "TR"
        }
    
    async def standardize_data(
        self,
        raw_data: Dict[str, Any],
        data_type: DataType,
        source_info: Optional[Dict[str, Any]] = None
    ) -> StandardizedData:
        """Standardize raw data according to type-specific rules.
        
        Args:
            raw_data: Raw data to standardize
            data_type: Type of data being standardized
            source_info: Information about data source
        
        Returns:
            Standardized data with validation results
        """
        data_id = str(uuid4())
        
        # Apply field mappings
        mapped_data = self._apply_field_mappings(raw_data, data_type)
        
        # Standardize field values
        standardized_data = self._standardize_field_values(mapped_data, data_type)
        
        # Validate standardized data
        validation_result = await self._validate_data(standardized_data, data_type)
        
        # Apply consistency checks
        self._apply_consistency_checks(standardized_data, data_type, validation_result)
        
        return StandardizedData(
            data_id=data_id,
            data_type=data_type,
            original_data=raw_data,
            standardized_data=standardized_data,
            validation_result=validation_result,
            standardization_timestamp=datetime.utcnow(),
            source_info=source_info or {}
        )
    
    def _apply_field_mappings(
        self, 
        raw_data: Dict[str, Any], 
        data_type: DataType
    ) -> Dict[str, Any]:
        """Apply field mappings to normalize field names."""
        if data_type not in self._field_mappings:
            return raw_data.copy()
        
        mappings = self._field_mappings[data_type]
        mapped_data = {}
        
        # Create reverse mapping for lookup
        reverse_mapping = {}
        for standard_field, variants in mappings.items():
            for variant in variants:
                reverse_mapping[variant.lower()] = standard_field
        
        # Apply mappings
        for key, value in raw_data.items():
            key_lower = key.lower()
            if key_lower in reverse_mapping:
                standard_key = reverse_mapping[key_lower]
                mapped_data[standard_key] = value
            else:
                # Keep unmapped fields as-is
                mapped_data[key] = value
        
        return mapped_data
    
    def _standardize_field_values(
        self, 
        mapped_data: Dict[str, Any], 
        data_type: DataType
    ) -> Dict[str, Any]:
        """Standardize field values according to type-specific rules."""
        standardized = {}
        
        for field_name, value in mapped_data.items():
            try:
                standardized_value = self._standardize_single_value(
                    field_name, value, data_type
                )
                standardized[field_name] = standardized_value
            except Exception as e:
                logger.warning(f"Failed to standardize field {field_name}: {e}")
                # Keep original value if standardization fails
                standardized[field_name] = value
        
        return standardized
    
    def _standardize_single_value(
        self, 
        field_name: str, 
        value: Any, 
        data_type: DataType
    ) -> Any:
        """Standardize a single field value."""
        if value is None:
            return None
        
        # Get validation rules for this field
        rules = self._validation_rules.get(data_type, {}).get(field_name, {})
        field_type = rules.get("type", "string")
        
        # Apply type-specific standardization
        if field_type == "numeric":
            return self._standardize_numeric(value)
        elif field_type == "integer":
            return self._standardize_integer(value)
        elif field_type == "string":
            return self._standardize_string(value)
        elif field_type == "date":
            return self._standardize_date(value)
        elif field_type == "url":
            return self._standardize_url(value)
        elif field_type == "currency_code":
            return self._standardize_currency_code(value)
        elif field_type == "country_code":
            return self._standardize_country_code(value)
        else:
            return value
    
    def _standardize_numeric(self, value: Any) -> Optional[float]:
        """Standardize numeric values."""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common formatting
            cleaned = re.sub(r'[,$%\s]', '', value.strip())
            
            # Handle percentage
            if '%' in str(value):
                cleaned = cleaned.replace('%', '')
                try:
                    return float(cleaned) / 100.0
                except ValueError:
                    pass
            
            # Handle currency symbols
            cleaned = re.sub(r'[^\d.-]', '', cleaned)
            
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def _standardize_integer(self, value: Any) -> Optional[int]:
        """Standardize integer values."""
        if isinstance(value, int):
            return value
        
        if isinstance(value, float):
            return int(value)
        
        if isinstance(value, str):
            # Remove formatting
            cleaned = re.sub(r'[,\s]', '', value.strip())
            try:
                return int(float(cleaned))
            except ValueError:
                return None
        
        return None
    
    def _standardize_string(self, value: Any) -> Optional[str]:
        """Standardize string values."""
        if isinstance(value, str):
            # Basic string cleaning
            return value.strip()
        
        if value is not None:
            return str(value).strip()
        
        return None
    
    def _standardize_date(self, value: Any) -> Optional[datetime]:
        """Standardize date values."""
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        
        if isinstance(value, str):
            # Try common date formats
            date_formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(value.strip(), fmt)
                except ValueError:
                    continue
        
        return None
    
    def _standardize_url(self, value: Any) -> Optional[str]:
        """Standardize URL values."""
        if not isinstance(value, str):
            return None
        
        url = value.strip()
        
        # Add protocol if missing
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        return url
    
    def _standardize_currency_code(self, value: Any) -> Optional[str]:
        """Standardize currency codes."""
        if not isinstance(value, str):
            return None
        
        code = value.strip().upper()
        return code if code in self._currency_codes else None
    
    def _standardize_country_code(self, value: Any) -> Optional[str]:
        """Standardize country codes."""
        if not isinstance(value, str):
            return None
        
        code = value.strip().upper()
        return code if code in self._country_codes else None
    
    async def _validate_data(
        self, 
        data: Dict[str, Any], 
        data_type: DataType
    ) -> ValidationResult:
        """Validate standardized data against rules."""
        result = ValidationResult(is_valid=True, quality_score=1.0)
        
        if data_type not in self._validation_rules:
            result.metadata["validation_note"] = f"No validation rules for {data_type}"
            return result
        
        rules = self._validation_rules[data_type]
        
        # Check each field
        for field_name, field_rules in rules.items():
            value = data.get(field_name)
            
            # Check required fields
            if field_rules.get("required", False) and value is None:
                result.add_issue(ValidationIssue(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field_name}' is missing",
                    suggestion=f"Provide a value for {field_name}"
                ))
                continue
            
            if value is None:
                continue  # Skip validation for optional null values
            
            # Type validation
            await self._validate_field_type(field_name, value, field_rules, result)
            
            # Range validation
            self._validate_field_range(field_name, value, field_rules, result)
            
            # Pattern validation
            self._validate_field_pattern(field_name, value, field_rules, result)
            
            # Length validation
            self._validate_field_length(field_name, value, field_rules, result)
        
        return result
    
    async def _validate_field_type(
        self, 
        field_name: str, 
        value: Any, 
        rules: Dict[str, Any], 
        result: ValidationResult
    ):
        """Validate field type."""
        expected_type = rules.get("type")
        
        if expected_type == "numeric" and not isinstance(value, (int, float)):
            result.add_issue(ValidationIssue(
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"Field '{field_name}' should be numeric",
                expected_value="numeric",
                actual_value=type(value).__name__
            ))
        
        elif expected_type == "integer" and not isinstance(value, int):
            result.add_issue(ValidationIssue(
                field_name=field_name,
                severity=ValidationSeverity.ERROR,
                message=f"Field '{field_name}' should be integer",
                expected_value="integer",
                actual_value=type(value).__name__
            ))
        
        elif expected_type == "string" and not isinstance(value, str):
            result.add_issue(ValidationIssue(
                field_name=field_name,
                severity=ValidationSeverity.WARNING,
                message=f"Field '{field_name}' should be string",
                expected_value="string",
                actual_value=type(value).__name__
            ))
    
    def _validate_field_range(
        self, 
        field_name: str, 
        value: Any, 
        rules: Dict[str, Any], 
        result: ValidationResult
    ):
        """Validate field range constraints."""
        if isinstance(value, (int, float)):
            min_val = rules.get("min")
            max_val = rules.get("max")
            
            if min_val is not None and value < min_val:
                result.add_issue(ValidationIssue(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' value {value} is below minimum {min_val}",
                    expected_value=f">= {min_val}",
                    actual_value=value
                ))
            
            if max_val is not None and value > max_val:
                result.add_issue(ValidationIssue(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' value {value} is above maximum {max_val}",
                    expected_value=f"<= {max_val}",
                    actual_value=value
                ))
    
    def _validate_field_pattern(
        self, 
        field_name: str, 
        value: Any, 
        rules: Dict[str, Any], 
        result: ValidationResult
    ):
        """Validate field pattern constraints."""
        pattern = rules.get("pattern")
        
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                result.add_issue(ValidationIssue(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' does not match required pattern",
                    expected_value=f"Pattern: {pattern}",
                    actual_value=value
                ))
    
    def _validate_field_length(
        self, 
        field_name: str, 
        value: Any, 
        rules: Dict[str, Any], 
        result: ValidationResult
    ):
        """Validate field length constraints."""
        if isinstance(value, str):
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")
            
            if min_length is not None and len(value) < min_length:
                result.add_issue(ValidationIssue(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' is too short (minimum {min_length} characters)",
                    expected_value=f">= {min_length} characters",
                    actual_value=f"{len(value)} characters"
                ))
            
            if max_length is not None and len(value) > max_length:
                result.add_issue(ValidationIssue(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' is too long (maximum {max_length} characters)",
                    expected_value=f"<= {max_length} characters",
                    actual_value=f"{len(value)} characters"
                ))
    
    def _apply_consistency_checks(
        self, 
        data: Dict[str, Any], 
        data_type: DataType, 
        result: ValidationResult
    ):
        """Apply data consistency checks."""
        if data_type == DataType.FINANCIAL:
            self._check_financial_consistency(data, result)
        elif data_type == DataType.COMPANY:
            self._check_company_consistency(data, result)
    
    def _check_financial_consistency(
        self, 
        data: Dict[str, Any], 
        result: ValidationResult
    ):
        """Check financial data consistency."""
        # Assets = Liabilities + Equity (basic accounting equation)
        assets = data.get("assets")
        liabilities = data.get("liabilities")
        equity = data.get("equity")
        
        if all(x is not None for x in [assets, liabilities, equity]):
            expected_assets = liabilities + equity
            tolerance = abs(assets) * 0.01  # 1% tolerance
            
            if abs(assets - expected_assets) > tolerance:
                result.add_issue(ValidationIssue(
                    field_name="assets",
                    severity=ValidationSeverity.WARNING,
                    message="Assets do not equal Liabilities + Equity",
                    expected_value=expected_assets,
                    actual_value=assets,
                    suggestion="Check accounting equation: Assets = Liabilities + Equity"
                ))
        
        # PE ratio consistency
        price = data.get("price")
        eps = data.get("eps")
        pe_ratio = data.get("pe_ratio")
        
        if all(x is not None for x in [price, eps, pe_ratio]) and eps != 0:
            expected_pe = price / eps
            tolerance = abs(expected_pe) * 0.05  # 5% tolerance
            
            if abs(pe_ratio - expected_pe) > tolerance:
                result.add_issue(ValidationIssue(
                    field_name="pe_ratio",
                    severity=ValidationSeverity.WARNING,
                    message="PE ratio inconsistent with price and EPS",
                    expected_value=expected_pe,
                    actual_value=pe_ratio,
                    suggestion="Verify PE ratio calculation: Price / EPS"
                ))
    
    def _check_company_consistency(
        self, 
        data: Dict[str, Any], 
        result: ValidationResult
    ):
        """Check company data consistency."""
        # Founded date should not be in the future
        founded = data.get("founded")
        if isinstance(founded, datetime) and founded > datetime.utcnow():
            result.add_issue(ValidationIssue(
                field_name="founded",
                severity=ValidationSeverity.ERROR,
                message="Founded date cannot be in the future",
                actual_value=founded.isoformat(),
                suggestion="Check the founded date"
            ))
        
        # Employee count should be reasonable
        employees = data.get("employees")
        if isinstance(employees, int) and employees > 10000000:  # 10 million
            result.add_issue(ValidationIssue(
                field_name="employees",
                severity=ValidationSeverity.WARNING,
                message="Employee count seems unusually high",
                actual_value=employees,
                suggestion="Verify employee count"
            ))
    
    async def validate_data_consistency(
        self, 
        datasets: List[StandardizedData]
    ) -> Dict[str, Any]:
        """Validate consistency across multiple datasets.
        
        Args:
            datasets: List of standardized datasets to check for consistency
        
        Returns:
            Consistency validation report
        """
        consistency_report = {
            "total_datasets": len(datasets),
            "consistency_issues": [],
            "cross_references": {},
            "quality_summary": {}
        }
        
        if len(datasets) < 2:
            consistency_report["note"] = "Need at least 2 datasets for consistency checking"
            return consistency_report
        
        # Group datasets by type
        datasets_by_type = {}
        for dataset in datasets:
            data_type = dataset.data_type
            if data_type not in datasets_by_type:
                datasets_by_type[data_type] = []
            datasets_by_type[data_type].append(dataset)
        
        # Check consistency within each type
        for data_type, type_datasets in datasets_by_type.items():
            if len(type_datasets) > 1:
                type_issues = self._check_cross_dataset_consistency(type_datasets, data_type)
                consistency_report["consistency_issues"].extend(type_issues)
        
        # Calculate quality summary
        quality_scores = [ds.validation_result.quality_score for ds in datasets]
        consistency_report["quality_summary"] = {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "datasets_with_issues": len([ds for ds in datasets if not ds.validation_result.is_valid])
        }
        
        return consistency_report
    
    def _check_cross_dataset_consistency(
        self, 
        datasets: List[StandardizedData], 
        data_type: DataType
    ) -> List[Dict[str, Any]]:
        """Check consistency across datasets of the same type."""
        issues = []
        
        if data_type == DataType.FINANCIAL:
            # Check for consistent symbols/identifiers
            symbols = []
            for dataset in datasets:
                symbol = dataset.standardized_data.get("symbol")
                if symbol:
                    symbols.append((dataset.data_id, symbol))
            
            # Look for potential duplicates or conflicts
            symbol_groups = {}
            for data_id, symbol in symbols:
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(data_id)
            
            for symbol, data_ids in symbol_groups.items():
                if len(data_ids) > 1:
                    # Check if the data is actually consistent
                    prices = []
                    for data_id in data_ids:
                        dataset = next(ds for ds in datasets if ds.data_id == data_id)
                        price = dataset.standardized_data.get("price")
                        if price is not None:
                            prices.append(price)
                    
                    if len(prices) > 1:
                        price_variance = max(prices) - min(prices)
                        avg_price = sum(prices) / len(prices)
                        
                        if price_variance > avg_price * 0.05:  # 5% variance threshold
                            issues.append({
                                "type": "price_inconsistency",
                                "symbol": symbol,
                                "datasets": data_ids,
                                "price_range": f"{min(prices):.2f} - {max(prices):.2f}",
                                "variance_pct": (price_variance / avg_price) * 100
                            })
        
        return issues
    
    def get_standardization_stats(self) -> Dict[str, Any]:
        """Get statistics about standardization rules and mappings."""
        return {
            "supported_data_types": [dt.value for dt in DataType],
            "field_mappings_count": {
                dt.value: len(mappings) 
                for dt, mappings in self._field_mappings.items()
            },
            "validation_rules_count": {
                dt.value: len(rules) 
                for dt, rules in self._validation_rules.items()
            },
            "currency_codes_count": len(self._currency_codes),
            "country_codes_count": len(self._country_codes)
        }