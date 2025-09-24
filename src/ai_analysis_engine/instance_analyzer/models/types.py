"""
Type definitions for analysis results and related data structures
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class HypothesisType(str, Enum):
    """Types of hypotheses that can be generated"""
    SPEC_INCONSISTENCY = "spec_inconsistency"
    SPECIFICATION_INCONSISTENCY = "specification_inconsistency"
    PARAMETER_INAPPROPRIATE = "parameter_inappropriate"
    UNEXPECTED_BEHAVIOR = "unexpected_behavior"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    ALGORITHM_BUG = "algorithm_bug"


class VerificationStatus(str, Enum):
    """Status of hypothesis verification"""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class Hypothesis(BaseModel):
    """A hypothesis about the cause of an issue"""
    id: str = Field(..., description="Unique identifier")
    type: HypothesisType = Field(..., description="Type of hypothesis")
    category: str = Field(default="unknown", description="Category: data, algorithm, or context")
    description: str = Field(..., description="Detailed description")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    spec_reference: Optional[str] = Field(default=None, description="Reference to algorithm specification")
    analysis_step: Optional[str] = Field(default=None, description="Related analysis step from detailed procedure")
    verification_status: VerificationStatus = Field(default=VerificationStatus.PENDING)
    verification_result: Optional[str] = Field(default=None, description="Result of verification")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix or mitigation")
    expected_impact: Optional[str] = Field(default=None, description="Expected impact of the fix")


class VerificationResult(BaseModel):
    """Result of verifying a hypothesis"""
    hypothesis_id: str = Field(..., description="ID of the hypothesis being verified")
    success: bool = Field(..., description="Whether verification was successful")
    result_details: str = Field(..., description="Detailed results of verification")
    code_executed: Optional[str] = Field(default=None, description="Code that was executed")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Output data from execution")
    error_message: Optional[str] = Field(default=None, description="Error message if verification failed")
    execution_time: Optional[float] = Field(default=None, description="Time taken for execution")


class AnalysisResult(BaseModel):
    """Complete analysis result for a dataset"""
    dataset_id: str = Field(..., description="ID of the analyzed dataset")
    summary: str = Field(..., description="Brief summary of analysis")
    data_quality_score: float = Field(..., description="Data quality score (0-1)")
    consistency_score: float = Field(..., description="Consistency score (0-1)")
    hypotheses: List[Hypothesis] = Field(default_factory=list, description="Generated hypotheses")
    verified_issues: List[str] = Field(default_factory=list, description="Verified issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    generated_report: Optional[str] = Field(default=None, description="Generated report content")
    processing_time: Optional[float] = Field(default=None, description="Total processing time")
    error_messages: List[str] = Field(default_factory=list, description="Any error messages")


class DataSummary(BaseModel):
    """Summary of dataset characteristics"""
    row_count: int = Field(..., description="Number of rows in dataset")
    column_count: int = Field(..., description="Number of columns")
    columns: List[str] = Field(default_factory=list, description="Column names")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Data types for each column")
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Missing value counts")
    basic_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Basic statistics")
    time_range: Optional[Dict[str, str]] = Field(default=None, description="Time range if applicable")


class ConsistencyCheckResult(BaseModel):
    """Result of consistency checking"""
    is_consistent: bool = Field(..., description="Overall consistency result")
    issues_found: List[str] = Field(default_factory=list, description="List of consistency issues")
    expected_pattern: str = Field(..., description="Expected pattern from natural language")
    actual_pattern: str = Field(..., description="Actual pattern found in data")
    matched_records: int = Field(default=0, description="Number of records matching expectation")
    total_records: int = Field(default=0, description="Total number of records checked")
    confidence_score: float = Field(default=0.0, description="Confidence in the consistency check")
