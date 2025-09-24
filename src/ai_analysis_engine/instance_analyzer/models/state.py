"""
State models for LangGraph workflow
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path


class DatasetInfo(BaseModel):
    """Information about a single dataset"""
    id: str = Field(..., description="Unique identifier for the dataset")
    algorithm_output_csv: str = Field(..., description="Path to algorithm output CSV")
    core_output_csv: str = Field(..., description="Path to core library output CSV")
    algorithm_spec_md: str = Field(..., description="Path to algorithm specification Markdown")
    algorithm_code_files: List[str] = Field(default_factory=list, description="Paths to algorithm implementation code files")
    evaluation_spec_md: str = Field(..., description="Path to evaluation specification Markdown")
    evaluation_code_files: List[str] = Field(default_factory=list, description="Paths to evaluation environment code files")
    expected_result: str = Field(..., description="Natural language description of expected results")

    # Processing state
    data_summary: Optional[Dict[str, Any]] = Field(default=None, description="Data summary from DataChecker")
    consistency_check: Optional[Dict[str, Any]] = Field(default=None, description="Consistency check results")
    hypotheses: Optional[List[Dict[str, Any]]] = Field(default=None, description="Generated hypotheses")
    verification_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Verification results")
    report_content: Optional[str] = Field(default=None, description="Generated report content")

    # Status tracking
    status: str = Field(default="pending", description="Current processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class VectorStoreInfo(BaseModel):
    """Information about vector stores"""
    segments: Dict[str, Any] = Field(default_factory=dict, description="Segment name to path mapping")
    last_updated: Optional[str] = Field(default=None, description="Last update timestamp")
    is_initialized: bool = Field(default=False, description="Whether vector stores are initialized")


class AnalysisState(BaseModel):
    """Main state for the analysis workflow"""
    # Input data
    datasets: List[DatasetInfo] = Field(default_factory=list, description="List of datasets to analyze")
    spec_documents: List[str] = Field(default_factory=list, description="Paths to specification documents")
    code_documents: List[str] = Field(default_factory=list, description="Paths to code documents")

    # Processing state
    current_dataset_index: int = Field(default=0, description="Index of currently processing dataset")
    vector_stores: VectorStoreInfo = Field(default_factory=VectorStoreInfo, description="Vector store information")

    # Workflow control
    workflow_step: str = Field(default="initialization", description="Current workflow step")
    next_action: Optional[str] = Field(default=None, description="Next action to take")

    # Results
    analysis_results: Dict[str, Any] = Field(default_factory=dict, description="Analysis results for each dataset")
    hypotheses: Dict[str, Any] = Field(default_factory=dict, description="Generated hypotheses for each dataset")
    final_reports: List[str] = Field(default_factory=list, description="Generated report contents")

    # Error handling
    errors: List[str] = Field(default_factory=list, description="List of error messages")

    # Metadata
    start_time: Optional[str] = Field(default=None, description="Workflow start timestamp")
    end_time: Optional[str] = Field(default=None, description="Workflow end timestamp")
    output_dir: Optional[str] = Field(default=None, description="Custom output directory for results")

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'  # Allow extra fields to be set dynamically

    def get_current_dataset(self) -> Optional[DatasetInfo]:
        """Get the currently processing dataset"""
        if 0 <= self.current_dataset_index < len(self.datasets):
            return self.datasets[self.current_dataset_index]
        return None

    def advance_dataset(self) -> bool:
        """Advance to the next dataset. Returns True if there are more datasets."""
        self.current_dataset_index += 1
        return self.current_dataset_index < len(self.datasets)

    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)

    def is_complete(self) -> bool:
        """Check if all datasets have been processed"""
        return self.current_dataset_index >= len(self.datasets)

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing status"""
        total = len(self.datasets)
        completed = sum(1 for ds in self.datasets if ds.status == "completed")
        failed = sum(1 for ds in self.datasets if ds.status == "failed")

        return {
            "total_datasets": total,
            "completed": completed,
            "failed": failed,
            "in_progress": total - completed - failed,
            "current_index": self.current_dataset_index
        }
