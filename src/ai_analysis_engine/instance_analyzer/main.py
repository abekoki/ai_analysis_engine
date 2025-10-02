"""
Main application entry point for AI Analysis Engine
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .config import config
from .models.state import AnalysisState, DatasetInfo
from .core.graph import AnalysisGraph
from .utils.logger import get_logger, setup_logging
from .utils.file_utils import ensure_analysis_output_structure, write_file, get_report_paths

logger = get_logger(__name__)


class AIAnalysisEngine:
    """
    Main application class for the AI Analysis Engine
    """

    def __init__(self):
        self.graph = None
        self.logger = logger

    def initialize(self) -> bool:
        """
        Initialize the analysis engine

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing AI Analysis Engine...")

            # Ensure directories exist
            config.ensure_directories()

            # Validate API keys
            if not config.validate_api_keys():
                self.logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return False

            # Build the analysis graph
            self.graph = AnalysisGraph()
            self.graph.build_graph()

            self.logger.info("AI Analysis Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Analysis Engine: {e}")
            return False

    def create_analysis_request(self,
                              algorithm_outputs: List[str],
                              core_outputs: List[str],
                              algorithm_specs: List[str],
                              evaluation_specs: List[str],
                              expected_results: List[str],
                              algorithm_codes: Optional[List[List[str]]] = None,
                              evaluation_codes: Optional[List[List[str]]] = None,
                              evaluation_intervals: Optional[List[Optional[Dict[str, Any]]]] = None,
                              dataset_ids: Optional[List[str]] = None,
                              output_dir: Optional[str] = None) -> AnalysisState:
        """
        Create an analysis request from input files

        Args:
            algorithm_outputs: List of algorithm output CSV files
            core_outputs: List of core library output CSV files
            algorithm_specs: List of algorithm specification Markdown files
            evaluation_specs: List of evaluation specification Markdown files
            expected_results: List of expected results (natural language)
            algorithm_codes: Optional list of lists containing algorithm implementation code files
            evaluation_codes: Optional list of lists containing evaluation environment code files
            dataset_ids: Optional list of dataset IDs
            output_dir: Optional custom output directory for results

        Returns:
            AnalysisState object ready for processing
        """
        if len(algorithm_outputs) != len(core_outputs) or len(algorithm_outputs) != len(expected_results):
            raise ValueError("All input lists must have the same length")

        datasets = []

        for i, (algo_csv, core_csv, expected) in enumerate(zip(algorithm_outputs, core_outputs, expected_results)):
            dataset_id = dataset_ids[i] if dataset_ids else f"dataset_{i+1}"

            # Find corresponding specs and codes
            algo_spec = algorithm_specs[i] if i < len(algorithm_specs) else (algorithm_specs[0] if algorithm_specs else None)
            eval_spec = evaluation_specs[i] if i < len(evaluation_specs) else (evaluation_specs[0] if evaluation_specs else None)
            algo_codes = algorithm_codes[i] if algorithm_codes is not None and i < len(algorithm_codes) else []
            eval_codes = evaluation_codes[i] if evaluation_codes is not None and i < len(evaluation_codes) else []

            # evaluation_intervals から対応する区間情報を取得
            interval_info = evaluation_intervals[i] if evaluation_intervals and i < len(evaluation_intervals) else None

            dataset = DatasetInfo(
                id=dataset_id,
                algorithm_output_csv=algo_csv,
                core_output_csv=core_csv,
                algorithm_spec_md=algo_spec,
                algorithm_code_files=algo_codes,
                evaluation_spec_md=eval_spec,
                evaluation_code_files=eval_codes,
                expected_result=expected,
                evaluation_interval=interval_info
            )

            datasets.append(dataset)

        # Collect all spec and code documents
        all_spec_docs = []
        all_code_docs = []

        if algorithm_specs:
            all_spec_docs.extend(algorithm_specs)
        if evaluation_specs:
            all_spec_docs.extend(evaluation_specs)

        if algorithm_codes is not None:
            for code_list in algorithm_codes:
                all_code_docs.extend(code_list)
        if evaluation_codes is not None:
            for code_list in evaluation_codes:
                all_code_docs.extend(code_list)

        # Create initial state
        state = AnalysisState(
            datasets=datasets,
            spec_documents=all_spec_docs,
            code_documents=all_code_docs,
            start_time=self._get_current_time(),
            output_dir=output_dir
        )

        self.logger.info(f"Created analysis request with {len(datasets)} datasets")
        return state

    def run_analysis(self, state: AnalysisState, *, max_instances: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete analysis workflow

        Args:
            state: Analysis state to process

        Returns:
            Analysis results
        """
        if not self.graph:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        try:
            self.logger.info("Starting analysis workflow...")

            state_dict = state.model_dump()
            if max_instances is not None:
                state_dict["max_instances"] = max_instances

            # Run the analysis
            result = self.graph.run_analysis(state_dict)

            # Update end time
            result["end_time"] = self._get_current_time()

            # Save results
            self._save_results(result, getattr(state, 'output_dir', None))

            self.logger.info("Analysis workflow completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "end_time": self._get_current_time()
            }

    def _save_results(self, results: Dict[str, Any], custom_output_dir: Optional[str] = None) -> None:
        """Save analysis results to files"""
        base_dir = Path(custom_output_dir) if custom_output_dir else Path(config.output_dir)
        # Save individual reports first (independent of JSON saving)
        try:
            if "datasets" in results:
                for dataset in results["datasets"]:
                    if hasattr(dataset, 'report_content') and dataset.report_content and hasattr(dataset, 'id'):
                        report_paths = get_report_paths(base_dir, f"report_{dataset.id}")
                        report_file = report_paths["report_dir"] / f"report_{dataset.id}.md"
                        write_file(str(report_file), dataset.report_content)
                        self.logger.info(f"Report saved: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save reports: {e}")

        try:
            per_dataset_results: List[Dict[str, Any]] = []
            if "datasets" in results:
                for dataset in results["datasets"]:
                    dataset_data = dataset.model_dump() if hasattr(dataset, "model_dump") else dataset
                    dataset_id = dataset_data.get("id") or dataset_data.get("dataset_id")
                    if not dataset_id:
                        continue

                    report_paths = get_report_paths(base_dir, f"report_{dataset_id}")

                    if dataset_data.get("report_content"):
                        report_file = report_paths["report_dir"] / f"report_{dataset_id}.md"
                        write_file(str(report_file), dataset_data["report_content"])
                        self.logger.info(f"Report saved: {report_file}")

                    dataset_json_file = report_paths["report_dir"] / "analysis_results.json"
                    dataset_serializable = self._make_serializable(dataset_data)
                    with open(dataset_json_file, "w", encoding="utf-8") as f:
                        json.dump(dataset_serializable, f, indent=2, ensure_ascii=False, default=str)
                    self.logger.info(f"Dataset JSON saved: {dataset_json_file}")

                    # Save LangGraph context details if present
                    metadata = dataset_data.get("metadata") if isinstance(dataset_data, dict) else None
                    langgraph_blob = metadata.get("langgraph_memory") if metadata else None
                    if isinstance(langgraph_blob, dict):
                        events = langgraph_blob.get("events")
                        checkpoint = langgraph_blob.get("checkpointer")
                        events_file = report_paths["contexts"] / f"langgraph_events_{dataset_id}.json"
                        checkpoint_file = report_paths["contexts"] / f"langgraph_checkpointer_{dataset_id}.json"
                        try:
                            if events is not None:
                                with open(events_file, "w", encoding="utf-8") as ef:
                                    json.dump(events, ef, indent=2, ensure_ascii=False, default=str)
                                self.logger.info(f"LangGraph events saved: {events_file}")
                            if checkpoint:
                                with open(checkpoint_file, "w", encoding="utf-8") as cf:
                                    json.dump(checkpoint, cf, indent=2, ensure_ascii=False, default=str)
                                self.logger.info(f"LangGraph checkpointer saved: {checkpoint_file}")
                        except Exception as ctx_error:
                            self.logger.warning(f"Failed to save LangGraph context for {dataset_id}: {ctx_error}")

                    per_dataset_results.append(dataset_serializable)
            else:
                self.logger.warning("No dataset results found to save per dataset")
        except Exception as e:
            self.logger.error(f"Failed to save per-dataset results: {e}")

        try:
            serializable_results = self._make_serializable(results)
            global_langgraph = results.get("langgraph_memory")
            if isinstance(global_langgraph, dict):
                serializable_results["langgraph_memory"] = {
                    "events": global_langgraph.get("events"),
                    "checkpointer": global_langgraph.get("checkpointer"),
                }

            def check_serializable(obj, path="root"):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        check_serializable(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_serializable(item, f"{path}[{i}]")
                else:
                    try:
                        json.dumps(obj)
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Non-serializable object at {path}: {type(obj)} - {e}")

            check_serializable(serializable_results)

            results_file = base_dir / "summary" / "analysis_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"JSON results saved to {results_file}")

        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            try:
                error_file = base_dir / "summary" / "error_summary.json"
                error_summary = {
                    "error": str(e),
                    "timestamp": self._get_current_time(),
                    "datasets_count": len(results.get("datasets", []))
                }
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_summary, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Error summary saved to {error_file}")
            except Exception as e2:
                self.logger.error(f"Failed to save error summary: {e2}")

        self.logger.info(f"Results processing completed in {base_dir}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'model_dump'):  # Pydantic models
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):  # Regular objects
            return {key: self._make_serializable(value) for key, value in obj.__dict__.items() if not key.startswith('_')}
        elif hasattr(obj, 'dtype'):  # pandas/numpy dtypes
            if hasattr(obj.dtype, 'name'):
                return obj.dtype.name
            else:
                return str(obj.dtype)
        elif hasattr(obj, 'to_dict'):  # pandas objects
            try:
                return obj.to_dict()
            except:
                return str(obj)
        else:
            # Try to convert to basic types
            try:
                import json
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # Handle special pandas/numpy types
                type_str = str(type(obj))
                if 'DType' in type_str or 'dtype' in type_str.lower():
                    # Handle pandas DType objects more specifically
                    if hasattr(obj, 'name'):
                        return obj.name
                    elif hasattr(obj, 'type'):
                        return str(obj.type)
                    elif hasattr(obj, 'kind'):
                        return obj.kind
                    else:
                        # For complex DType objects, try to get a simple representation
                        try:
                            return str(obj).split('(')[0]  # Get just the type name
                        except:
                            return f"<{type(obj).__name__}>"
                elif 'numpy' in type_str:
                    # Handle numpy types
                    if hasattr(obj, 'item'):
                        try:
                            return obj.item()
                        except:
                            return str(obj)
                    elif hasattr(obj, 'tolist'):
                        try:
                            return obj.tolist()
                        except:
                            return str(obj)
                    else:
                        return str(obj)
                elif hasattr(obj, '__class__'):
                    # Handle other complex objects
                    try:
                        return str(obj)
                    except:
                        return f"<{obj.__class__.__name__} object>"
                else:
                    # Fallback for unknown objects
                    return str(obj)

    def _get_current_time(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "initialized": self.graph is not None,
            "config_valid": config.validate_api_keys(),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "logs_dir": str(config.logs_dir)
        }


def main():
    """Main entry point for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Analysis Engine for Time Series Data")
    parser.add_argument("--algorithm-outputs", nargs="+", required=True,
                       help="Algorithm output CSV files")
    parser.add_argument("--core-outputs", nargs="+", required=True,
                       help="Core library output CSV files")
    parser.add_argument("--algorithm-specs", nargs="+", required=True,
                       help="Algorithm specification Markdown files")
    parser.add_argument("--algorithm-codes", nargs="*", action="append",
                       help="Algorithm implementation code files (can be specified multiple times)")
    parser.add_argument("--evaluation-specs", nargs="+", required=True,
                       help="Evaluation specification Markdown files")
    parser.add_argument("--evaluation-codes", nargs="*", action="append",
                       help="Evaluation environment code files (can be specified multiple times)")
    parser.add_argument("--expected-results", nargs="+", required=True,
                       help="Expected results (natural language)")
    parser.add_argument("--dataset-ids", nargs="+",
                       help="Dataset IDs (optional)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results (default: ./output)")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Initialize engine
    engine = AIAnalysisEngine()

    if not engine.initialize():
        logger.error("Failed to initialize engine")
        return 1

    try:
        # Process code arguments (convert from list of lists to proper format)
        algorithm_codes = None
        if args.algorithm_codes:
            algorithm_codes = [code_list for code_list in args.algorithm_codes if code_list]

        evaluation_codes = None
        if args.evaluation_codes:
            evaluation_codes = [code_list for code_list in args.evaluation_codes if code_list]

        # Create analysis request
        state = engine.create_analysis_request(
            algorithm_outputs=args.algorithm_outputs,
            core_outputs=args.core_outputs,
            algorithm_specs=args.algorithm_specs,
            evaluation_specs=args.evaluation_specs,
            expected_results=args.expected_results,
            algorithm_codes=algorithm_codes,
            evaluation_codes=evaluation_codes,
            dataset_ids=args.dataset_ids,
            output_dir=args.output_dir
        )

        # Run analysis
        results = engine.run_analysis(state)

        if "error" in results:
            logger.error(f"Analysis failed: {results['error']}")
            return 1
        else:
            logger.info("Analysis completed successfully")
            return 0

    except Exception as e:
        logger.error(f"Analysis execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
