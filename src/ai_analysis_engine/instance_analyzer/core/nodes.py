"""
LangGraph nodes for the AI Analysis Engine workflow
"""

from typing import Dict, Any, Iterable, List, Optional
import pandas as pd
from datetime import datetime

from ..models.state import AnalysisState, DatasetInfo
from ..models.types import DataSummary, ConsistencyCheckResult, Hypothesis, VerificationResult
from ..tools.rag_tool import RAGTool
from ..tools.repl_tool import REPLTool
from ..utils.logger import get_logger
from ..utils.exploration_utils import extract_frame_range_with_llm
from ..config import config
from ..utils.file_utils import (
    compute_representative_stats,
    filter_dataframe_by_interval,
    get_report_paths,
)

logger = get_logger(__name__)


class InitializationNode:
    """Node for initializing the RAG system"""

    def __init__(self):
        self.rag_tool = RAGTool()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Initialize vector stores and prepare for analysis"""
        logger.info("Starting initialization")

        try:
            # Prepare documents for RAG
            documents = self._prepare_documents(state)

            # Initialize vector stores
            success = self.rag_tool.initialize_vector_stores(documents)

            if success:
                # Update state
                state.vector_stores.is_initialized = True
                state.vector_stores.segments = {segment: f"vectorstore_{segment}" for segment in documents.keys()}
                state.vector_stores.last_updated = datetime.now().isoformat()
                try:
                    state.messages.append("Initialization: vector stores ready")
                except Exception:
                    pass

                logger.info("Initialization completed successfully")
            else:
                state.errors.append("Failed to initialize vector stores")
                logger.error("Initialization failed")

        except Exception as e:
            error_msg = f"Initialization error: {e}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        state.workflow_step = "supervisor"
        return state

    def _prepare_documents(self, state: AnalysisState) -> Dict[str, List[str]]:
        """Prepare documents for vectorization by segment"""
        documents = {
            "algorithm_specs": [],
            "evaluation_specs": [],
            "algorithm_code": [],
            "evaluation_code": []
        }

        # Add specification documents
        for spec_doc in state.spec_documents:
            if "algorithm" in spec_doc.lower():
                documents["algorithm_specs"].append(spec_doc)
            elif "evaluation" in spec_doc.lower():
                documents["evaluation_specs"].append(spec_doc)
            else:
                documents["algorithm_specs"].append(spec_doc)

        # Add code documents
        for code_doc in state.code_documents:
            if "algorithm" in code_doc.lower() or code_doc.endswith(('.py', '.cpp', '.java', '.js', '.ts')):
                documents["algorithm_code"].append(code_doc)
            else:
                documents["evaluation_code"].append(code_doc)

        # Add dataset-specific documents
        for dataset in state.datasets:
            if dataset.algorithm_spec_md:
                documents["algorithm_specs"].append(dataset.algorithm_spec_md)
            if dataset.evaluation_spec_md:
                documents["evaluation_specs"].append(dataset.evaluation_spec_md)

            # Add algorithm code files
            documents["algorithm_code"].extend(dataset.algorithm_code_files)

            # Add evaluation code files
            documents["evaluation_code"].extend(dataset.evaluation_code_files)

        return documents


class SupervisorNode:
    """Node for supervising the analysis workflow"""

    def process(self, state: AnalysisState) -> AnalysisState:
        """Supervise and route to appropriate next step"""
        logger.info("Supervisor processing")
        try:
            state.messages.append("Supervisor: routing decision")
        except Exception:
            pass

        current_dataset = state.get_current_dataset()

        if current_dataset is None:
            # All datasets processed
            state.workflow_step = "completed"
            logger.info("All datasets processed")
            return state

        # Determine next step based on dataset status
        if current_dataset.status == "failed":
            # If failed, advance to next dataset
            state.advance_dataset()
            if state.get_current_dataset():
                state.workflow_step = "data_checker"
            else:
                state.workflow_step = "completed"
        elif current_dataset.status == "pending":
            state.workflow_step = "data_checker"
        elif current_dataset.status == "data_checked":
            state.workflow_step = "consistency_checker"
        elif current_dataset.status == "consistency_checked":
            state.workflow_step = "hypothesis_generator"
        elif current_dataset.status == "hypothesis_generated":
            state.workflow_step = "verifier"
        elif current_dataset.status == "verified":
            state.workflow_step = "reporter"
        elif current_dataset.status == "completed":
            # Advance to next dataset
            state.advance_dataset()
            if state.get_current_dataset():
                state.workflow_step = "data_checker"
            else:
                state.workflow_step = "completed"
        else:
            state.workflow_step = "data_checker"

        logger.info(f"Next step: {state.workflow_step} for dataset {current_dataset.id}")
        return state


class DataCheckerNode:
    """Node for checking and analyzing input data"""

    def __init__(self):
        self.repl_tool = REPLTool()
        self.rag_tool = RAGTool()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Check and analyze data for current dataset"""
        logger.info("Data checker processing")
        try:
            state.messages.append("DataChecker: start")
        except Exception:
            pass

        current_dataset = state.get_current_dataset()
        if not current_dataset:
            return state

        try:
            # Load CSV data
            csv_files = [
                current_dataset.algorithm_output_csv,
                current_dataset.core_output_csv
            ]

            dataframes = self.repl_tool.load_csv_data(csv_files)
            logger.info(f"Loaded {len(dataframes)} dataframes: {list(dataframes.keys())}")

            interval: Dict[str, Any] = {}
            if getattr(current_dataset, "evaluation_interval", None):
                interval = current_dataset.evaluation_interval
            elif isinstance(current_dataset.data_summary, dict):
                interval = current_dataset.data_summary.get("evaluation_interval", {}) or {}

            filtered_dataframes: Dict[str, Any] = {}
            for name, df in dataframes.items():
                filtered_df = filter_dataframe_by_interval(df, interval)
                filtered_dataframes[name] = filtered_df if filtered_df is not None else df

            # Analyze data
            analysis_results = {}
            basic_analysis = {}
            representative_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
            for name, df in filtered_dataframes.items():
                try:
                    logger.info(f"Analyzing dataframe {name} with shape {df.shape}")
                    analysis = self.repl_tool.analyze_dataframe(df, name)
                    analysis_results[name] = analysis
                    # Normalize: expose summary under basic_analysis for downstream agents
                    basic_analysis[name] = {
                        "shape": analysis.get("shape"),
                        "columns": analysis.get("columns"),
                        "dtypes": analysis.get("dtypes", {}),
                        "missing_values": analysis.get("missing_values", {}),
                        "basic_stats": analysis.get("basic_stats", {}),
                    }
                    representative_stats[name] = compute_representative_stats(df)
                    logger.info(f"Computed representative stats for {name}: {list(representative_stats[name].keys())}")
                except Exception as e:
                    logger.error(f"Failed to analyze dataframe {name}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    analysis_results[name] = {"error": str(e)}

            # Create plots
            logger.info("Creating data plots")
            plot_paths = self._create_data_plots(filtered_dataframes, current_dataset.id, state)
            logger.info(f"Created plot paths: {plot_paths}")

            # Query column information from specs
            logger.info("Getting column information from specs")
            try:
                column_info = self._get_column_info_from_specs(current_dataset)
                logger.info(f"Column info keys: {list(column_info.keys()) if column_info else 'None'}")
            except Exception as e:
                logger.error(f"Failed to get column info: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                column_info = {}

            # Ensure spec texts via RAG cache for downstream agents (domain-agnostic)
            try:
                if not getattr(current_dataset, 'algorithm_spec_text', None):
                    hits = self.rag_tool.search("algorithm specification core sections", "algorithm_specs", k=3)
                    current_dataset.algorithm_spec_text = "\n\n".join([r.get('content', '') for r in (hits or [])])[:3000] or None
                if not getattr(current_dataset, 'evaluation_spec_text', None):
                    hits = self.rag_tool.search("evaluation specification core sections", "evaluation_specs", k=3)
                    current_dataset.evaluation_spec_text = "\n\n".join([r.get('content', '') for r in (hits or [])])[:3000] or None
            except Exception as _ex:
                logger.warning(f"Failed to populate spec text cache: {_ex}")

            # Extract column mapping from specs using LLM
            logger.info("Extracting column mapping from specs")
            try:
                column_mapping_result = self._extract_columns_from_specs(state)
                logger.info(f"Column mapping keys: {list(column_mapping_result.keys()) if column_mapping_result else 'None'}")
            except Exception as e:
                logger.error(f"Failed to extract column mapping: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                column_mapping_result = {}

            # Update dataset state
            current_dataset.data_summary = {
                "analysis": analysis_results,
                "basic_analysis": basic_analysis,
                "plots": plot_paths,
                "column_info": column_info,
                "column_mapping": column_mapping_result,
                "representative_stats": representative_stats,
                "evaluation_interval": interval,
            }
            current_dataset.status = "data_checked"

            # Store analysis results in state for later use
            logger.info("Storing analysis results in state")
            try:
                if not hasattr(state, 'analysis_results'):
                    state.analysis_results = {}
                state.analysis_results[current_dataset.id] = {
                    "analysis": analysis_results,
                    "basic_analysis": basic_analysis,
                    "plots": plot_paths,
                    "column_info": column_info,
                    "column_mapping": column_mapping_result,
                    "representative_stats": representative_stats,
                    "evaluation_interval": interval,
                }
                # Debug snapshot for step-by-step trace
                try:
                    dataset_paths = get_report_paths(config.output_dir, f"report_{current_dataset.id}")
                    logs_dir = dataset_paths["logs"]
                    import json
                    with open(logs_dir / "state_snapshot.json", 'w', encoding='utf-8') as f:
                        json.dump({
                            "messages": getattr(state, 'messages', []),
                            "workflow_step": state.workflow_step,
                            "current_dataset": current_dataset.model_dump() if hasattr(current_dataset, 'model_dump') else {},
                            "analysis_results": state.analysis_results.get(current_dataset.id, {}),
                        }, f, ensure_ascii=False, indent=2)
                except Exception as ex:
                    logger.warning(f"Failed to write state snapshot: {ex}")
                logger.info(f"Successfully stored results for dataset {current_dataset.id}")
            except Exception as e:
                logger.error(f"Failed to store analysis results: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

            logger.info(f"Data checking completed for dataset {current_dataset.id}")
            try:
                state.messages.append(f"DataChecker: completed for {current_dataset.id}")
            except Exception:
                pass

        except Exception as e:
            error_msg = f"Data checking failed: {e}"
            current_dataset.error_message = error_msg
            current_dataset.status = "failed"
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state

    def _create_data_plots(self, dataframes: Dict[str, Any], dataset_id: str, state: Any = None) -> Dict[str, str]:
        """Create time series plots for the data"""
        plot_paths = {}

        try:
            dataset_paths = get_report_paths(config.output_dir, f"report_{dataset_id}")
            plots_dir = dataset_paths["images"]

            interval: Dict[str, Any] = {}
            if state and hasattr(state, "current_dataset"):
                current_ds = state.get_current_dataset()
                if current_ds and getattr(current_ds, "evaluation_interval", None):
                    interval = current_ds.evaluation_interval

            for name, df in dataframes.items():
                if len(df) == 0:
                    continue

                filtered_df = filter_dataframe_by_interval(df, interval)
                if filtered_df.empty:
                    logger.warning("Filtered dataframe for %s is empty after applying evaluation interval; skipping plot.", name)
                    continue

                x_col = None
                if 'frame_num' in filtered_df.columns:
                    x_col = 'frame_num'
                elif 'frame' in filtered_df.columns:
                    x_col = 'frame'

                y_series = []
                y_label = 'Value'

                available_cols = set(filtered_df.columns)
                numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()

                if hasattr(state, 'algorithm_config') and state.algorithm_config:
                    config_obj = state.algorithm_config
                    if any(col in available_cols for col in config_obj.output_columns):
                        for col in config_obj.output_columns:
                            if col in available_cols:
                                if col in numeric_cols:
                                    label = col.replace('_', ' ').title()
                                    y_series.append((col, label))
                                    y_label = 'Algorithm Output'
                                elif df[col].dtype in ['object', 'category']:
                                    if df[col].str.isnumeric().all():
                                        df_copy = df.copy()
                                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                                        if not df_copy[col].isna().all():
                                            label = col.replace('_', ' ').title()
                                            y_series.append((col, label))
                                            y_label = 'Algorithm Output'
                    elif any(col in available_cols for col in config_obj.input_columns):
                        for col in config_obj.input_columns:
                            if col in available_cols and col in numeric_cols:
                                label = col.replace('_', ' ').title()
                                y_series.append((col, label))
                                y_label = 'Input Values'
                else:
                    numeric_candidates = df.select_dtypes(include=['number']).columns
                    if len(numeric_candidates) > 0:
                        primary_col = numeric_candidates[0]
                        if 'continuous_time' in numeric_candidates:
                            primary_col = 'continuous_time'
                        y_series.append((primary_col, str(primary_col)))

                if x_col is None and len(y_series) == 0:
                    continue

                series_code_lines = []
                for col, label in y_series:
                    series_code_lines.append(f"(df['{col}'].astype(int) if df['{col}'].dtype == 'bool' else df['{col}'])")
                series_code = "\n".join([f"plt.plot(df['{x_col}'] if '{x_col}' in df.columns else df.index, {line}, label='{label}')" for (line, (_, label)) in zip(series_code_lines, y_series)])

                code = f"""
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
{series_code}
plt.title(f'{name} Time Series')
plt.xlabel('{x_col if x_col else 'Index'}')
plt.ylabel('{y_label}')
plt.legend()
plt.tight_layout()
"""
                plot_path = plots_dir / f"{name}_timeseries.png"
                result = self.repl_tool.create_plot(code, {"df": df}, str(plot_path))

                if result.get("success"):
                    plot_paths[name] = plot_path.name

        except Exception as e:
            logger.error(f"Failed to create plots: {e}")

        return plot_paths

    def _get_column_info_from_specs(self, dataset: DatasetInfo) -> Dict[str, Any]:
        """Get column information from evaluation specifications"""
        try:
            logger.info("Searching for column information in specs")
            # Search for column-related information
            spec_results = self.rag_tool.search("column format data structure", "evaluation_specs", k=3)
            logger.info(f"Found {len(spec_results)} spec results")

            column_info = {
                "spec_results": spec_results,
                "inferred_columns": {}
            }

            return column_info

        except Exception as e:
            logger.error(f"Failed to get column info: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def _extract_columns_from_specs(self, state: AnalysisState) -> Dict[str, Any]:
        from ..utils.text_utils import extract_columns_with_llm, extract_thresholds_with_llm

        dataset = state.get_current_dataset()
        if not dataset:
            return {}

        # Use RAG tool to search for column information
        column_info = self._get_column_info_from_specs(dataset)

        # Extract columns and thresholds using LLM with spec content via RAG/cache (no direct file I/O)
        spec_content = ""
        try:
            algo_text = getattr(dataset, 'algorithm_spec_text', None) or ""
            eval_text = getattr(dataset, 'evaluation_spec_text', None) or ""
            if algo_text:
                spec_content += algo_text
            if eval_text:
                spec_content += ("\n" + eval_text)
            if not spec_content:
                algo_hits = self.rag_tool.search("algorithm specification core sections", "algorithm_specs", k=3)
                eval_hits = self.rag_tool.search("evaluation specification core sections", "evaluation_specs", k=3)
                parts = [r.get('content', '') for r in (algo_hits or [])] + [r.get('content', '') for r in (eval_hits or [])]
                spec_content = "\n\n".join([p for p in parts if p])
        except Exception:
            spec_content = spec_content or ""

        # Extract columns using LLM with fallback strategies
        input_columns = []
        output_columns = []

        if spec_content:
            try:
                # Extract input columns (features)
                input_query = "このアルゴリズムで使用される入力特徴量や列名を抽出してください。例: confidence, openness, scoreなど"
                rag_input_results = self.rag_tool.search(input_query, "algorithm_specs", k=3)
                if rag_input_results:
                    input_columns = extract_columns_with_llm(" ".join([r['content'] for r in rag_input_results]))
                else:
                    input_columns = extract_columns_with_llm(spec_content)

                # If LLM failed, try regex-based extraction
                if not input_columns:
                    input_columns = self._extract_columns_with_regex(spec_content, 'input')

            except Exception as e:
                logger.warning(f"LLM input column extraction failed: {e}")
                input_columns = self._extract_columns_with_regex(spec_content, 'input')

            try:
                # Extract output columns (predictions)
                output_query = "このアルゴリズムの出力列や予測結果の列名を抽出してください。例: is_drowsy, detection_resultなど"
                rag_output_results = self.rag_tool.search(output_query, "algorithm_specs", k=3)
                if rag_output_results:
                    output_columns = extract_columns_with_llm(" ".join([r['content'] for r in rag_output_results]))
                else:
                    output_columns = extract_columns_with_llm(spec_content)

                # If LLM failed, try regex-based extraction
                if not output_columns:
                    output_columns = self._extract_columns_with_regex(spec_content, 'output')

            except Exception as e:
                logger.warning(f"LLM output column extraction failed: {e}")
                output_columns = self._extract_columns_with_regex(spec_content, 'output')

        # Extract thresholds using RAG and LLM
        thresholds = {}
        if spec_content:
            threshold_query = "このアルゴリズムで使用される閾値やしきい値を抽出してください。例: 0.5, 0.8など"
            try:
                rag_threshold_results = self.rag_tool.search(threshold_query, "algorithm_specs", k=3)
                if rag_threshold_results:
                    thresholds = extract_thresholds_with_llm(" ".join([r['content'] for r in rag_threshold_results]))
                else:
                    thresholds = extract_thresholds_with_llm(spec_content)
            except:
                thresholds = extract_thresholds_with_llm(spec_content)

        # Map to actual CSV columns using fuzzy matching with multiple strategies
        available_columns = []
        if 'algorithm_data' in state and hasattr(state['algorithm_data'], 'columns'):
            available_columns = list(state['algorithm_data'].columns)
        elif 'core_data' in state and hasattr(state['core_data'], 'columns'):
            available_columns = list(state['core_data'].columns)

        # Enhanced column mapping with multiple matching strategies
        column_mapping = {
            'input': self._map_columns_to_csv(input_columns, available_columns),
            'output': self._map_columns_to_csv(output_columns, available_columns),
            'thresholds': thresholds
        }

        # Ensure we have at least some columns for plotting: rely on LLM/RAG results only (no domain-specific fallback)
        # If none extracted, leave mapping empty and downstream will handle/report appropriately.

        logger.info(f"Extracted column mapping - input: {list(column_mapping['input'].keys())}, output: {list(column_mapping['output'].keys())}, thresholds: {list(thresholds.keys())}")

        # Log final column mappings used for analysis
        logger.info(f"COLUMN_MAPPING_USED - Input: {column_mapping['input']}")
        logger.info(f"COLUMN_MAPPING_USED - Output: {column_mapping['output']}")
        logger.info(f"COLUMN_MAPPING_USED - Thresholds: {thresholds}")

        return {'column_mapping': column_mapping}

    def _map_columns_to_csv(self, spec_columns: List[str], csv_columns: List[str]) -> Dict[str, str]:
        """Map specification columns to actual CSV columns using multiple strategies"""
        mapping = {}

        for spec_col in spec_columns:
            if not spec_col:
                continue

            # Strategy 1: Exact match
            if spec_col in csv_columns:
                mapping[spec_col] = spec_col
                continue

            # Strategy 2: Fuzzy matching
            matched = self._fuzzy_match_column(spec_col, csv_columns)
            if matched:
                mapping[spec_col] = matched
                continue

            # Strategy 3: Common variations
            variations = self._generate_column_variations(spec_col)
            for variation in variations:
                if variation in csv_columns:
                    mapping[spec_col] = variation
                    break

        return mapping

    def _generate_column_variations(self, column: str) -> List[str]:
        """Generate minimal, generic variations of column names (domain-agnostic)."""
        base = column.lower()
        collapsed = base.replace('_', '')
        vars_set = {base, collapsed}
        return list(vars_set)

    def _extract_columns_with_regex(self, spec_content: str, column_type: str) -> List[str]:
        """Extract columns using regex patterns (generic, domain-agnostic)."""
        import re
        columns = []

        # Common column patterns in specifications
        if column_type == 'input':
            # Input feature patterns (generic)
            patterns = [
                r'入力[:\s]*([^、。\n]+)',
                r'特徴量[:\s]*([^、。\n]+)',
                r'feature[:\s]*([^,.\n]+)',
                r'column[:\s]*([^,.\n]+)'
            ]
            # Generic inputs
            common_inputs = ['confidence', 'score', 'probability', 'feature', 'signal', 'value']
        else:  # output
            # Output/result patterns (generic)
            patterns = [
                r'出力[:\s]*([^、。\n]+)',
                r'結果[:\s]*([^、。\n]+)',
                r'result[:\s]*([^,.\n]+)',
                r'output[:\s]*([^,.\n]+)'
            ]
            # Generic outputs
            common_outputs = ['result', 'prediction', 'output', 'label', 'status']

        # Extract using patterns
        for pattern in patterns:
            matches = re.findall(pattern, spec_content, re.IGNORECASE)
            for match in matches:
                # Clean the match
                clean_match = re.sub(r'[「」\[\](){}]', '', match.strip())
                if clean_match and len(clean_match) > 1:
                    columns.append(clean_match.lower())

        # Add common columns if none found
        if not columns:
            columns = common_inputs if column_type == 'input' else common_outputs

        # Remove duplicates and filter
        columns = list(set(columns))
        columns = [col for col in columns if len(col) > 1 and not col.isdigit()]

        logger.info(f"Regex extracted {column_type} columns: {columns}")
        return columns

    def _fuzzy_match_column(self, target: str, available: List[str]) -> Optional[str]:
        from difflib import get_close_matches
        matches = get_close_matches(target.lower(), [col.lower() for col in available], n=1, cutoff=0.6)
        return matches[0] if matches else None


class ConsistencyCheckerNode:
    """Node for checking consistency between data and specifications"""

    def __init__(self):
        self.repl_tool = REPLTool()
        self.rag_tool = RAGTool()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Check consistency for current dataset"""
        logger.info("Consistency checker processing")
        try:
            state.messages.append("ConsistencyChecker: start")
        except Exception:
            pass

        current_dataset = state.get_current_dataset()
        if not current_dataset:
            return state

        try:
            # Parse expected result from natural language
            expected_result = current_dataset.expected_result

            # Load data for checking
            dataframes = self.repl_tool.load_csv_data([
                current_dataset.algorithm_output_csv,
                current_dataset.core_output_csv
            ])

            # Perform consistency checks with algorithm config
            algorithm_config = getattr(state, 'algorithm_config', None)
            try:
                consistency_results = self._check_consistency(dataframes, expected_result, algorithm_config)
            except Exception as e:
                logger.error(f"Consistency checking failed: {e}")
                consistency_results = {
                    "expected_interpretation": expected_result,
                    "checks_performed": ["Consistency check failed due to error"],
                    "overall_consistent": False,
                    "issues": [f"Consistency check error: {str(e)}"],
                    "target_interval": None,
                    "require_exists": False,
                    "detection": {"exists": False, "longest_run": 0, "runs": []},
                    "input_column_stats": {}
                }

            # Update dataset state
            current_dataset.consistency_check = consistency_results
            current_dataset.status = "consistency_checked"

            logger.info(f"Consistency checking completed for dataset {current_dataset.id}")
            try:
                state.messages.append(f"ConsistencyChecker: completed for {current_dataset.id}")
            except Exception:
                pass

        except Exception as e:
            error_msg = f"Consistency checking failed: {e}"
            current_dataset.error_message = error_msg
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state

    def _check_consistency(self, dataframes: Dict[str, Any], expected: str, algorithm_config=None) -> Dict[str, Any]:
        """Check consistency between data and expectations using RAG-derived column information"""
        import re
        import numpy as np

        results = {
            "expected_interpretation": expected,
            "checks_performed": [],
            "overall_consistent": True,
            "issues": [],
            "target_interval": None,
            "require_exists": False,
            "detection": {
                "exists": False,
                "longest_run": 0,
                "runs": []
            },
            "input_column_stats": {},
            "rag_column_validation": {}  # Add RAG-based validation results
        }

        # Get column mapping from current dataset if available
        column_mapping = {}
        if hasattr(self, 'current_dataset') and self.current_dataset and hasattr(self.current_dataset, 'data_summary'):
            column_mapping = self.current_dataset.data_summary.get('column_mapping', {}).get('column_mapping', {})

        # Validate columns using RAG-derived mapping
        if column_mapping:
            results["rag_column_validation"] = self._validate_columns_with_mapping(dataframes, column_mapping)
            results["checks_performed"].append("RAG column mapping validation")

        # Basic file presence checks
        for name, df in dataframes.items():
            if len(df) == 0:
                results["issues"].append(f"Empty dataframe: {name}")
            else:
                results["checks_performed"].append(f"Dataframe {name}: {len(df)} rows")

        # Parse expected text: frame interval and keyword using LLM
        try:
            frame_range = extract_frame_range_with_llm(expected)
            if frame_range and len(frame_range) == 2:
                start_f, end_f = frame_range
                if start_f is not None and end_f is not None:
                    if start_f > end_f:
                        start_f, end_f = end_f, start_f
                    results["target_interval"] = {"start": start_f, "end": end_f}
                else:
                    logger.warning("Frame range contains None values")
            else:
                logger.warning("Frame range extraction returned invalid result")
        except RuntimeError as e:
            logger.warning(f"Frame range extraction failed: {e}")
            # Continue without target interval

        # Dynamic detection of required patterns based on algorithm config
        require_exists = False
        if algorithm_config:
            # Check if expected result mentions detection patterns
            for pattern_name in algorithm_config.detection_patterns.keys():
                if pattern_name.lower() in expected.lower():
                    require_exists = True
                    break
        else:
            # Fallback: check for common detection keywords
            detection_keywords = ['連続', '検知', '検出', '存在', '発生']
            require_exists = any(keyword in expected for keyword in detection_keywords)

        results["require_exists"] = require_exists

        # Analyze input data based on algorithm configuration
        core_df = None
        algo_df = None

        if algorithm_config:
            # Use algorithm config to distinguish input vs output data
            for df in dataframes.values():
                cols = set(df.columns)
                # Check if this dataframe contains output columns (algorithm output)
                if any(col in cols for col in algorithm_config.output_columns):
                    algo_df = df.copy()
                # Check if this dataframe contains input columns (core input)
                elif any(col in cols for col in algorithm_config.input_columns):
                    core_df = df.copy()
        else:
            # Fallback: heuristic-based detection
            for df in dataframes.values():
                cols = set(df.columns)
                # Simple heuristic: if dataframe has detection/result columns, it's algorithm output
                if any(col for col in cols if 'detection' in col.lower() or 'result' in col.lower()):
                    algo_df = df.copy()
                else:
                    core_df = df.copy()
                if core_df is not None and algo_df is not None:
                    break

        if core_df is not None:
            frame_col = 'frame' if 'frame' in core_df.columns else None
            # Use algorithm config to determine which columns to analyze
            if algorithm_config:
                analysis_cols = [col for col in algorithm_config.input_columns
                               if col in core_df.columns and core_df[col].dtype in ['int64', 'float64']]
            else:
                # Fallback: analyze numeric columns
                analysis_cols = [col for col in core_df.columns
                               if core_df[col].dtype in ['int64', 'float64'] and col != frame_col]

            if analysis_cols and frame_col and results["target_interval"] and frame_col in core_df.columns:
                s = results["target_interval"]["start"]
                e = results["target_interval"]["end"]
                sub = core_df[(core_df[frame_col] >= s) & (core_df[frame_col] <= e)].copy()

                if len(sub) > 0:
                    stats = {}
                    for col in analysis_cols:
                        mean_val = float(sub[col].mean())
                        stats[col] = {
                            "mean": round(mean_val, 3),
                            "min": float(sub[col].min()),
                            "max": float(sub[col].max())
                        }
                    results["input_column_stats"] = stats

        # Detect algorithm output data using configuration
        if algo_df is None and algorithm_config:
            # Use algorithm config to find output dataframe
            for df in dataframes.values():
                cols = set(df.columns)
                if any(col in cols for col in algorithm_config.output_columns):
                    algo_df = df.copy()
                    break

        # Fallback: heuristic detection if config didn't work
        if algo_df is None:
            for df in dataframes.values():
                cols = set(df.columns)
                if 'frame_num' in cols or 'frame' in cols:
                    # Look for detection/result columns dynamically
                    detection_cols = [col for col in cols if any(keyword in col.lower()
                        for keyword in ['detection', 'result', 'closed', 'drowsy', 'sleep', 'blink'])]
                    if detection_cols:
                        algo_df = df.copy()
                        break

        if algo_df is None:
            results["issues"].append("Algorithm dataframe not found for detection")
            results["overall_consistent"] = False
            return results

        # Select frame column
        frame_col = 'frame_num' if 'frame_num' in algo_df.columns else ('frame' if 'frame' in algo_df.columns else None)
        if frame_col is None:
            results["issues"].append("Frame column not found in algorithm output")
            results["overall_consistent"] = False
            return results

        # Restrict to interval if provided
        if results["target_interval"] and frame_col and frame_col in algo_df.columns:
            s = results["target_interval"]["start"]
            e = results["target_interval"]["end"]
            algo_df = algo_df[(algo_df[frame_col] >= s) & (algo_df[frame_col] <= e)].copy()

        # Build detection series dynamically based on algorithm config
        detection_series = None

        if algorithm_config:
            # Use output columns from algorithm config
            detection_cols = [col for col in algorithm_config.output_columns
                            if col in algo_df.columns and algo_df[col].dtype in ['int64', 'float64', 'bool']]
        else:
            # Fallback: heuristic detection
            detection_cols = [col for col in algo_df.columns if any(keyword in col.lower()
                for keyword in ['detection', 'result', 'closed', 'drowsy', 'sleep', 'blink'])]

        if detection_cols:
            # Combine all detection indicators
            combined_detection = np.zeros(len(algo_df), dtype=bool)
            for col in detection_cols:
                if algo_df[col].dtype == bool:
                    combined_detection |= algo_df[col].astype(bool)
                elif algo_df[col].dtype in ['int64', 'float64']:
                    # Use algorithm config thresholds if available
                    threshold = 0.5  # default threshold
                    if algorithm_config:
                        for thresh_name, thresh_val in algorithm_config.thresholds.items():
                            if col.lower() in thresh_name.lower():
                                threshold = thresh_val
                                break
                    combined_detection |= (algo_df[col] > threshold)
            detection_series = combined_detection

        if detection_series is None or detection_series.size == 0:
            results["issues"].append("No detection indicators found in algorithm output")
            results["overall_consistent"] = False
            return results

        # Detect runs of consecutive True (length >= 2)
        runs = []
        longest = 0
        if detection_series.size > 0:
            start_idx = None
            for idx, val in enumerate(detection_series):
                if val and start_idx is None:
                    start_idx = idx
                if (not val or idx == len(detection_series) - 1) and start_idx is not None:
                    end_idx = idx if val and idx == len(detection_series) - 1 else idx - 1
                    run_len = end_idx - start_idx + 1
                    if run_len >= 2:
                        # Map back to frame numbers
                        frame_values = algo_df[frame_col].to_numpy()
                        runs.append({
                            "start_frame": int(frame_values[start_idx]),
                            "end_frame": int(frame_values[end_idx]),
                            "length": int(run_len)
                        })
                        longest = max(longest, run_len)
                    start_idx = None

        exists = len(runs) > 0
        results["detection"] = {
            "exists": exists,
            "longest_run": int(longest),
            "runs": runs
        }

        # Consistency decision
        if require_exists and not exists:
            results["overall_consistent"] = False
            results["issues"].append("Expected continuous closure not found in target interval")
        else:
            results["overall_consistent"] = True

        return results

    def _validate_columns_with_mapping(self, dataframes: Dict[str, Any], column_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataframes using RAG-derived column mapping"""
        validation_results = {
            "input_columns_found": {},
            "output_columns_found": {},
            "threshold_validation": {},
            "column_coverage": {}
        }

        # Check input columns
        if 'input' in column_mapping:
            for spec_col, actual_col in column_mapping['input'].items():
                found = False
                for df_name, df in dataframes.items():
                    if actual_col in df.columns:
                        validation_results["input_columns_found"][spec_col] = {
                            "mapped_to": actual_col,
                            "dataframe": df_name,
                            "exists": True
                        }
                        found = True
                        break
                if not found:
                    validation_results["input_columns_found"][spec_col] = {
                        "mapped_to": actual_col,
                        "exists": False
                    }

        # Check output columns
        if 'output' in column_mapping:
            for spec_col, actual_col in column_mapping['output'].items():
                found = False
                for df_name, df in dataframes.items():
                    if actual_col in df.columns:
                        validation_results["output_columns_found"][spec_col] = {
                            "mapped_to": actual_col,
                            "dataframe": df_name,
                            "exists": True
                        }
                        found = True
                        break
                if not found:
                    validation_results["output_columns_found"][spec_col] = {
                        "mapped_to": actual_col,
                        "exists": False
                    }

        # Validate thresholds if available
        if 'thresholds' in column_mapping:
            for threshold_name, threshold_value in column_mapping['thresholds'].items():
                validation_results["threshold_validation"][threshold_name] = {
                    "value": threshold_value,
                    "validated": False
                }
                # Try to find columns that might be related to this threshold
                for df_name, df in dataframes.items():
                    threshold_related_cols = [col for col in df.columns if threshold_name.lower().replace('_threshold', '') in col.lower()]
                    if threshold_related_cols:
                        validation_results["threshold_validation"][threshold_name]["validated"] = True
                        validation_results["threshold_validation"][threshold_name]["related_columns"] = threshold_related_cols

        # Calculate column coverage
        total_spec_columns = len(column_mapping.get('input', {})) + len(column_mapping.get('output', {}))
        found_columns = len([col for col in validation_results["input_columns_found"].values() if col.get("exists")]) + \
                       len([col for col in validation_results["output_columns_found"].values() if col.get("exists")])

        validation_results["column_coverage"] = {
            "total_specified": total_spec_columns,
            "found": found_columns,
            "coverage_percentage": (found_columns / total_spec_columns * 100) if total_spec_columns > 0 else 0
        }

        logger.info(f"RAG column validation: {found_columns}/{total_spec_columns} columns found ({validation_results['column_coverage']['coverage_percentage']:.1f}%)")
        return validation_results


class HypothesisGeneratorNode:
    """Node for generating hypotheses about issues"""

    def __init__(self):
        from ..agents.hypothesis_generator_agent import HypothesisGeneratorAgent
        self.hypothesis_agent = HypothesisGeneratorAgent()
        self.rag_tool = RAGTool()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Generate hypotheses for current dataset"""
        logger.info("Hypothesis generator processing")
        try:
            state.messages.append("HypothesisGenerator: start")
        except Exception:
            pass

        current_dataset = state.get_current_dataset()
        if not current_dataset:
            return state

        try:
            # Get analysis results and consistency check from state/dataset
            analysis_results = getattr(state, 'analysis_results', {}).get(current_dataset.id, {})
            consistency_check = current_dataset.consistency_check if current_dataset.consistency_check else {}

            # Generate hypotheses using HypothesisGeneratorAgent
            hypotheses = self.hypothesis_agent.generate_hypotheses(
                current_dataset, analysis_results, consistency_check
            )

            # Update dataset state
            current_dataset.hypotheses = hypotheses
            current_dataset.status = "hypothesis_generated"

            # Store hypotheses in state for later use
            if not hasattr(state, 'hypotheses'):
                state.hypotheses = {}
            state.hypotheses[current_dataset.id] = hypotheses

            logger.info(f"Generated {len(hypotheses)} hypotheses for dataset {current_dataset.id}")
            try:
                state.messages.append(f"HypothesisGenerator: {len(hypotheses)} hypotheses for {current_dataset.id}")
            except Exception:
                pass

        except Exception as e:
            error_msg = f"Hypothesis generation failed: {e}"
            current_dataset.error_message = error_msg
            current_dataset.status = "failed"
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state

    def _generate_hypotheses(self, dataset: DatasetInfo) -> List[Dict[str, Any]]:
        """Generate hypotheses based on analysis results"""
        hypotheses = []

        # Simplified hypothesis generation
        # In a real system, this would be more sophisticated

        if dataset.consistency_check and not dataset.consistency_check.get("overall_consistent", True):
            hypotheses.append({
                "id": "consistency_issue",
                "type": "consistency",
                "description": "Data consistency issues detected",
                "confidence": 0.8
            })

        # Add more hypotheses based on data patterns
        hypotheses.append({
            "id": "performance_baseline",
            "type": "performance",
            "description": "Establish performance baseline",
            "confidence": 0.6
        })

        return hypotheses


class VerifierNode:
    """Node for verifying hypotheses through testing"""

    def __init__(self):
        from ..agents.verifier_agent import VerifierAgent
        self.verifier_agent = VerifierAgent()
        self.max_iterations = config.langgraph.max_iterations

    def process(self, state: AnalysisState) -> AnalysisState:
        """Verify hypotheses for current dataset"""
        logger.info("Verifier processing")
        try:
            state.messages.append("Verifier: start")
        except Exception:
            pass

        current_dataset = state.get_current_dataset()
        if not current_dataset:
            return state

        try:
            # Verify each hypothesis using VerifierAgent
            verification_results = []

            for hypothesis in current_dataset.hypotheses or []:
                if hasattr(hypothesis, 'model_dump'):  # Pydantic model
                    result = self.verifier_agent.verify_hypothesis(current_dataset, hypothesis)
                else:
                    # Legacy dict format
                    result = self._verify_hypothesis(hypothesis, current_dataset)

                verification_results.append(result)

                # Check if we should continue or stop
                if hasattr(result, 'success'):
                    if result.success:
                        break
                elif isinstance(result, dict) and result.get("success"):
                    break

            # Update dataset state
            current_dataset.verification_results = verification_results
            current_dataset.status = "verified"

            logger.info(f"Verification completed for dataset {current_dataset.id}")
            try:
                state.messages.append(f"Verifier: completed for {current_dataset.id}")
            except Exception:
                pass

        except Exception as e:
            error_msg = f"Verification failed: {e}"
            current_dataset.error_message = error_msg
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state

    def _verify_hypothesis(self, hypothesis: Dict[str, Any], dataset: DatasetInfo) -> Dict[str, Any]:
        """Verify a single hypothesis"""
        # Use consistency check results and core data to assign likely causes
        success = True
        causes = []
        evidence = []

        cc = dataset.consistency_check or {}
        require_exists = cc.get("require_exists", False)
        detection = (cc.get("detection") or {})
        exists = bool(detection.get("exists", False))
        input_stats = cc.get("input_column_stats", {})

        if require_exists and not exists:
            success = False

            # Analyze input column statistics dynamically
            if input_stats:
                for col, stats in input_stats.items():
                    mean_val = stats.get("mean", 1.0)
                    evidence.append(f"{col}平均={mean_val}")

                    # Dynamic analysis based on column name patterns
                    if 'openness' in col.lower() or 'confidence' in col.lower():
                        if mean_val < 0.3:
                            causes.append(f"{col}の値が低すぎる（検知対象が存在しない可能性）")
                        elif mean_val > 0.8:
                            causes.append(f"{col}の値が高すぎる（検知機能の感度が低い可能性）")
                        else:
                            causes.append(f"{col}の閾値設定ミスの可能性")
                    elif 'probability' in col.lower() or 'score' in col.lower():
                        if mean_val < 0.5:
                            causes.append(f"{col}の確率値が低い（信頼性の低い入力データ）")
                        else:
                            causes.append(f"{col}の出力範囲確認の必要性")
                    else:
                        # Generic analysis for other numeric columns
                        min_val = stats.get("min", 0)
                        max_val = stats.get("max", 1)
                        if max_val - min_val < 0.1:
                            causes.append(f"{col}の変動幅が小さい（安定した入力値）")
                        else:
                            causes.append(f"{col}の値分布を確認する必要性")

            if not input_stats:
                causes.append("被験者のタスク不正の可能性")
                causes.append("コアの検出機能に問題がある可能性")

        if not causes:
            causes.append("仕様通り動作")

        return {
            "hypothesis_id": hypothesis["id"],
            "success": success,
            "result": "検証完了",
            "causes": causes,
            "evidence": evidence
        }


class ReporterNode:
    """Node for generating final reports"""

    def __init__(self):
        from ..agents.reporter_agent import ReporterAgent
        self.reporter_agent = ReporterAgent()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Generate report for current dataset"""
        logger.info("Reporter processing")
        try:
            state.messages.append("Reporter: start")
        except Exception:
            pass

        current_dataset = state.get_current_dataset()
        if not current_dataset:
            return state

        try:
            # Get analysis results and hypotheses from state
            analysis_results = state.analysis_results.get(current_dataset.id, {}) if hasattr(state, 'analysis_results') else {}
            hypotheses = state.hypotheses.get(current_dataset.id, []) if hasattr(state, 'hypotheses') and isinstance(state.hypotheses, dict) else []

            # Generate report using ReporterAgent
            report_content = self.reporter_agent.generate_report(
                current_dataset, analysis_results, hypotheses
            )

            # Update dataset state & write debug snapshot
            current_dataset.report_content = report_content
            current_dataset.status = "completed"

            try:
                dataset_paths = get_report_paths(config.output_dir, f"report_{current_dataset.id}")
                logs_dir = dataset_paths["logs"]
                contexts_dir = dataset_paths["contexts"]
                import json
                # Write a compact state snapshot for reporter stage
                with open(logs_dir / "reporter_snapshot.json", 'w', encoding='utf-8') as f:
                    json.dump({
                        "messages": getattr(state, 'messages', []),
                        "workflow_step": state.workflow_step,
                        "dataset_id": current_dataset.id,
                        "algorithm_config": getattr(config, 'algorithm', {}).model_dump() if hasattr(getattr(config, 'algorithm', {}), 'model_dump') else {},
                        "has_report": bool(report_content),
                    }, f, ensure_ascii=False, indent=2)
                # Persist analysis_results for this dataset for manual inspection
                with open(contexts_dir / "analysis_results_snapshot.json", 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            except Exception as ex:
                logger.warning(f"Failed to write reporter snapshots: {ex}")

            # Move to next dataset
            state.advance_dataset()

            logger.info(f"Report generated for dataset {current_dataset.id}")
            try:
                state.messages.append(f"Reporter: report generated for {current_dataset.id}")
            except Exception:
                pass

        except Exception as e:
            error_msg = f"Report generation failed: {e}"
            current_dataset.error_message = error_msg
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state
