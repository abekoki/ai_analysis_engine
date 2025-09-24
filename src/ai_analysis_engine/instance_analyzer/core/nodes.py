"""
LangGraph nodes for the AI Analysis Engine workflow
"""

from typing import Dict, Any, List
from datetime import datetime

from ..models.state import AnalysisState, DatasetInfo
from ..models.types import DataSummary, ConsistencyCheckResult, Hypothesis, VerificationResult
from ..tools.rag_tool import RAGTool
from ..tools.repl_tool import REPLTool
from ..utils.logger import get_logger
from ..utils.exploration_utils import extract_frame_range_with_llm
from ..config import config

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

            # Analyze data
            analysis_results = {}
            for name, df in dataframes.items():
                try:
                    logger.info(f"Analyzing dataframe {name} with shape {df.shape}")
                    analysis_results[name] = self.repl_tool.analyze_dataframe(df, name)
                except Exception as e:
                    logger.error(f"Failed to analyze dataframe {name}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    analysis_results[name] = {"error": str(e)}

            # Create plots
            logger.info("Creating data plots")
            plot_paths = self._create_data_plots(dataframes, current_dataset.id, state)
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

            # Update dataset state
            current_dataset.data_summary = {
                "analysis": analysis_results,
                "plots": plot_paths,
                "column_info": column_info
            }
            current_dataset.status = "data_checked"

            # Store analysis results in state for later use
            logger.info("Storing analysis results in state")
            try:
                if not hasattr(state, 'analysis_results'):
                    state.analysis_results = {}
                state.analysis_results[current_dataset.id] = {
                    "analysis": analysis_results,
                    "plots": plot_paths,
                    "column_info": column_info
                }
                logger.info(f"Successfully stored results for dataset {current_dataset.id}")
            except Exception as e:
                logger.error(f"Failed to store analysis results: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

            logger.info(f"Data checking completed for dataset {current_dataset.id}")

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
            # Create output directory
            from ..config import config as global_config
            plots_dir = global_config.output_dir / "plots" / dataset_id
            plots_dir.mkdir(parents=True, exist_ok=True)

            for name, df in dataframes.items():
                if len(df) == 0:
                    continue

                # Determine x axis (prefer frame-based)
                x_col = None
                if 'frame_num' in df.columns:
                    x_col = 'frame_num'
                elif 'frame' in df.columns:
                    x_col = 'frame'

                # Determine y series dynamically based on available columns
                y_series = []
                y_label = 'Value'

                # Dynamic column selection based on algorithm configuration
                available_cols = set(df.columns)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # If we have algorithm config, use it to determine what to plot
                if hasattr(state, 'algorithm_config') and state.algorithm_config:
                    config = state.algorithm_config
                    # For algorithm output files, plot output columns
                    if any(col in available_cols for col in config.output_columns):
                        for col in config.output_columns:
                            if col in available_cols:
                                if col in numeric_cols:
                                    # Create readable label from column name
                                    label = col.replace('_', ' ').title()
                                    y_series.append((col, label))
                                    y_label = 'Algorithm Output'
                                elif df[col].dtype in ['object', 'category']:
                                    # For categorical data, try to convert to numeric if possible
                                    if df[col].str.isnumeric().all():
                                        df_copy = df.copy()
                                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                                        if not df_copy[col].isna().all():
                                            label = col.replace('_', ' ').title()
                                            y_series.append((col, label))
                                            y_label = 'Algorithm Output'
                    # For core input files, plot input columns
                    elif any(col in available_cols for col in config.input_columns):
                        for col in config.input_columns:
                            if col in available_cols and col in numeric_cols:
                                # Create readable label from column name
                                label = col.replace('_', ' ').title()
                                y_series.append((col, label))
                                y_label = 'Input Values'
                else:
                    # Fallback: plot first numeric column
                    num_cols = df.select_dtypes(include=['number']).columns
                    if len(num_cols) > 0:
                        y_series.append((num_cols[0], str(num_cols[0])))

                if x_col is None and len(y_series) == 0:
                    continue

                # Build plotting code
                series_code_lines = []
                for col, label in y_series:
                    # Booleans to int for visualization
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
                plot_path = str(plots_dir / f"{name}_timeseries.png")
                result = self.repl_tool.create_plot(code, {"df": df}, plot_path)

                if result.get("success"):
                    plot_paths[name] = plot_path

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


class ConsistencyCheckerNode:
    """Node for checking consistency between data and specifications"""

    def __init__(self):
        self.repl_tool = REPLTool()
        self.rag_tool = RAGTool()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Check consistency for current dataset"""
        logger.info("Consistency checker processing")

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

        except Exception as e:
            error_msg = f"Consistency checking failed: {e}"
            current_dataset.error_message = error_msg
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state

    def _check_consistency(self, dataframes: Dict[str, Any], expected: str, algorithm_config=None) -> Dict[str, Any]:
        """Check consistency between data and expectations"""
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
            "input_column_stats": {}
        }

        # Basic file presence checks
        for name, df in dataframes.items():
            if len(df) == 0:
                results["issues"].append(f"Empty dataframe: {name}")
            else:
                results["checks_performed"].append(f"Dataframe {name}: {len(df)} rows")

        # Parse expected text: frame interval and keyword using LLM
        try:
            frame_range = extract_frame_range_with_llm(expected)
            if frame_range:
                start_f, end_f = frame_range
                if start_f > end_f:
                    start_f, end_f = end_f, start_f
                results["target_interval"] = {"start": start_f, "end": end_f}
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
                if core_df and algo_df:
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


class HypothesisGeneratorNode:
    """Node for generating hypotheses about issues"""

    def __init__(self):
        from ..agents.hypothesis_generator_agent import HypothesisGeneratorAgent
        self.hypothesis_agent = HypothesisGeneratorAgent()
        self.rag_tool = RAGTool()

    def process(self, state: AnalysisState) -> AnalysisState:
        """Generate hypotheses for current dataset"""
        logger.info("Hypothesis generator processing")

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

        current_dataset = state.get_current_dataset()
        if not current_dataset:
            return state

        try:
            # Verify each hypothesis using VerifierAgent
            verification_results = []

            for hypothesis in current_dataset.hypotheses or []:
                if hasattr(hypothesis, 'model_dump'):  # Pydantic model
                    # Convert Hypothesis object to dict for compatibility
                    hypothesis_dict = hypothesis.model_dump()
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

            # Update dataset state
            current_dataset.report_content = report_content
            current_dataset.status = "completed"

            # Move to next dataset
            state.advance_dataset()

            logger.info(f"Report generated for dataset {current_dataset.id}")

        except Exception as e:
            error_msg = f"Report generation failed: {e}"
            current_dataset.error_message = error_msg
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state
