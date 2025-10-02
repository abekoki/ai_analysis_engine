"""
Compatibility adapter for the legacy InstanceAnalyzer API.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple



from ..config.settings import Settings
from .config import config
from .utils.file_utils import ensure_directory, ensure_analysis_output_structure, get_report_paths, write_file
from .utils.logger import update_instance_log_file
from .config.library_config import AnalysisConfig
from .library_api import AIAnalysisEngine

logger = logging.getLogger(__name__)


class InstanceAnalyzer:
    """Legacy-compatible adapter around the new AI Analysis Engine."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.output_base_dir: Optional[Path] = None
        self._engine: Optional[AIAnalysisEngine] = None
        self._initialized = False

    def set_output_base_dir(self, base_dir: Path) -> None:
        self.output_base_dir = Path(base_dir)
        # Update global config output directory so downstream utilities use the run-specific path
        config.output_dir = self.output_base_dir
        get_report_paths(self.output_base_dir, "summary")
        summary_logs = self.output_base_dir / "summary" / "instance_analysis.log"
        ensure_directory(str(summary_logs.parent))
        update_instance_log_file(summary_logs)

    def analyze_instances(self, evaluation_data: Dict[str, Any], max_instances: Optional[int] = None) -> List[Dict[str, Any]]:
        logger.info("Starting legacy instance analysis via adapter")
        logger.info(f"evaluation_data keys: {list(evaluation_data.keys()) if evaluation_data else 'None'}")
        self._ensure_engine_initialized()
        output_dir = self._resolve_output_dir()

        algorithm_outputs_data = evaluation_data.get("algorithm_outputs", [])
        core_outputs_data = evaluation_data.get("core_outputs", [])

        # データオブジェクトからCSVディレクトリ情報を抽出し、IDとのマッピングを構築
        algorithm_outputs: List[str] = []
        core_outputs: List[str] = []
        algo_dirs_by_id: Dict[int, str] = {}
        core_dirs_by_id: Dict[int, str] = {}
        algo_dir_fallbacks: List[str] = []
        core_dir_fallbacks: List[str] = []

        for algo_data in algorithm_outputs_data:
            if not isinstance(algo_data, Mapping):
                continue
            algo_dir_rel = algo_data.get('algorithm_output_dir')
            if not algo_dir_rel:
                continue
            algo_id = self._safe_int(algo_data.get('algorithm_output_ID') or algo_data.get('algorithm_output_id'))
            if algo_id is not None:
                algo_dirs_by_id[algo_id] = algo_dir_rel
            else:
                algo_dir_fallbacks.append(algo_dir_rel)

        for core_data in core_outputs_data:
            if not isinstance(core_data, Mapping):
                continue
            core_dir_rel = core_data.get('core_lib_output_dir')
            if not core_dir_rel:
                continue
            core_id = self._safe_int(core_data.get('core_lib_output_ID') or core_data.get('core_lib_output_id'))
            if core_id is not None:
                core_dirs_by_id[core_id] = core_dir_rel
            else:
                core_dir_fallbacks.append(core_dir_rel)

        # expected_results は評価データ（DataFrame）から取得
        eval_df = evaluation_data.get("data")
        logger.info(f"eval_df type: {type(eval_df)}, is None: {eval_df is None}")
        if eval_df is not None:
            logger.info(f"eval_df columns: {list(eval_df.columns) if hasattr(eval_df, 'columns') else 'no columns'}")
            logger.info(f"eval_df shape: {eval_df.shape if hasattr(eval_df, 'shape') else 'no shape'}")

        # expected_results の初期化
        expected_results = []
        evaluation_intervals = []

        if eval_df is not None and hasattr(eval_df, 'expected_is_drowsy'):
            logger.info("Processing expected_is_drowsy column")
            # 失敗タスクのみをフィルタリング（is_correctが0のもの）
            if 'is_correct' in eval_df.columns:
                logger.info(f"is_correct values: {eval_df['is_correct'].value_counts().to_dict() if hasattr(eval_df['is_correct'], 'value_counts') else 'no value_counts'}")
                failed_tasks_df = eval_df[eval_df['is_correct'] == 0].copy()
                logger.info(f"Filtering to failed tasks only: {len(failed_tasks_df)} failed tasks out of {len(eval_df)} total tasks")
                eval_df = failed_tasks_df
            else:
                logger.warning("is_correct column not found, processing all tasks")
        else:
            logger.warning("eval_df is None or does not have expected_is_drowsy column")

        if eval_df is not None and len(eval_df) > 0:
            aligned_algorithm_outputs: List[str] = []
            aligned_core_outputs: List[str] = []

            for _, row in eval_df.iterrows():
                expected = int(row['expected_is_drowsy'])  # intに変換
                expected_results.append(expected)

                # 評価区間情報を別途収集
                start = row.get('start', None)
                end = row.get('end', None)
                interval_info = {
                    'start': start,
                    'end': end,
                    'task_id': row.get('task_id'),
                    'video_ID': row.get('video_ID'),
                    'tag_ID': row.get('tag_ID')
                } if start is not None or end is not None else None
                evaluation_intervals.append(interval_info)

                video_id = self._safe_int(row.get('video_ID'))
                algo_id = self._safe_int(row.get('algorithm_output_ID') or row.get('algorithm_output_id'))
                core_id = self._safe_int(row.get('core_lib_output_ID') or row.get('core_lib_output_id'))

                algo_dir_rel = algo_dirs_by_id.get(algo_id)
                if algo_dir_rel is None and isinstance(row.get('algorithm_output_dir'), str):
                    row_algo_dir = row.get('algorithm_output_dir').strip()
                    if row_algo_dir:
                        algo_dir_rel = row_algo_dir
                if algo_dir_rel is None and algo_dir_fallbacks:
                    algo_dir_rel = algo_dir_fallbacks[0]

                core_dir_rel = core_dirs_by_id.get(core_id)
                if core_dir_rel is None and isinstance(row.get('core_lib_output_dir'), str):
                    row_core_dir = row.get('core_lib_output_dir').strip()
                    if row_core_dir:
                        core_dir_rel = row_core_dir
                if core_dir_rel is None and core_dir_fallbacks:
                    core_dir_rel = core_dir_fallbacks[0]

                if not algo_dir_rel or not core_dir_rel:
                    raise ValueError(
                        "Failed to resolve output directories for evaluation row",
                        {
                            'algorithm_output_ID': algo_id,
                            'core_lib_output_ID': core_id,
                            'available_algorithm_dirs': list(algo_dirs_by_id.keys()),
                            'available_core_dirs': list(core_dirs_by_id.keys()),
                        },
                    )

                algo_path = self._resolve_csv_within_directory(algo_dir_rel, video_id=video_id)
                core_path = self._resolve_csv_within_directory(core_dir_rel, video_id=video_id)

                aligned_algorithm_outputs.append(algo_path)
                aligned_core_outputs.append(core_path)

            algorithm_outputs = aligned_algorithm_outputs
            core_outputs = aligned_core_outputs
        else:
            expected_results = []
        # dataset_ids を失敗タスクに基づいて生成
        if eval_df is not None and len(expected_results) > 0:
            # 失敗タスクのタスクIDを使用
            if 'task_id' in eval_df.columns:
                dataset_ids = eval_df['task_id'].tolist()
            else:
                dataset_ids = [f"failed_task_{i + 1}" for i in range(len(expected_results))]
        else:
            dataset_ids = evaluation_data.get("dataset_ids")

        # 各リストの長さをexpected_resultsに合わせる
        num_samples = len(expected_results)
        logger.info(f"Before expansion - algorithm_outputs: {len(algorithm_outputs)}, core_outputs: {len(core_outputs)}, expected_results: {num_samples}")
        if algorithm_outputs and num_samples > len(algorithm_outputs):
            # algorithm_outputsを拡張（同じ要素を繰り返す）
            algorithm_outputs = algorithm_outputs * ((num_samples // len(algorithm_outputs)) + 1)
            algorithm_outputs = algorithm_outputs[:num_samples]
        if core_outputs and num_samples > len(core_outputs):
            # core_outputsを拡張（同じ要素を繰り返す）
            core_outputs = core_outputs * ((num_samples // len(core_outputs)) + 1)
            core_outputs = core_outputs[:num_samples]
        logger.info(f"After expansion - algorithm_outputs: {len(algorithm_outputs)}, core_outputs: {len(core_outputs)}, expected_results: {num_samples}")

        # DatasetInfoに必要な形式にデータを準備
        # algorithm_codes と evaluation_codes は設定から取得
        algorithm_codes = []
        evaluation_codes = []

        # 設定からコードファイルを取得
        algo_resources = self.settings.get('instance_analyzer', {}).get('resources', {}).get('algorithm', {})
        eval_resources = self.settings.get('instance_analyzer', {}).get('resources', {}).get('evaluation', {})

        # アルゴリズムコードファイルを取得
        for code_dir in algo_resources.get('code_dirs', []):
            code_path = os.path.join(self.settings.get('global.external_resources_path', ''), code_dir)
            if os.path.exists(code_path):
                for root, dirs, files in os.walk(code_path):
                    for file in files:
                        if file.endswith('.py'):
                            algorithm_codes.append(os.path.join(root, file))

        # 評価コードファイルを取得
        for code_dir in eval_resources.get('code_dirs', []):
            code_path = os.path.join(self.settings.get('global.external_resources_path', ''), code_dir)
            if os.path.exists(code_path):
                for root, dirs, files in os.walk(code_path):
                    for file in files:
                        if file.endswith('.py'):
                            evaluation_codes.append(os.path.join(root, file))

        # algorithm_codes と evaluation_codes を拡張（各要素をリストに変換）
        if algorithm_codes:
            # 各データセットに対して同じコードファイルリストを使用
            algorithm_codes_list = [algorithm_codes[:] for _ in range(num_samples)]  # リストのコピーを作成
        else:
            algorithm_codes_list = [[] for _ in range(num_samples)]

        if evaluation_codes:
            # 各データセットに対して同じコードファイルリストを使用
            evaluation_codes_list = [evaluation_codes[:] for _ in range(num_samples)]  # リストのコピーを作成
        else:
            evaluation_codes_list = [[] for _ in range(num_samples)]

        if not algorithm_outputs:
            raise ValueError("evaluation_data.algorithm_outputs is required and cannot be empty")
        if not core_outputs:
            raise ValueError("evaluation_data.core_outputs is required and cannot be empty")
        if not expected_results:
            raise ValueError("evaluation_data.expected_results is required and cannot be empty")

        if not isinstance(algorithm_outputs, list) or not isinstance(core_outputs, list) or not isinstance(expected_results, list):
            raise TypeError("algorithm_outputs, core_outputs, expected_results must be lists")
        if len({len(algorithm_outputs), len(core_outputs), len(expected_results)}) != 1:
            raise ValueError("algorithm_outputs, core_outputs, expected_results must have identical lengths")

        if dataset_ids is None:
            dataset_ids_resolved = [f"dataset_{i + 1}" for i in range(len(algorithm_outputs))]
        else:
            if not isinstance(dataset_ids, list):
                raise TypeError("dataset_ids must be a list when provided")
            if len(dataset_ids) != len(algorithm_outputs):
                raise ValueError("dataset_ids length must match algorithm_outputs length")
            dataset_ids_resolved = dataset_ids

        if max_instances is not None and max_instances > 0:
            algorithm_outputs = algorithm_outputs[:max_instances]
            core_outputs = core_outputs[:max_instances]
            expected_results = expected_results[:max_instances]
            dataset_ids_resolved = dataset_ids_resolved[:max_instances]
            evaluation_intervals = evaluation_intervals[:max_instances]

            if hasattr(evaluation_data, "update"):
                evaluation_data["dataset_ids"] = dataset_ids_resolved

        results = self._engine.analyze(
            algorithm_outputs=algorithm_outputs,
            core_outputs=core_outputs,
            expected_results=expected_results,
            output_dir=str(output_dir),
            dataset_ids=dataset_ids_resolved,
            algorithm_codes=algorithm_codes_list,
            evaluation_codes=evaluation_codes_list,
            evaluation_intervals=evaluation_intervals,
        )

        serialized_results = []
        for result in results:
            data = result.model_dump() if hasattr(result, "model_dump") else result
            metadata = data.get("metadata") or {}

            dataset_id = data.get("dataset_id") or (data.get("dataset") or {}).get("id")
            report_name = f"report_{dataset_id}" if dataset_id else "report"
            paths = get_report_paths(self.output_base_dir, report_name)

            context_blob = metadata.get("langgraph_memory")
            if context_blob:
                events = context_blob.get("events") if isinstance(context_blob, dict) else None
                checkpointer_dump = context_blob.get("checkpointer") if isinstance(context_blob, dict) else None

                base_name = dataset_id or "unknown"
                events_file = paths["contexts"] / f"langgraph_events_{base_name}.json"
                checkpoint_file = paths["contexts"] / f"langgraph_checkpointer_{base_name}.json"

                try:
                    if events is not None:
                        write_file(str(events_file), json.dumps(events, ensure_ascii=False, indent=2))
                        self.logger.info(f"LangGraph events saved: {events_file}")
                    else:
                        self.logger.debug("No LangGraph events available to save for %s", dataset_id)

                    if checkpointer_dump:
                        write_file(str(checkpoint_file), json.dumps(checkpointer_dump, ensure_ascii=False, indent=2))
                        self.logger.info(f"LangGraph checkpointer saved: {checkpoint_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save LangGraph context for {dataset_id}: {e}")

            log_entries = metadata.get("log_buffer")
            if log_entries:
                log_file = paths["logs"] / "analysis.log"
                try:
                    log_text = "\n".join(str(entry) for entry in log_entries)
                    with open(log_file, "a", encoding="utf-8") as lp:
                        lp.write(log_text + "\n")
                except Exception as e:
                    self.logger.warning(f"Failed to append logs for {dataset_id}: {e}")

            serialized_results.append(data)

        return serialized_results

    def _ensure_engine_initialized(self) -> None:
        if self._initialized:
            return

        config = self._build_analysis_config()
        self._engine = AIAnalysisEngine(config=config)
        resources = self._collect_resources()

        if not self._engine.initialize(**resources):
            raise RuntimeError("Failed to initialize AI analysis engine")

        self._initialized = True

    def _build_analysis_config(self) -> AnalysisConfig:
        instance_config = self.settings.get("instance_analyzer", {})
        llm_model = instance_config.get("llm_model", "gpt-4")
        temperature = instance_config.get("temperature", 0.1)
        max_iterations = instance_config.get("max_hypothesis_attempts", 3)

        output_dir = str(self._resolve_output_dir())

        return AnalysisConfig(
            model=llm_model,
            temperature=float(temperature),
            max_iterations=int(max_iterations),
            output_dir=output_dir,
        )

    def _collect_resources(self) -> Dict[str, List[str]]:
        resources_config = self.settings.get("instance_analyzer.resources")
        if not resources_config:
            raise ValueError("instance_analyzer.resources configuration is required")

        root_path = Path(self.settings.get("global.external_resources_path", "./external")).resolve()

        def gather(paths: Sequence[str], *, allow_extensions: Optional[Iterable[str]] = None) -> List[str]:
            resolved: List[str] = []
            for rel in paths:
                path = (root_path / rel).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Resource path not found: {path}")

                if path.is_dir():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            if allow_extensions and file_path.suffix.lower() not in allow_extensions:
                                continue
                            resolved.append(str(file_path))
                else:
                    if allow_extensions and path.suffix.lower() not in allow_extensions:
                        raise ValueError(f"Unsupported file extension for path: {path}")
                    resolved.append(str(path))

            if not resolved:
                raise ValueError(f"No files found for resource paths: {paths}")
            return sorted(resolved)

        algorithm_config = resources_config.get("algorithm", {})
        evaluation_config = resources_config.get("evaluation", {})

        algorithm_spec_dirs = algorithm_config.get("spec_dirs", [])
        evaluation_spec_dirs = evaluation_config.get("spec_dirs", [])
        algorithm_code_dirs = algorithm_config.get("code_dirs", [])
        evaluation_code_dirs = evaluation_config.get("code_dirs", [])

        algorithm_specs = gather(algorithm_spec_dirs, allow_extensions={".md", ".markdown", ".txt"})
        evaluation_specs = gather(evaluation_spec_dirs, allow_extensions={".md", ".markdown", ".txt"})
        algorithm_codes = gather(algorithm_code_dirs, allow_extensions={".py"})
        evaluation_codes = gather(evaluation_code_dirs, allow_extensions={".py"})

        for extra in algorithm_config.get("extra_files", []):
            algorithm_specs.extend(gather([extra]))
        for extra in evaluation_config.get("extra_files", []):
            evaluation_specs.extend(gather([extra]))

        if not (algorithm_specs and algorithm_codes and evaluation_specs and evaluation_codes):
            raise ValueError("Incomplete external resources collected for RAG initialization")

        return {
            "algorithm_specs": sorted(set(algorithm_specs)),
            "algorithm_codes": sorted(set(algorithm_codes)),
            "evaluation_specs": sorted(set(evaluation_specs)),
            "evaluation_codes": sorted(set(evaluation_codes)),
        }

    def _resolve_output_dir(self) -> Path:
        if self.output_base_dir:
            return self.output_base_dir
        default_dir = Path(
            self.settings.get("instance_analyzer.output_dir", "./outputs/instance_analysis")
        ).resolve()
        ensure_directory(str(default_dir))
        self.output_base_dir = default_dir
        return default_dir

    def _safe_int(self, value: Any) -> Optional[int]:
        """Return value cast to int when possible, otherwise None."""
        try:
            if value is None:
                return None
            if isinstance(value, float) and not float(value).is_integer():
                return None
            return int(value)
        except (ValueError, TypeError):
            return None

    def _resolve_csv_within_directory(self, relative_path: str, *, video_id: Optional[int] = None) -> str:
        """Resolve a CSV path within a directory (or direct file) relative to the database path."""
        db_path = self.settings.get('global.database_path')
        if not db_path:
            raise ValueError("global.database_path is not configured")

        root_dir = Path(db_path).resolve().parent
        target_path = (root_dir / relative_path).resolve()

        if not target_path.exists():
            raise FileNotFoundError(f"CSV path not found: {target_path}")

        if target_path.is_file():
            if target_path.suffix.lower() != '.csv':
                raise ValueError(f"Expected CSV file but found different type: {target_path}")
            return str(target_path)

        csv_files = sorted(p for p in target_path.glob('*.csv') if p.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {target_path}")

        if video_id is not None:
            target = str(video_id)
            exact = [p for p in csv_files if p.stem == target]
            if exact:
                return str(exact[0])
            prefix = [p for p in csv_files if p.stem.startswith(target + '_')]
            if prefix:
                return str(prefix[0])
            contains = [p for p in csv_files if f"_{target}_" in p.stem]
            if contains:
                return str(contains[0])

        return str(csv_files[0])
