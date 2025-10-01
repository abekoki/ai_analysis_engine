"""
Compatibility adapter for the legacy InstanceAnalyzer API.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config.settings import Settings
from .config import config
from .utils.file_utils import ensure_directory, ensure_analysis_output_structure, get_report_paths
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

        # データオブジェクトからCSVファイルパスを抽出
        algorithm_outputs = []
        core_outputs = []

        for algo_data in algorithm_outputs_data:
            if isinstance(algo_data, dict) and 'algorithm_output_dir' in algo_data:
                # データベースからの相対パスを絶対パスに変換
                algo_dir = algo_data['algorithm_output_dir']
                if not os.path.isabs(algo_dir):
                    algo_dir = os.path.join(self.settings.get('global.database_path', '').replace('database.db', ''), algo_dir)
                # ディレクトリ内の全CSVファイルを取得
                if os.path.exists(algo_dir):
                    for csv_file in os.listdir(algo_dir):
                        if csv_file.endswith('.csv'):
                            algorithm_outputs.append(os.path.join(algo_dir, csv_file))

        for core_data in core_outputs_data:
            if isinstance(core_data, dict) and 'core_lib_output_dir' in core_data:
                # データベースからの相対パスを絶対パスに変換
                core_dir = core_data['core_lib_output_dir']
                logger.info(f"Original core_dir: {core_dir}")
                if not os.path.isabs(core_dir):
                    # '../DataWareHouse/...' のようなパスは '../development_datas/...' に置換
                    if core_dir.startswith('../DataWareHouse/'):
                        core_dir = '../development_datas/' + core_dir[len('../DataWareHouse/'):]
                        logger.info(f"Replaced core_dir: {core_dir}")
                    else:
                        # その他の相対パスの場合はデータベースディレクトリからの相対として扱う
                        db_dir = os.path.dirname(self.settings.get('global.database_path', ''))
                        core_dir = os.path.join(db_dir, core_dir.lstrip('../'))
                    logger.info(f"Final core_dir: {core_dir}")
                # ディレクトリ内の全CSVファイルを取得
                if os.path.exists(core_dir):
                    logger.info(f"Core dir exists, contents: {os.listdir(core_dir)}")
                    for csv_file in os.listdir(core_dir):
                        if csv_file.endswith('.csv'):
                            core_outputs.append(os.path.join(core_dir, csv_file))
                else:
                    logger.warning(f"Core dir does not exist: {core_dir}")

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
                context_file = paths["contexts"] / f"graph_state_{dataset_id or 'unknown'}.json"
                try:
                    write_file(str(context_file), json.dumps(context_blob, ensure_ascii=False, indent=2))
                    self.logger.info(f"LangGraph context saved: {context_file}")
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
