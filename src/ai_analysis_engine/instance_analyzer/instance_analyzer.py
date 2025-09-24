"""
Compatibility adapter for the legacy InstanceAnalyzer API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config.settings import Settings
from ..utils.file_utils import ensure_directory
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
        ensure_directory(str(self.output_base_dir))

    def analyze_instances(self, evaluation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("Starting legacy instance analysis via adapter")
        self._ensure_engine_initialized()
        output_dir = self._resolve_output_dir()

        algorithm_outputs = evaluation_data.get("algorithm_outputs", [])
        core_outputs = evaluation_data.get("core_outputs", [])
        expected_results = evaluation_data.get("expected_results", [])
        dataset_ids = evaluation_data.get("dataset_ids")

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

        results = self._engine.analyze(
            algorithm_outputs=algorithm_outputs,
            core_outputs=core_outputs,
            expected_results=expected_results,
            output_dir=str(output_dir),
            dataset_ids=dataset_ids_resolved,
        )

        return [result.model_dump() for result in results]

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
