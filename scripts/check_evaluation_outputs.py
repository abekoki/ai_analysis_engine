#!/usr/bin/env python3
"""Inspect evaluation outputs for a specific evaluation_result_ID.

Usage:
  uv run python scripts/check_evaluation_outputs.py --evaluation-result-id 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_analysis_engine.config.settings import Settings

try:
    from datawarehouse import (
        get_algorithm_output,
        get_core_lib_output,
        list_evaluation_data,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DataWareHouse パッケージが見つかりません。\n"
        "uv pip install git+https://github.com/abekoki/DataWareHouse@remake_pip_lib"
    ) from exc


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and not float(value).is_integer():
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_csv(settings: Settings, relative_path: str, video_id: Optional[int]) -> Dict[str, Any]:
    db_path = settings.get("global.database_path")
    if not db_path:
        return {"status": "error", "reason": "global.database_path not configured"}

    root_dir = Path(db_path).resolve().parent
    target_path = (root_dir / relative_path).resolve()

    if not target_path.exists():
        return {"status": "error", "reason": f"path not found: {target_path}"}

    if target_path.is_file():
        if target_path.suffix.lower() != ".csv":
            return {"status": "error", "reason": f"not a CSV file: {target_path}"}
        return {"status": "ok", "path": str(target_path)}

    csv_files = sorted(p for p in target_path.glob("*.csv") if p.is_file())
    if not csv_files:
        return {"status": "error", "reason": f"no CSV files in dir: {target_path}"}

    if video_id is not None:
        key = str(video_id)
        exact = [p for p in csv_files if p.stem == key]
        if exact:
            return {"status": "ok", "path": str(exact[0])}
        prefix = [p for p in csv_files if p.stem.startswith(f"{key}_")]
        if prefix:
            return {"status": "ok", "path": str(prefix[0])}
        contains = [p for p in csv_files if f"_{key}_" in p.stem]
        if contains:
            return {"status": "ok", "path": str(contains[0])}

    if len(csv_files) > 1:
        return {
            "status": "error",
            "reason": "multiple CSV candidates",
            "candidates": [str(p) for p in csv_files],
        }

    return {"status": "ok", "path": str(csv_files[0])}


def inspect_evaluation(settings: Settings, evaluation_result_id: int) -> List[Dict[str, Any]]:
    db_path = settings.get("global.database_path")
    if not db_path:
        raise ValueError("global.database_path is not configured")

    rows = list_evaluation_data(evaluation_result_id, db_path=db_path)
    results: List[Dict[str, Any]] = []

    for row in rows:
        algo_out_id = safe_int(row.get("algorithm_output_ID"))
        core_out_id = None
        algo_dir = None
        core_dir = None
        video_id = None
        algo_resolver: Dict[str, Any] = {"status": "skipped"}
        core_resolver: Dict[str, Any] = {"status": "skipped"}

        if algo_out_id is not None:
            algo_out = get_algorithm_output(algo_out_id, db_path)
            algo_dir = algo_out.get("algorithm_output_dir")
            core_out_id = safe_int(algo_out.get("core_lib_output_ID"))

        if core_out_id is not None:
            core_out = get_core_lib_output(core_out_id, db_path)
            core_dir = core_out.get("core_lib_output_dir")
            video_id = safe_int(core_out.get("video_ID"))

        if algo_dir:
            algo_resolver = resolve_csv(settings, algo_dir, video_id)
        if core_dir:
            core_resolver = resolve_csv(settings, core_dir, video_id)

        results.append({
            "evaluation_data_ID": row.get("evaluation_data_ID"),
            "algorithm_output_ID": algo_out_id,
            "algorithm_output_dir": algo_dir,
            "core_lib_output_ID": core_out_id,
            "core_lib_output_dir": core_dir,
            "video_ID": video_id,
            "algorithm_csv": algo_resolver,
            "core_csv": core_resolver,
        })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check evaluation outputs and CSV availability")
    parser.add_argument("--evaluation-result-id", "-e", type=int, default=3)
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional path to save JSON result")
    args = parser.parse_args()

    settings = Settings()
    results = inspect_evaluation(settings, args.evaluation_result_id)

    report = {
        "evaluation_result_id": args.evaluation_result_id,
        "total_rows": len(results),
        "summary": {
            "algorithm_ok": sum(1 for r in results if r["algorithm_csv"].get("status") == "ok"),
            "core_ok": sum(1 for r in results if r["core_csv"].get("status") == "ok"),
        },
        "details": results,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
