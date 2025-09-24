#!/usr/bin/env python3
"""List analysis targets for a given evaluation_result_ID by traversing DB FKs.

Usage:
  python scripts/list_targets.py -e 3
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict

# DataWareHouse パッケージは pip インストール済みを前提とする
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datawarehouse.evaluation.api import list_evaluation_data
from datawarehouse.algorithm.api import get_algorithm_output
from datawarehouse.core_lib.api import get_core_lib_output
from datawarehouse.tag.api import get_video_tags

DEFAULT_DB_ROOT = PROJECT_ROOT / ".." / "development_datas"
DB_PATH = str((DEFAULT_DB_ROOT / "database.db").resolve())


def gather_targets(evaluation_result_id: int) -> List[Dict]:
    """Collect  analysis targets by FK traversal from evaluation_data_table.

    Returns list of dicts: {evaluation_data_ID, algorithm_output_ID, core_lib_output_ID,
                            video_ID, tag_ID, task_ID, start, end}
    """
    rows = list_evaluation_data(evaluation_result_id, db_path=DB_PATH)
    targets: List[Dict] = []

    for row in rows:
        evaluation_data_id = row["evaluation_data_ID"]
        algorithm_output_id = row["algorithm_output_ID"]

        # algorithm_output -> core_lib_output
        algo_out = get_algorithm_output(algorithm_output_id, DB_PATH)
        core_lib_output_id = algo_out["core_lib_output_ID"]

        # core_lib_output -> video
        core_out = get_core_lib_output(core_lib_output_id, DB_PATH)
        video_id = core_out["video_ID"]

        # video -> tags
        tags = get_video_tags(video_id, db_path=DB_PATH)

        # append all tags under this video as candidates
        for tg in tags:
            targets.append({
                "evaluation_data_ID": evaluation_data_id,
                "algorithm_output_ID": algorithm_output_id,
                "core_lib_output_ID": core_lib_output_id,
                "video_ID": video_id,
                "tag_ID": tg["tag_ID"],
                "task_ID": tg["task_ID"],
                "start": tg["start"],
                "end": tg["end"],
                "task_name": tg.get("task_name"),
            })

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="List analysis targets for evaluation_result_ID")
    parser.add_argument("--evaluation-result-id", "-e", type=int, required=True)
    args = parser.parse_args()

    targets = gather_targets(args.evaluation_result_id)

    # Print summary
    print(f"evaluation_result_ID={args.evaluation_result_id} -> {len(targets)} tags")
    # Show first 20
    for i, t in enumerate(targets[:20], 1):
        print(f"{i:02d}: eval_data_ID={t['evaluation_data_ID']}, algo_out_ID={t['algorithm_output_ID']}, "
              f"core_out_ID={t['core_lib_output_ID']}, video_ID={t['video_ID']}, tag_ID={t['tag_ID']}, "
              f"task_ID={t['task_ID']} [{t['start']}-{t['end']}], task_name={t.get('task_name')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
