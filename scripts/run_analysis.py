#!/usr/bin/env python3
"""AI分析エンジン実行スクリプト

このスクリプトは、AI分析エンジンを実行するためのエントリーポイントです。
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートとDataWareHouseをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# src をパスに追加（ローカル実行でのimport安定化）
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# DataWareHouse APIをパスに追加
datawarehouse_path = project_root.parent / "DataWareHouse"
if datawarehouse_path.exists():
    sys.path.insert(0, str(datawarehouse_path))

from ai_analysis_engine.orchestrator.orchestrator import Orchestrator
from ai_analysis_engine.config.settings import Settings


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='AI分析エンジン実行スクリプト')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='設定ファイルのパス')
    parser.add_argument('--algorithm-output-id', '-a', type=int, default=None,
                       help='分析対象のアルゴリズム出力ID')
    parser.add_argument('--evaluation-result-id', '-e', type=int, default=None,
                       help='分析対象の評価結果ID (evaluation_result_table)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細なログ出力')

    args = parser.parse_args()

    try:
        # 設定の読み込み
        settings = Settings(args.config)

        # Orchestratorの初期化
        orchestrator = Orchestrator(settings)

        # 分析実行
        print("AI分析エンジンを開始します...")
        # 優先度: evaluation_result_id が指定されていればそれを使用
        if args.evaluation_result_id is not None:
            result = orchestrator.run_analysis_by_evaluation_result(args.evaluation_result_id)
        else:
            result = orchestrator.run_analysis(args.algorithm_output_id)

        if result['status'] == 'success':
            print("✅ 分析が正常に完了しました")
            print(f"📄 レポート: {result.get('report_path', 'N/A')}")
            print(f"📊 正解率: {result.get('integrated_results', {}).get('performance_summary', {}).get('accuracy', 'N/A')}")
        else:
            print("❌ 分析に失敗しました")
            print(f"エラー: {result.get('error', '不明なエラー')}")

        return 0 if result['status'] == 'success' else 1

    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
