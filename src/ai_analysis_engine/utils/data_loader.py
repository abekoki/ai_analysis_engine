"""DataWareHouse連携ユーティリティ

このモジュールは、DataWareHouseのAPIを使用してデータベースとの連携機能を
提供します。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import json

# DataWareHouseのAPIをインポート
datawarehouse_path = Path(__file__).parent.parent.parent.parent / "DataWareHouse"
if str(datawarehouse_path) not in sys.path:
    sys.path.insert(0, str(datawarehouse_path))

try:
    from datawarehouse.connection import get_connection
    from datawarehouse.algorithm_api import (
        get_algorithm_output,
        list_algorithm_outputs,
        get_latest_algorithm_version
    )
    from datawarehouse.analysis_api import (
        create_analysis_result,
        get_analysis_result,
        list_analysis_results,
        create_problem,
        create_analysis_data
    )
    from datawarehouse.exceptions import (
        DWHError,
        DWHNotFoundError,
        DWHConstraintError,
        DWHValidationError
    )
    DWH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DataWareHouse API not available: {e}")
    DWH_AVAILABLE = False


class DataWareHouseConnector:
    """DataWareHouse API接続クラス"""

    def __init__(self, db_path: str):
        """
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = str(Path(db_path))
        if not DWH_AVAILABLE:
            raise RuntimeError("DataWareHouse APIが利用できません。DataWareHouseが正しくインストールされているか確認してください。")



    def get_evaluation_data(self, algorithm_output_id: Optional[int] = None) -> Dict[str, Any]:
        """評価データを取得

        Args:
            algorithm_output_id: 特定のアルゴリズム出力ID（Noneの場合は最新）

        Returns:
            評価データ辞書
        """
        try:
            # アルゴリズム出力IDが指定されていない場合は最新を取得
            if algorithm_output_id is None:
                latest_version = get_latest_algorithm_version(self.db_path)
                if latest_version is None:
                    raise ValueError("アルゴリズムバージョンが見つかりません")

                # 最新バージョンの出力を取得
                outputs = list_algorithm_outputs(
                    algorithm_id=latest_version['algorithm_ID'],
                    db_path=self.db_path
                )
                if not outputs:
                    raise ValueError("アルゴリズム出力データが見つかりません")

                algorithm_output_id = outputs[-1]['algorithm_output_ID']  # 最新の出力

            # 指定されたアルゴリズム出力を取得
            output_data = get_algorithm_output(algorithm_output_id, self.db_path)

            # 評価結果ファイルを読み込み
            algorithm_output_dir = output_data['algorithm_output_dir']
            evaluation_df = self.load_evaluation_results(algorithm_output_dir)

            return {
                'metadata': output_data,
                'data': evaluation_df,
                'algorithm_output_id': algorithm_output_id
            }

        except Exception as e:
            raise RuntimeError(f"評価データの取得に失敗しました: {str(e)}")

    def load_evaluation_results(self, algorithm_output_dir: str) -> pd.DataFrame:
        """評価結果ファイルを読み込み

        Args:
            algorithm_output_dir: アルゴリズム出力ディレクトリ

        Returns:
            評価結果（DataFrame）
        """
        # DataWareHouseからの相対パスを解決
        result_path = self.db_path.parent / algorithm_output_dir

        if not result_path.exists():
            raise FileNotFoundError(f"評価結果ファイルが見つかりません: {result_path}")

        # CSVファイルとして読み込み（実際のフォーマットに合わせて調整）
        if result_path.is_file():
            return pd.read_csv(result_path)
        else:
            # ディレクトリ内のファイルを検索
            csv_files = list(result_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"評価結果CSVファイルが見つかりません: {result_path}")

            # 最初のCSVファイルを読み込み
            return pd.read_csv(csv_files[0])

    def save_analysis_results(self, results: Dict[str, Any],
                            algorithm_output_id: int,
                            timestamp: Optional[str] = None) -> int:
        """分析結果を保存

        Args:
            results: 分析結果
            algorithm_output_id: 元のアルゴリズム出力ID
            timestamp: タイムスタンプ（Noneの場合は現在時刻）

        Returns:
            作成された分析結果ID
        """
        # 出力ディレクトリを生成（相対パス）
        output_dir = f"05_analysis_output/analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        # ローカルファイルシステムに保存
        output_path = Path(self.db_path).parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # JSONファイルとして保存
        json_path = output_path / "analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # DataWareHouseに分析結果を登録
        try:
            analysis_result_id = create_analysis_result(
                analysis_result_dir=output_dir,
                evaluation_result_id=algorithm_output_id,
                analysis_timestamp=timestamp,
                db_path=self.db_path
            )
            return analysis_result_id
        except Exception as e:
            # DataWareHouseへの登録が失敗してもファイルは保存されている
            raise RuntimeError(f"分析結果のDataWareHouse登録に失敗しました: {str(e)}")

    def create_problem_record(self, problem_name: str, problem_description: str,
                            problem_status: str, analysis_result_id: int) -> int:
        """問題点を登録

        Args:
            problem_name: 問題点名
            problem_description: 問題点の説明
            problem_status: 問題点のステータス
            analysis_result_id: 分析結果ID

        Returns:
            作成された問題点ID
        """
        try:
            return create_problem(
                problem_name=problem_name,
                problem_description=problem_description,
                problem_status=problem_status,
                analysis_result_id=analysis_result_id,
                db_path=self.db_path
            )
        except Exception as e:
            raise RuntimeError(f"問題点の登録に失敗しました: {str(e)}")

    def create_analysis_data_record(self, evaluation_data_id: int,
                                  analysis_result_id: int,
                                  is_problem: int,
                                  data_dir: str,
                                  description: str,
                                  problem_id: Optional[int] = None) -> int:
        """分析データを登録

        Args:
            evaluation_data_id: 評価データID
            analysis_result_id: 分析結果ID
            is_problem: 問題点かどうか（0: 問題なし, 1: 問題あり）
            data_dir: データディレクトリ
            description: データの説明
            problem_id: 問題点ID（is_problem=1の場合必須）

        Returns:
            作成された分析データID
        """
        try:
            return create_analysis_data(
                evaluation_data_id=evaluation_data_id,
                analysis_result_id=analysis_result_id,
                analysis_data_isproblem=is_problem,
                analysis_data_dir=data_dir,
                analysis_data_description=description,
                problem_id=problem_id,
                db_path=self.db_path
            )
        except Exception as e:
            raise RuntimeError(f"分析データの登録に失敗しました: {str(e)}")
