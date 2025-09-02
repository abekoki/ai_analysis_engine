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
if datawarehouse_path.exists() and str(datawarehouse_path) not in sys.path:
    sys.path.insert(0, str(datawarehouse_path))

try:
    from datawarehouse.connection import get_connection
    from datawarehouse.algorithm_api import (
        get_algorithm_output,
        list_algorithm_outputs,
        get_latest_algorithm_version
    )
    from datawarehouse.evaluation_api import (
        list_evaluation_results,
        list_evaluation_data,
        get_evaluation_result,
    )
    from datawarehouse.core_lib_api import get_core_lib_output
    from datawarehouse.tag_api import get_video_tags
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
        self.db_path = Path(db_path)
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
            output_data = get_algorithm_output(algorithm_output_id, str(self.db_path))

            # 評価結果ファイルを読み込み（アルゴリズム出力ディレクトリ優先、無ければ評価出力をフォールバック）
            algorithm_output_dir = output_data['algorithm_output_dir']
            try:
                evaluation_df = self.load_evaluation_results(algorithm_output_dir)
            except FileNotFoundError:
                # 評価結果テーブルから該当algorithm_output_idの評価データCSVを探索
                evaluation_df = self._load_eval_data_from_evaluation_tables(algorithm_output_id)

            return {
                'metadata': output_data,
                'data': evaluation_df,
                'algorithm_output_id': algorithm_output_id
            }

        except Exception as e:
            raise RuntimeError(f"評価データの取得に失敗しました: {str(e)}")

    def get_evaluation_data_by_result_id(self, evaluation_result_id: int) -> Dict[str, Any]:
        """evaluation_result_ID を指定して評価データを取得

        Args:
            evaluation_result_id: 評価結果ID（evaluation_result_table）

        Returns:
            評価データ辞書 {metadata, data, evaluation_result_id}
        """
        try:
            # メタデータ取得
            eval_result = get_evaluation_result(evaluation_result_id, db_path=str(self.db_path))

            # 個別データの一覧を取得
            rows = list_evaluation_data(evaluation_result_id, db_path=str(self.db_path))
            if not rows:
                raise FileNotFoundError(f"evaluation_result_ID={evaluation_result_id} に紐づく評価データが見つかりません")

            # evaluation_data_path のCSVを読み込み（タスクレベル解析用）
            dataframes: List[pd.DataFrame] = []
            for row in rows:
                rel_path = row.get('evaluation_data_path')
                if not rel_path:
                    continue
                p = self.db_path.parent / rel_path
                if p.exists():
                    try:
                        df = pd.read_csv(p)
                        dataframes.append(self._normalize_schema(df))
                    except Exception:
                        continue

            if not dataframes:
                raise FileNotFoundError(f"evaluation_result_ID={evaluation_result_id} の評価CSVが見つからないか読み込めません")

            evaluation_df = pd.concat(dataframes, ignore_index=True)

            return {
                'metadata': eval_result,
                'data': evaluation_df,
                'evaluation_result_id': evaluation_result_id,
            }
        except Exception as e:
            raise RuntimeError(f"evaluation_result_IDからの評価データ取得に失敗しました: {str(e)}")

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
            df = pd.read_csv(result_path)
            return self._normalize_schema(df)
        else:
            # ディレクトリ内のファイルを検索
            csv_files = list(result_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"評価結果CSVファイルが見つかりません: {result_path}")

            # 最初のCSVファイルを読み込み
            df = pd.read_csv(csv_files[0])
            return self._normalize_schema(df)

    def _load_eval_data_from_evaluation_tables(self, algorithm_output_id: int) -> pd.DataFrame:
        """評価テーブルからalgorithm_output_idに紐づくCSVを収集して読み込み

        Args:
            algorithm_output_id: アルゴリズム出力ID

        Returns:
            連結済みの評価DataFrame
        """
        collected_paths: List[Path] = []

        # 全評価結果を走査して、該当algorithm_output_idの評価データを収集
        results = list_evaluation_results(db_path=str(self.db_path))
        for r in results:
            eval_id = r.get('evaluation_result_ID')
            try:
                rows = list_evaluation_data(eval_id, db_path=str(self.db_path))
            except Exception:
                continue
            for row in rows:
                if row.get('algorithm_output_ID') == algorithm_output_id:
                    rel_path = row.get('evaluation_data_path')
                    if not rel_path:
                        continue
                    p = self.db_path.parent / rel_path
                    if p.exists():
                        collected_paths.append(p)

        if not collected_paths:
            raise FileNotFoundError(
                f"該当algorithm_output_ID={algorithm_output_id}の評価データCSVが見つかりません"
            )

        # 読み込みと連結
        dataframes: List[pd.DataFrame] = []
        for p in collected_paths:
            try:
                df = pd.read_csv(p)
                dataframes.append(self._normalize_schema(df))
            except Exception:
                continue

        if not dataframes:
            raise FileNotFoundError(
                f"評価データCSVは検出されましたが読み込みに失敗しました: {collected_paths}"
            )

        return pd.concat(dataframes, ignore_index=True)

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """評価CSVの列を本ツールの期待スキーマに正規化する。

        期待スキーマ:
          - frame_num (int)
          - left_eye_open (float: 0.0~1.0)
          - right_eye_open (float: 0.0~1.0)
          - face_confidence (float: 0.0~1.0, 無ければ1.0)
          - is_drowsy (int: 0/1)
        """
        if df is None or df.empty:
            return df

        result = df.copy()

        # left_eye_open / right_eye_open の補完（left/right_eye_closed から生成）
        if 'left_eye_open' not in result.columns and 'left_eye_closed' in result.columns:
            result['left_eye_open'] = (~result['left_eye_closed'].astype(bool)).astype(float)
        if 'right_eye_open' not in result.columns and 'right_eye_closed' in result.columns:
            result['right_eye_open'] = (~result['right_eye_closed'].astype(bool)).astype(float)

        # face_confidence が無ければ 1.0 をデフォルト設定
        if 'face_confidence' not in result.columns:
            result['face_confidence'] = 1.0

        # 型の安定化
        for col in ['left_eye_open', 'right_eye_open', 'face_confidence']:
            if col in result.columns:
                result[col] = result[col].astype(float)

        # is_drowsy の型をintに
        if 'is_drowsy' in result.columns:
            try:
                result['is_drowsy'] = result['is_drowsy'].astype(int)
            except Exception:
                # True/False などは bool->int へ
                result['is_drowsy'] = result['is_drowsy'].astype(bool).astype(int)

        return result

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

        # JSONファイルとして保存（Python組み込み型に正規化）
        json_path = output_path / "analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._to_serializable(results), f, ensure_ascii=False, indent=2)

        # DataWareHouseに分析結果を登録
        # DataWareHouseへの登録（軽いリトライ付き）
        last_err = None
        for attempt in range(3):
            try:
                analysis_result_id = create_analysis_result(
                    analysis_result_dir=output_dir,
                    evaluation_result_id=algorithm_output_id,
                    analysis_timestamp=timestamp,
                    db_path=self.db_path
                )
                return analysis_result_id
            except Exception as e:
                last_err = e
                import time
                time.sleep(0.5 * (attempt + 1))
        # 失敗
        raise RuntimeError(f"分析結果のDataWareHouse登録に失敗しました: {last_err}")

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

    def _to_serializable(self, obj: Any) -> Any:
        """JSONシリアライズ可能なPython型に変換"""
        import numpy as np
        if isinstance(obj, dict):
            return {self._to_serializable(k): self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    def build_task_level_dataframe(self, evaluation_result_id: int) -> pd.DataFrame:
        """evaluation_result_IDに紐づくタグ単位のタスク行を構築する

        各タスク行: task_id, video_ID, tag_ID, start, end, expected_is_drowsy(=1), is_drowsy(=区間内に1があるか)
        """
        rows = list_evaluation_data(evaluation_result_id, db_path=str(self.db_path))
        tasks: List[Dict[str, Any]] = []

        # キャッシュ: algo_out_id -> concatenated frame df
        frames_cache: Dict[int, pd.DataFrame] = {}

        for r in rows:
            algo_out_id = r.get('algorithm_output_ID')
            if algo_out_id is None:
                continue
            try:
                algo_out = get_algorithm_output(algo_out_id, str(self.db_path))
                core_out_id = algo_out.get('core_lib_output_ID')
                core_out = get_core_lib_output(core_out_id, str(self.db_path))
                video_id = core_out.get('video_ID')

                # フレームCSVを読み込み（結合）
                if algo_out_id not in frames_cache:
                    rel_dir = algo_out.get('algorithm_output_dir')
                    d = self.db_path.parent / rel_dir
                    dfs: List[pd.DataFrame] = []
                    if d.exists():
                        for p in sorted(d.glob('*.csv')):
                            try:
                                dfp = pd.read_csv(p)
                                dfs.append(dfp)
                            except Exception:
                                continue
                    frames_cache[algo_out_id] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

                frame_df = frames_cache.get(algo_out_id, pd.DataFrame()).copy()
                # カラムの正規化
                if not frame_df.empty and 'frame_num' not in frame_df.columns:
                    if 'frame' in frame_df.columns:
                        frame_df = frame_df.rename(columns={'frame': 'frame_num'})

                # タグ一覧を取得
                tags = get_video_tags(video_id, db_path=str(self.db_path))
                for tg in tags:
                    start = tg.get('start', 0)
                    end = tg.get('end', -1)
                    detected = 0
                    if not frame_df.empty and {'frame_num', 'is_drowsy'}.issubset(set(frame_df.columns)):
                        seg = frame_df[(frame_df['frame_num'] >= start) & (frame_df['frame_num'] <= end)]
                        if not seg.empty:
                            detected = int((seg['is_drowsy'] == 1).any())

                    task_row = {
                        'task_id': f"{video_id}_{tg.get('tag_ID')}",
                        'video_ID': video_id,
                        'tag_ID': tg.get('tag_ID'),
                        'task_ID': tg.get('task_ID'),
                        'start': start,
                        'end': end,
                        'expected_is_drowsy': 1,
                        'is_drowsy': detected,
                    }
                    tasks.append(task_row)
            except Exception:
                continue

        return pd.DataFrame(tasks)
