"""期待値生成ユーティリティ

このモジュールは、drowsy_detectionアルゴリズムの仕様に基づいて
評価結果から期待値を内部生成します。
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class ExpectationGenerator:
    """期待値生成クラス"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: drowsy_detectionアルゴリズムの設定
        """
        self.left_eye_threshold = config.get('left_eye_close_threshold', 0.10)
        self.right_eye_threshold = config.get('right_eye_close_threshold', 0.10)
        self.continuous_close_time = config.get('continuous_close_time', 1.00)
        self.face_conf_threshold = config.get('face_conf_threshold', 0.75)

    def generate_expectation(self, frame_data: Dict[str, Any]) -> int:
        """単一フレームの期待値を生成

        Args:
            frame_data: フレームデータ（frame_num, left_eye_open, right_eye_open, face_confidence）

        Returns:
            期待値（0: 非連続閉眼, 1: 連続閉眼）
        """
        frame_num = frame_data.get('frame_num', 0)
        left_eye_open = frame_data.get('left_eye_open', 1.0)
        right_eye_open = frame_data.get('right_eye_open', 1.0)
        face_confidence = frame_data.get('face_confidence', 1.0)

        # 顔検出信頼度が閾値未満の場合は無効
        if face_confidence < self.face_conf_threshold:
            return -1  # エラー状態

        # 左右両方の目が閉眼状態かチェック
        left_closed = left_eye_open <= self.left_eye_threshold
        right_closed = right_eye_open <= self.right_eye_threshold

        # 両目が閉眼していない場合は非連続閉眼
        if not (left_closed and right_closed):
            return 0

        # 両目が閉眼している場合、ここでは簡易的に1を返す
        # （実際の連続判定は時系列データが必要）
        return 1

    def generate_expectations_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameから期待値を生成

        Args:
            df: 評価データ（frame_num, left_eye_open, right_eye_open, face_confidence列を含む）

        Returns:
            期待値が追加されたDataFrame
        """
        if df.empty:
            return df.copy()

        # 必要な列が存在するかチェック
        required_columns = ['frame_num', 'left_eye_open', 'right_eye_open', 'face_confidence']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"必要な列が不足しています: {missing_columns}")

        # コピーを作成
        result_df = df.copy()

        # 各行に対して期待値を計算
        expectations = []
        for _, row in result_df.iterrows():
            frame_data = {
                'frame_num': row['frame_num'],
                'left_eye_open': row['left_eye_open'],
                'right_eye_open': row['right_eye_open'],
                'face_confidence': row['face_confidence']
            }
            expectation = self.generate_expectation(frame_data)
            expectations.append(expectation)

        result_df['expected_is_drowsy'] = expectations

        return result_df

    def detect_continuous_closure(self, df: pd.DataFrame,
                                fps: float = 30.0) -> pd.DataFrame:
        """連続閉眼を時系列で検知

        Args:
            df: 評価データ（frame_num, left_eye_open, right_eye_open, face_confidence列を含む）
            fps: フレームレート（デフォルト: 30.0）

        Returns:
            連続閉眼判定が追加されたDataFrame
        """
        if df.empty:
            return df.copy()

        result_df = df.copy()
        result_df = result_df.sort_values('frame_num').reset_index(drop=True)

        # 連続閉眼カウンタ
        continuous_count = 0
        continuous_frames = []

        for i, row in result_df.iterrows():
            frame_data = {
                'frame_num': row['frame_num'],
                'left_eye_open': row['left_eye_open'],
                'right_eye_open': row['right_eye_open'],
                'face_confidence': row['face_confidence']
            }

            expectation = self.generate_expectation(frame_data)

            if expectation == 1:  # 両目閉眼
                continuous_count += 1
                continuous_frames.append(row['frame_num'])
            else:
                continuous_count = 0
                continuous_frames = []

            # 連続閉眼時間が閾値を超えているかチェック
            continuous_time = continuous_count / fps
            is_continuous_drowsy = 1 if continuous_time >= self.continuous_close_time else 0

            result_df.at[i, 'expected_is_drowsy'] = is_continuous_drowsy
            result_df.at[i, 'continuous_count'] = continuous_count
            result_df.at[i, 'continuous_time'] = continuous_time

        return result_df

    def calculate_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """正解率などの指標を計算

        Args:
            df: expected_is_drowsyと実際のis_drowsyを含むDataFrame

        Returns:
            指標辞書
        """
        if df.empty:
            return {}

        # 必要な列が存在するかチェック
        if 'expected_is_drowsy' not in df.columns or 'is_drowsy' not in df.columns:
            raise ValueError("expected_is_drowsyまたはis_drowsy列が不足しています")

        # エラー状態（-1）を除外
        valid_df = df[df['expected_is_drowsy'] != -1].copy()

        if valid_df.empty:
            return {'accuracy': 0.0, 'total_samples': 0}

        # 正解数をカウント
        correct = (valid_df['expected_is_drowsy'] == valid_df['is_drowsy']).sum()
        total = len(valid_df)

        accuracy = correct / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'incorrect_predictions': total - correct
        }
