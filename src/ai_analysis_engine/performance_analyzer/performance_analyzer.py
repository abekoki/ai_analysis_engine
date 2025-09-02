"""PerformanceAnalyzerクラス

全体性能の確認・差分分析を行うクラスです。
評価データセット全体の指標を集計し、ベースラインとの比較を行います。
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..config.settings import Settings
from ..utils.expectation_generator import ExpectationGenerator


class PerformanceAnalyzer:
    """全体性能分析クラス"""

    def __init__(self, settings: Settings):
        """
        Args:
            settings: 設定オブジェクト
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # 期待値生成器の初期化
        drowsy_config = settings.get('instance_analyzer.drowsy_detection', {})
        self.expectation_generator = ExpectationGenerator(drowsy_config)

        # 設定値の取得
        self.metrics = settings.get('performance_analyzer.metrics', ['accuracy'])
        self.visualization_level = settings.get('performance_analyzer.visualization_level', 'standard')

        self.logger.info("PerformanceAnalyzer initialized")

    def analyze_performance(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """性能分析を実行

        Args:
            evaluation_data: 評価データ（metadataとdataを含む）

        Returns:
            分析結果辞書
        """
        try:
            self.logger.info("Starting performance analysis")

            # データの取得
            df = evaluation_data.get('data')
            metadata = evaluation_data.get('metadata', {})

            if df is None or df.empty:
                raise ValueError("評価データが空です")

            # カラムによって処理を分岐（フレームレベル or タスクレベル）
            required_cols = {'left_eye_open', 'right_eye_open', 'face_confidence'}
            if required_cols.issubset(set(df.columns)):
                # フレームレベル
                df_with_expectations = self.expectation_generator.detect_continuous_closure(df)
                metrics_results = self._calculate_metrics(df_with_expectations)
            else:
                # タスクレベル: 期待値生成なしで基本指標のみ
                df_with_expectations = df.copy()
                metrics_results = {'total_samples': len(df_with_expectations)}
                if {'expected_is_drowsy', 'is_drowsy'}.issubset(set(df_with_expectations.columns)):
                    acc = (df_with_expectations['expected_is_drowsy'] == df_with_expectations['is_drowsy']).mean()
                    metrics_results.update({
                        'accuracy': float(acc),
                        'correct_predictions': int((df_with_expectations['expected_is_drowsy'] == df_with_expectations['is_drowsy']).sum()),
                        'incorrect_predictions': int((df_with_expectations['expected_is_drowsy'] != df_with_expectations['is_drowsy']).sum()),
                    })

            # ベースライン比較
            baseline_comparison = self._compare_with_baseline(metrics_results)

            # 可視化の生成
            visualization_paths = self._generate_visualizations(df_with_expectations, metrics_results)

            # 結果の統合
            result = {
                'summary': metrics_results,
                'baseline_comparison': baseline_comparison,
                'visualizations': visualization_paths,
                'metadata': metadata,
                'data_shape': df.shape,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }

            self.logger.info(f"Performance analysis completed: {metrics_results}")
            return result

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}", exc_info=True)
            raise

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """指標を計算

        Args:
            df: 期待値が追加されたDataFrame

        Returns:
            指標結果辞書
        """
        metrics = {}

        # 正解率の計算
        if 'accuracy' in self.metrics:
            accuracy_metrics = self.expectation_generator.calculate_accuracy_metrics(df)
            metrics.update(accuracy_metrics)

        # 過検知数の計算（1時間あたりの過検知数）
        if 'over_detection_count_per_hour' in self.metrics:
            over_detection_count = self._calculate_over_detection_rate(df)
            metrics['over_detection_count_per_hour'] = over_detection_count

        # その他の指標
        metrics.update(self._calculate_additional_metrics(df))

        return metrics

    def _calculate_over_detection_rate(self, df: pd.DataFrame) -> float:
        """過検知率を計算

        Args:
            df: 評価データ

        Returns:
            1時間あたりの過検知数
        """
        # 過検知: expected_is_drowsy = 0 かつ is_drowsy = 1
        false_positives = ((df['expected_is_drowsy'] == 0) & (df['is_drowsy'] == 1)).sum()

        # フレームレートを仮定（30fps）
        fps = 30.0

        # データの総時間を計算（秒）
        total_frames = len(df)
        total_time_hours = total_frames / fps / 3600

        # 1時間あたりの過検知数
        if total_time_hours > 0:
            over_detection_rate = false_positives / total_time_hours
        else:
            over_detection_rate = 0.0

        return over_detection_rate

    def _calculate_additional_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """追加指標を計算

        Args:
            df: 評価データ

        Returns:
            追加指標辞書
        """
        additional_metrics = {}

        # 検知された連続閉眼の平均時間
        drowsy_periods = df[df['is_drowsy'] == 1]
        if not drowsy_periods.empty:
            avg_continuous_time = drowsy_periods['continuous_time'].mean()
            additional_metrics['avg_continuous_drowsy_time'] = avg_continuous_time

        # 総閉眼時間
        total_drowsy_time = drowsy_periods['continuous_time'].sum() if not drowsy_periods.empty else 0
        additional_metrics['total_drowsy_time_seconds'] = total_drowsy_time

        # 信頼度分布
        if 'face_confidence' in df.columns:
            avg_confidence = df['face_confidence'].mean()
            min_confidence = df['face_confidence'].min()
            additional_metrics['avg_face_confidence'] = avg_confidence
            additional_metrics['min_face_confidence'] = min_confidence

        return additional_metrics

    def _compare_with_baseline(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """ベースラインとの比較

        Args:
            current_metrics: 現在の指標

        Returns:
            比較結果辞書
        """
        # TODO: 実際のベースライン取得を実装
        # ここでは仮のベースラインを使用
        baseline_metrics = {
            'accuracy': 0.85,
            'over_detection_count_per_hour': 2.5,
            'avg_face_confidence': 0.88
        }

        comparison = {}
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric, 0)
            diff = current_value - baseline_value if isinstance(current_value, (int, float)) else 0
            comparison[metric] = {
                'current': current_value,
                'baseline': baseline_value,
                'difference': diff,
                'improvement': diff > 0 if metric in ['accuracy', 'avg_face_confidence'] else diff < 0
            }

        return comparison

    def _generate_visualizations(self, df: pd.DataFrame,
                               metrics: Dict[str, Any]) -> Dict[str, str]:
        """可視化を生成

        Args:
            df: 評価データ
            metrics: 指標結果

        Returns:
            可視化ファイルパスの辞書
        """
        visualization_paths = {}

        try:
            # 出力ディレクトリの作成
            charts_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            # 時系列グラフ（frame_numがある場合のみ）
            if self.visualization_level in ['standard', 'detailed'] and 'frame_num' in df.columns:
                time_series_path = self._create_time_series_plot(df, charts_dir)
                visualization_paths['time_series'] = str(time_series_path)

            # 混同行列
            if self.visualization_level == 'detailed':
                confusion_matrix_path = self._create_confusion_matrix_plot(df, charts_dir)
                visualization_paths['confusion_matrix'] = str(confusion_matrix_path)

            # 性能比較チャート
            performance_path = self._create_performance_comparison_plot(metrics, charts_dir)
            visualization_paths['performance_comparison'] = str(performance_path)

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {str(e)}")

        return visualization_paths

    def _create_time_series_plot(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """時系列グラフを作成

        Args:
            df: 評価データ
            output_dir: 出力ディレクトリ

        Returns:
            グラフファイルのパス
        """
        plt.figure(figsize=(12, 8))

        # 左右目の開眼度
        plt.subplot(3, 1, 1)
        plt.plot(df['frame_num'], df['left_eye_open'], label='Left Eye', alpha=0.7)
        plt.plot(df['frame_num'], df['right_eye_open'], label='Right Eye', alpha=0.7)
        plt.axhline(y=self.expectation_generator.left_eye_threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('Eye Openness Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Eye Openness')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 顔検出信頼度
        plt.subplot(3, 1, 2)
        plt.plot(df['frame_num'], df['face_confidence'], color='green', alpha=0.7)
        plt.axhline(y=self.expectation_generator.face_conf_threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('Face Detection Confidence')
        plt.xlabel('Frame Number')
        plt.ylabel('Confidence')
        plt.grid(True, alpha=0.3)

        # 検知結果
        plt.subplot(3, 1, 3)
        plt.plot(df['frame_num'], df['is_drowsy'], label='Detected', alpha=0.7)
        plt.plot(df['frame_num'], df['expected_is_drowsy'], label='Expected', alpha=0.7, linestyle='--')
        plt.title('Drowsiness Detection Results')
        plt.xlabel('Frame Number')
        plt.ylabel('Drowsiness (0/1)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        output_path = output_dir / "time_series.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_confusion_matrix_plot(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """混同行列を作成

        Args:
            df: 評価データ
            output_dir: 出力ディレクトリ

        Returns:
            グラフファイルのパス
        """
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        # エラー状態を除外
        valid_df = df[df['expected_is_drowsy'] != -1]

        if valid_df.empty:
            # 空のプロットを作成
            plt.figure(figsize=(6, 6))
            plt.text(0.5, 0.5, 'No valid data for confusion matrix', ha='center', va='center')
            plt.title('Confusion Matrix - No Data')
        else:
            cm = confusion_matrix(valid_df['expected_is_drowsy'], valid_df['is_drowsy'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Drowsy', 'Drowsy'])
            disp.plot(cmap='Blues')
            plt.title('Confusion Matrix')

        output_path = output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _create_performance_comparison_plot(self, metrics: Dict[str, Any], output_dir: Path) -> Path:
        """性能比較チャートを作成

        Args:
            metrics: 指標結果
            output_dir: 出力ディレクトリ

        Returns:
            グラフファイルのパス
        """
        plt.figure(figsize=(10, 6))

        # 主要指標の棒グラフ
        metric_names = ['accuracy', 'over_detection_count_per_hour', 'avg_face_confidence']
        values = []

        for metric in metric_names:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float)):
                    values.append((metric, value))
                else:
                    values.append((metric, 0))
            else:
                values.append((metric, 0))

        if values:
            names, vals = zip(*values)
            bars = plt.bar(names, vals)
            plt.title('Performance Metrics')
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')

            # 値のラベルを追加
            for bar, val in zip(bars, vals):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        '.3f', ha='center', va='bottom')

        plt.tight_layout()

        output_path = output_dir / "performance_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path
