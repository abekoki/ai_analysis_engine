"""InstanceAnalyzerクラス

個別データの分析を行うクラスです。
仕様書・ソースコードを参照しつつ、仮説生成・検証を行います。
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path

from ..config.settings import Settings
from ..utils.expectation_generator import ExpectationGenerator
from ..utils.data_loader import DataWareHouseConnector


class InstanceAnalyzer:
    """個別データ分析クラス"""

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

        # DataWareHouse接続の初期化
        db_path = settings.get('global.database_path')
        self.datawarehouse = DataWareHouseConnector(db_path)

        # 設定値の取得
        self.max_attempts = settings.get('instance_analyzer.max_hypothesis_attempts', 3)
        self.llm_model = settings.get('instance_analyzer.llm_model', 'gpt-4')
        self.temperature = settings.get('instance_analyzer.temperature', 0.1)

        self.logger.info("InstanceAnalyzer initialized")

    def analyze_instances(self, evaluation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """個別データ分析を実行

        Args:
            evaluation_data: 評価データ

        Returns:
            個別分析結果のリスト
        """
        try:
            self.logger.info("Starting instance analysis")

            df = evaluation_data.get('data')
            if df is None or df.empty:
                return []

            # データの分割（フレーム単位または時間単位でグループ化）
            instance_groups = self._split_into_instances(df)

            results = []
            for i, (instance_id, instance_df) in enumerate(instance_groups.items()):
                self.logger.info(f"Analyzing instance {i+1}/{len(instance_groups)}: {instance_id}")

                try:
                    result = self._analyze_single_instance(instance_id, instance_df, evaluation_data)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to analyze instance {instance_id}: {str(e)}")
                    # エラーが発生しても処理を継続
                    results.append({
                        'instance_id': instance_id,
                        'status': 'error',
                        'error': str(e),
                        'summary': {'has_errors': True}
                    })

            self.logger.info(f"Instance analysis completed: {len(results)} instances processed")
            return results

        except Exception as e:
            self.logger.error(f"Instance analysis failed: {str(e)}", exc_info=True)
            raise

    def _split_into_instances(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """データを個別インスタンスに分割

        Args:
            df: 評価データ

        Returns:
            インスタンスIDをキーとしたDataFrame辞書
        """
        instances: Dict[str, pd.DataFrame] = {}

        # タスクレベル（frame_numが無い）: 1行=1タスクとして扱う or task_idでグルーピング
        if 'frame_num' not in df.columns:
            if 'task_id' in df.columns:
                for task_value, g in df.groupby('task_id'):
                    instance_id = f"task_{str(task_value)}"
                    instances[instance_id] = g.reset_index(drop=True)
            else:
                for idx, row in df.reset_index(drop=True).iterrows():
                    instance_id = f"task_{idx:03d}"
                    instances[instance_id] = pd.DataFrame([row])
            return instances

        # フレームレベル: 連続したフレームを1つのインスタンスにまとめる（固定長分割）
        current_instance = []
        instance_counter = 0

        df_sorted = df.sort_values('frame_num').reset_index(drop=True)
        for _, row in df_sorted.iterrows():
            current_instance.append(row)
            if len(current_instance) >= 100:
                instance_df = pd.DataFrame(current_instance)
                instance_id = f"instance_{instance_counter:03d}"
                instances[instance_id] = instance_df
                current_instance = []
                instance_counter += 1

        if current_instance:
            instance_df = pd.DataFrame(current_instance)
            instance_id = f"instance_{instance_counter:03d}"
            instances[instance_id] = instance_df

        return instances

    def _analyze_single_instance(self, instance_id: str, instance_df: pd.DataFrame,
                               evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """単一インスタンスの分析を実行

        Args:
            instance_id: インスタンスID
            instance_df: インスタンスのDataFrame
            evaluation_data: 全体評価データ

        Returns:
            分析結果辞書
        """
        try:
            # 期待値の生成（フレーム列がある場合のみ）。タスクレベルはそのまま扱う
            required_cols = {'left_eye_open', 'right_eye_open', 'face_confidence'}
            if required_cols.issubset(set(instance_df.columns)):
                df_with_expectations = self.expectation_generator.detect_continuous_closure(instance_df)
            else:
                df_with_expectations = instance_df.copy()
                if 'expected_is_drowsy' not in df_with_expectations.columns:
                    df_with_expectations['expected_is_drowsy'] = 1

            # 仮説生成と検証
            hypothesis_results = self._generate_and_verify_hypothesis(df_with_expectations, evaluation_data)

            # 異常検知（列が足りない場合はスキップ）
            try:
                anomalies = self._detect_anomalies(df_with_expectations)
            except Exception:
                anomalies = []

            # レポート生成
            instance_report = self._generate_instance_report(instance_id, df_with_expectations,
                                                          hypothesis_results, anomalies)

            result = {
                'instance_id': instance_id,
                'status': 'success',
                'data_shape': instance_df.shape,
                'hypothesis_results': hypothesis_results,
                'anomalies': anomalies,
                'report': instance_report,
                'summary': self._generate_instance_summary(df_with_expectations, anomalies)
            }

            return result

        except Exception as e:
            self.logger.error(f"Single instance analysis failed for {instance_id}: {str(e)}")
            raise

    def _generate_and_verify_hypothesis(self, df: pd.DataFrame,
                                       evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """仮説の生成と検証を実行

        Args:
            df: 期待値付きDataFrame
            evaluation_data: 全体評価データ

        Returns:
            仮説検証結果
        """
        # 簡易的な仮説生成（実際にはLLMやルールベースの仮説生成を実装）
        hypothesis = self._generate_basic_hypothesis(df)

        # 仮説の検証
        verification = self._verify_hypothesis(hypothesis, df)

        return {
            'hypothesis': hypothesis,
            'verification': verification,
            'confidence': verification.get('confidence', 0.5)
        }

    def _generate_basic_hypothesis(self, df: pd.DataFrame) -> str:
        """基本的な仮説を生成

        Args:
            df: 評価データ

        Returns:
            仮説文字列
        """
        # 誤検知の分析
        false_positives = ((df['expected_is_drowsy'] == 0) & (df['is_drowsy'] == 1)).sum()
        false_negatives = ((df['expected_is_drowsy'] == 1) & (df['is_drowsy'] == 0)).sum()

        if false_positives > false_negatives:
            return "過検知が多い可能性があります。閾値設定の見直しが必要です。"
        elif false_negatives > false_positives:
            return "未検知が多い可能性があります。アルゴリズムの感度向上が必要です。"
        else:
            return "検知精度は比較的良好ですが、さらなる最適化の余地があります。"

    def _verify_hypothesis(self, hypothesis: str, df: pd.DataFrame) -> Dict[str, Any]:
        """仮説の検証

        Args:
            hypothesis: 仮説文字列
            df: 評価データ

        Returns:
            検証結果
        """
        # 簡易的な検証（実際にはより詳細な検証ロジックを実装）
        accuracy_metrics = self.expectation_generator.calculate_accuracy_metrics(df)
        accuracy = accuracy_metrics.get('accuracy', 0)

        if accuracy < 0.7:
            confidence = 0.8  # 低精度の場合、仮説の信頼性が高い
            valid = True
        elif accuracy > 0.9:
            confidence = 0.3  # 高精度の場合、仮説の信頼性が低い
            valid = False
        else:
            confidence = 0.6
            valid = True

        return {
            'valid': valid,
            'confidence': confidence,
            'accuracy': accuracy,
            'reason': f"Accuracy: {accuracy:.1%}"
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """異常を検知

        Args:
            df: 評価データ

        Returns:
            異常リスト
        """
        anomalies = []

        # 誤検知の検知
        false_positives = df[(df['expected_is_drowsy'] == 0) & (df['is_drowsy'] == 1)]
        if not false_positives.empty:
            anomalies.append({
                'type': 'false_positive',
                'description': '過検知が発生しています',
                'count': len(false_positives),
                'frames': false_positives['frame_num'].tolist()[:10],  # 最初の10件
                'severity': 'medium' if len(false_positives) < 50 else 'high'
            })

        # 未検知の検知
        false_negatives = df[(df['expected_is_drowsy'] == 1) & (df['is_drowsy'] == 0)]
        if not false_negatives.empty:
            anomalies.append({
                'type': 'false_negative',
                'description': '未検知が発生しています',
                'count': len(false_negatives),
                'frames': false_negatives['frame_num'].tolist()[:10],  # 最初の10件
                'severity': 'high' if len(false_negatives) > 10 else 'medium'
            })

        # 低信頼度の検知
        low_confidence = df[df['face_confidence'] < self.expectation_generator.face_conf_threshold * 0.8]
        if not low_confidence.empty:
            anomalies.append({
                'type': 'low_confidence',
                'description': '顔検出信頼度が低いフレームが存在します',
                'count': len(low_confidence),
                'avg_confidence': low_confidence['face_confidence'].mean(),
                'severity': 'low'
            })

        return anomalies

    def _generate_instance_report(self, instance_id: str, df: pd.DataFrame,
                               hypothesis_results: Dict[str, Any],
                               anomalies: List[Dict[str, Any]]) -> str:
        """インスタンスレポートを生成

        Args:
            instance_id: インスタンスID
            df: 評価データ
            hypothesis_results: 仮説結果
            anomalies: 異常リスト

        Returns:
            マークダウンレポート文字列
        """
        # 基本統計
        total_frames = len(df)
        drowsy_frames = int((df['is_drowsy'] == 1).sum()) if 'is_drowsy' in df.columns else 0
        expected_drowsy_frames = int((df['expected_is_drowsy'] == 1).sum()) if 'expected_is_drowsy' in df.columns else 0

        # 図表の生成（タスクレベルの場合は簡易バー、フレームレベルの場合は連続時間を利用）
        chart_path = self._create_instance_chart(instance_id, df)

        report = f"""# 個別データ分析レポート - {instance_id}

## 概要
- 総フレーム数: {total_frames}
- 検知された居眠りフレーム数: {drowsy_frames}
- 期待される居眠りフレーム数: {expected_drowsy_frames}
 - 図表: {chart_path if chart_path else 'N/A'}

## 仮説分析
**仮説**: {hypothesis_results.get('hypothesis', 'N/A')}
**検証結果**: {'有効' if hypothesis_results.get('verification', {}).get('valid', False) else '無効'}
**信頼度**: {hypothesis_results.get('confidence', 0):.1%}
**正解率**: {hypothesis_results.get('verification', {}).get('accuracy', 0):.1%}

## 検知された異常
"""

        if anomalies:
            for i, anomaly in enumerate(anomalies, 1):
                report += f"""
### 異常{i}: {anomaly['type']}
- **説明**: {anomaly['description']}
- **件数**: {anomaly['count']}
- **深刻度**: {anomaly['severity']}
"""
                if 'frames' in anomaly:
                    report += f"- **対象フレーム**: {anomaly['frames'][:5]}..."
        else:
            report += "\n異常は検知されませんでした。\n"

        report += "\n## 推奨事項\n"

        # 推奨事項の生成
        if anomalies:
            high_severity = [a for a in anomalies if a.get('severity') == 'high']
            if high_severity:
                report += "- 深刻度の高い異常が検知されました。即時の対応を推奨します。\n"

            false_positives = [a for a in anomalies if a.get('type') == 'false_positive']
            if false_positives:
                report += "- 過検知を減らすため、閾値の見直しを検討してください。\n"

            false_negatives = [a for a in anomalies if a.get('type') == 'false_negative']
            if false_negatives:
                report += "- 未検知を減らすため、アルゴリズムの感度向上を検討してください。\n"

        # レポートをファイルにも保存（タスク/フレーム両対応）
        report_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "reports" / "instance_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        safe_name = instance_id.replace('/', '_')
        report_file = report_dir / f"{safe_name}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return report

    def _create_instance_chart(self, instance_id: str, df: pd.DataFrame) -> str:
        """個別データの図表を作成して保存し、パスを返す"""
        try:
            import matplotlib.pyplot as plt
            charts_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "charts" / "instances"
            charts_dir.mkdir(parents=True, exist_ok=True)
            output_path = charts_dir / f"{instance_id}.png"

            plt.figure(figsize=(6, 4))
            if 'frame_num' in df.columns and len(df) > 1:
                # フレームレベル: 検知と期待の推移
                plt.plot(df.get('frame_num', range(len(df))), df.get('is_drowsy', 0), label='Detected')
                if 'expected_is_drowsy' in df.columns:
                    plt.plot(df.get('frame_num', range(len(df))), df['expected_is_drowsy'], label='Expected', linestyle='--')
                plt.xlabel('Frame')
                plt.ylabel('Drowsy (0/1)')
                plt.title(f'{instance_id} Drowsiness over time')
                plt.legend()
            else:
                # タスクレベル: 期待 vs 実測（バー）
                expected = int(df.get('expected_is_drowsy', pd.Series([0])).iloc[0]) if 'expected_is_drowsy' in df.columns else 0
                actual = int(df.get('is_drowsy', pd.Series([0])).iloc[0]) if 'is_drowsy' in df.columns else 0
                plt.bar(['Expected', 'Detected'], [expected, actual], color=['gray', 'blue'])
                plt.ylim(0, 1)
                plt.title(f'{instance_id} Expected vs Detected')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return str(output_path)
        except Exception:
            return ""

    def _generate_instance_summary(self, df: pd.DataFrame,
                                 anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """インスタンスのサマリーを生成

        Args:
            df: 評価データ
            anomalies: 異常リスト

        Returns:
            サマリー辞書
        """
        # 正解率の計算
        accuracy_metrics = self.expectation_generator.calculate_accuracy_metrics(df)

        # 異常の深刻度評価
        severity_scores = {'high': 3, 'medium': 2, 'low': 1}
        max_severity = max([severity_scores.get(a.get('severity', 'low'), 1) for a in anomalies]) if anomalies else 0

        summary = {
            'accuracy': accuracy_metrics.get('accuracy', 0),
            'total_samples': accuracy_metrics.get('total_samples', 0),
            'anomaly_count': len(anomalies),
            'max_severity_score': max_severity,
            'has_errors': len(anomalies) > 0,
            'needs_attention': max_severity >= 2  # medium以上の異常がある場合
        }

        return summary
