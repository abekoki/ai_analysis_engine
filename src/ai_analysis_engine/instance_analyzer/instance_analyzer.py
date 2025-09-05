"""InstanceAnalyzerクラス

個別データの分析を行うクラスです。
仕様書・ソースコードを参照しつつ、仮説生成・検証を行います。
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import os

from ..config.settings import Settings
from ..utils.expectation_generator import ExpectationGenerator
from ..utils.data_loader import DataWareHouseConnector
from ..utils.rag_system import RAGSystem

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape  # type: ignore
except Exception:  # pragma: no cover
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    select_autoescape = None  # type: ignore


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
        self.only_problem_instances = bool(self.settings.get('instance_analyzer.only_problem_instances', True))

        self.logger.info("InstanceAnalyzer initialized")
        self.rag = RAGSystem(self.settings)
        # 出力ベースディレクトリ（オーケストレータから設定される想定）
        self.output_base_dir: Optional[Path] = None

    def set_output_base_dir(self, base_dir: Path) -> None:
        """オーケストレータから出力ルートを受け取る"""
        self.output_base_dir = base_dir

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
            # タスクレベル（frame_numが無い）かつ is_correct 列があれば、不正解のみを対象にフィルタ
            if 'frame_num' not in df.columns and 'is_correct' in df.columns:
                df = df[df['is_correct'] == 0].reset_index(drop=True)
            instance_groups = self._split_into_instances(df)

            results = []
            for i, (instance_id, instance_df) in enumerate(instance_groups.items()):
                self.logger.info(f"Analyzing instance {i+1}/{len(instance_groups)}: {instance_id}")

                try:
                    result = self._analyze_single_instance(instance_id, instance_df, evaluation_data)
                    # 課題が無いデータを除外（設定で切替）
                    if self.only_problem_instances and not result.get('summary', {}).get('has_errors', False):
                        continue
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

            # レポート相対パス（summary.md からの相対）
            safe_name = instance_id.replace('/', '_')
            if self.output_base_dir is not None:
                report_relpath = f"instance_reports/{safe_name}.md"
            else:
                report_relpath = str((Path(__file__).parent.parent.parent.parent / "outputs" / "instance_reports" / f"{safe_name}.md").resolve())

            result = {
                'instance_id': instance_id,
                'status': 'success',
                'data_shape': instance_df.shape,
                'hypothesis_results': hypothesis_results,
                'anomalies': anomalies,
                'report': instance_report,
                'report_relpath': report_relpath,
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
        # LLM/RAG を用いた仮説生成（フォールバックあり）
        context = {"metadata": evaluation_data.get('metadata', {})}
        hypothesis = self.rag.generate_hypothesis(df, context)

        # 仮説の検証
        verification = self.rag.verify_hypothesis(hypothesis, df)

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

        # 図表と付帯情報の生成
        chart_path = self._create_instance_chart(instance_id, df)

        # タスクレベル（1行）で必要情報がある場合は、サンプル準拠の2種グラフとメタ情報を生成
        enriched: Dict[str, Any] = {}
        is_task_level = 'frame_num' not in df.columns and len(df) >= 1
        if is_task_level and {'start', 'end', 'algorithm_output_dir', 'core_lib_output_dir', 'video_dir'}.issubset(set(df.columns)):
            try:
                enriched = self._create_task_charts_and_summary(instance_id, df.iloc[0].to_dict())
            except Exception as e:
                self.logger.error(f"Failed to create task-level charts for {instance_id}: {str(e)}")
                enriched = {}
        # pandasaiで時系列の自然言語要約（可能なら）
        try:
            narrative = self.rag.pandasai_narrative(df)
        except Exception:
            narrative = ""

        # テンプレートがあれば使用
        templates_dir = Path(self.settings.get('global.templates_path', './templates/'))
        template_file = templates_dir / 'instance_report_template.md.j2'
        if Environment is not None and template_file.exists():
            env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=select_autoescape(disabled_extensions=(".md", ".j2"))
            )
            template = env.get_template('instance_report_template.md.j2')
            report = template.render(
                instance_id=instance_id,
                total_frames=total_frames,
                drowsy_frames=drowsy_frames,
                expected_drowsy_frames=expected_drowsy_frames,
                chart_path=chart_path,
                hypothesis_results=hypothesis_results,
                anomalies=anomalies,
                rag_sources=getattr(self.rag, 'last_sources', []),
                # サンプル準拠の追加項目（存在しない場合はNone）
                conclusion=enriched.get('conclusion'),
                video_display=enriched.get('video_display'),
                video_link_path=enriched.get('video_link_path'),
                frame_range=enriched.get('frame_range'),
                expected_label=enriched.get('expected_label'),
                detected_label=enriched.get('detected_label'),
                algo_chart_path=enriched.get('algo_chart_path'),
                core_chart_path=enriched.get('core_chart_path'),
                io_summary=enriched.get('io_summary'),
                possible_causes=enriched.get('possible_causes'),
            )
        else:
            # フォールバック: 既存のテキスト生成
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

            if narrative:
                report += "\n## pandasaiによる時系列要約\n" + narrative + "\n"

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
        report_dir = (self.output_base_dir / "instance_reports") if self.output_base_dir is not None else (Path(__file__).parent.parent.parent.parent / "outputs" / "instance_reports")
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
            # charts出力先: ラン配下/instance_reports/charts
            if self.output_base_dir is not None:
                charts_dir = self.output_base_dir / "instance_reports" / "charts"
            else:
                charts_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "instance_reports" / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            output_path = charts_dir / f"{instance_id}.png"

            if 'frame_num' in df.columns and len(df) > 1:
                # フレームレベル: 説得力のある時系列（3段プロット）
                plt.figure(figsize=(10, 6))
                frames = df.get('frame_num', range(len(df)))

                # 1) 目の開眼度
                ax1 = plt.subplot(3, 1, 1)
                if 'left_eye_open' in df.columns:
                    ax1.plot(frames, df['left_eye_open'], label='Left Eye', alpha=0.8)
                if 'right_eye_open' in df.columns:
                    ax1.plot(frames, df['right_eye_open'], label='Right Eye', alpha=0.8)
                ax1.set_ylabel('Eye Openness')
                ax1.set_title(f'{instance_id} Eye Openness & Confidence & Detection')
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper right')

                # 2) 顔検出信頼度
                ax2 = plt.subplot(3, 1, 2, sharex=ax1)
                if 'face_confidence' in df.columns:
                    ax2.plot(frames, df['face_confidence'], color='green', alpha=0.8, label='Face Confidence')
                ax2.set_ylabel('Confidence')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right')

                # 3) 検知 vs 期待
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                ax3.plot(frames, df.get('is_drowsy', 0), label='Detected', alpha=0.9)
                if 'expected_is_drowsy' in df.columns:
                    ax3.plot(frames, df['expected_is_drowsy'], label='Expected', linestyle='--', alpha=0.9)
                ax3.set_xlabel('Frame')
                ax3.set_ylabel('Drowsy (0/1)')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper right')

                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                # タスクレベル: 期待 vs 実測（バー）
                import pandas as _pd  # type: ignore
                expected = int(df.get('expected_is_drowsy', _pd.Series([0])).iloc[0]) if 'expected_is_drowsy' in df.columns else 0
                actual = int(df.get('is_drowsy', _pd.Series([0])).iloc[0]) if 'is_drowsy' in df.columns else 0
                plt.figure(figsize=(6, 4))
                plt.bar(['Expected', 'Detected'], [expected, actual], color=['gray', 'blue'])
                plt.ylim(0, 1)
                plt.title(f'{instance_id} Expected vs Detected')
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            return str(output_path)
        except Exception:
            return ""

    def _create_task_charts_and_summary(self, instance_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """サンプルに合わせたタスクレベルの2種グラフとメタ情報を生成

        - アルゴリズム出力の該当区間グラフ（is_drowsy, left/right_eye_closed 等があれば）
        - コア出力の該当区間グラフ（leye_openness, reye_openness, 閾値線）
        - 結論、動画リンク、フレーム区間、期待/検知ラベル
        """
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # local alias

        start = int(row.get('start', 0))
        end = int(row.get('end', -1))
        algo_dir_rel = row.get('algorithm_output_dir')
        core_dir_rel = row.get('core_lib_output_dir')
        video_dir_rel = row.get('video_dir')

        project_root = Path(__file__).parent.parent.parent.parent
        dwh_root = Path(self.settings.get('global.datawarehouse_path', '../DataWareHouse/'))

        # charts出力先: ラン配下/instance_reports/charts
        charts_dir = (self.output_base_dir / 'instance_reports' / 'charts') if self.output_base_dir is not None else (project_root / 'outputs' / 'instance_reports' / 'charts')
        charts_dir.mkdir(parents=True, exist_ok=True)

        # 1) アルゴリズム出力CSVの区間グラフ
        algo_chart_path = charts_dir / f"{instance_id}_algo.png"
        try:
            algo_df = pd.DataFrame()
            if algo_dir_rel:
                algo_dir_abs = dwh_root / algo_dir_rel
                csvs = sorted(list(algo_dir_abs.glob('*.csv')))
                if csvs:
                    # 代表CSV（最初）を使用。将来は該当videoに紐付けて選別も可
                    algo_df = pd.read_csv(csvs[0])
                    if 'frame' in algo_df.columns and 'frame_num' not in algo_df.columns:
                        algo_df = algo_df.rename(columns={'frame': 'frame_num'})
            if not algo_df.empty and 'frame_num' in algo_df.columns:
                seg = algo_df[(algo_df['frame_num'] >= start) & (algo_df['frame_num'] <= end)].copy()
                if not seg.empty:
                    plt.figure(figsize=(10, 4))
                    frames = seg['frame_num']
                    if 'is_drowsy' in seg.columns:
                        plt.plot(frames, seg['is_drowsy'], label='is_drowsy')
                    if 'left_eye_closed' in seg.columns:
                        plt.plot(frames, seg['left_eye_closed'].astype(int), label='left_eye_closed')
                    if 'right_eye_closed' in seg.columns:
                        plt.plot(frames, seg['right_eye_closed'].astype(int), label='right_eye_closed')
                    plt.title('アルゴリズム出力（区間）')
                    plt.xlabel('frame')
                    plt.legend(loc='upper right')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(algo_chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    algo_chart_path = None
            else:
                algo_chart_path = None
        except Exception:
            algo_chart_path = None

        # 2) コア出力CSVの区間グラフ（閾値線つき）
        core_chart_path = charts_dir / f"{instance_id}_core.png"
        try:
            core_df = pd.DataFrame()
            if core_dir_rel:
                core_dir_abs = dwh_root / core_dir_rel
                csvs = sorted(list(core_dir_abs.glob('*.csv')))
                if csvs:
                    core_df = pd.read_csv(csvs[0])
                    # 列名合わせ
                    # 期待: leye_openness, reye_openness（なければ推測）
                    if 'frame' in core_df.columns and 'frame_num' not in core_df.columns:
                        core_df = core_df.rename(columns={'frame': 'frame_num'})
                    # 別表記対応
                    if 'leye_openness' not in core_df.columns and 'left_eye_open' in core_df.columns:
                        core_df['leye_openness'] = core_df['left_eye_open']
                    if 'reye_openness' not in core_df.columns and 'right_eye_open' in core_df.columns:
                        core_df['reye_openness'] = core_df['right_eye_open']
            if not core_df.empty and {'frame_num'}.issubset(set(core_df.columns)):
                seg = core_df[(core_df['frame_num'] >= start) & (core_df['frame_num'] <= end)].copy()
                if not seg.empty and {'leye_openness', 'reye_openness'}.issubset(set(seg.columns)):
                    plt.figure(figsize=(10, 4))
                    frames = seg['frame_num']
                    plt.plot(frames, seg['leye_openness'], label='leye_openness')
                    plt.plot(frames, seg['reye_openness'], label='reye_openness')
                    thr = min(self.expectation_generator.left_eye_threshold, self.expectation_generator.right_eye_threshold)
                    plt.hlines(thr, xmin=frames.min(), xmax=frames.max(), colors='red', linestyles='--', label='threshold')
                    plt.title('コア出力（区間）')
                    plt.xlabel('frame')
                    plt.legend(loc='upper right')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(core_chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    core_chart_path = None
            else:
                core_chart_path = None
        except Exception:
            core_chart_path = None

        # 3) 付帯情報の作成
        expected_label = '連続閉眼あり'  # タスクタグは閉眼タスク想定
        detected_label = '連続閉眼あり' if int(row.get('is_drowsy', 0)) == 1 else '連続閉眼なし'
        # 結論: 簡易ロジック（未検知 or 仕様不一致）
        if detected_label == '連続閉眼なし':
            conclusion = '被験者が閉眼していない。 or コアの検出機能に問題がある。'
        else:
            conclusion = '特筆すべき問題は見られない。'

        # 動画リンク（相対パス表示優先）
        video_display = None
        video_link_path = None
        if video_dir_rel:
            video_link_path = str(Path('..') / 'ImproveAlgorithmDevelopment' / 'DataWareHouse' / video_dir_rel)
            video_display = Path(video_link_path).name

        frame_range = f"{start}-{end}"

        # IOサマリと原因（簡易）
        io_summary = None
        possible_causes: List[str] = []
        try:
            if 'leye_openness' in locals().get('seg', {}).columns and 'reye_openness' in locals().get('seg', {}).columns:  # type: ignore
                mean_leye = float(seg['leye_openness'].mean())  # type: ignore
                mean_reye = float(seg['reye_openness'].mean())  # type: ignore
                approx = (mean_leye + mean_reye) / 2.0
                trend = '閉眼傾向が見られない' if approx > self.expectation_generator.left_eye_threshold else '閉眼傾向が見られる'
                io_summary = f"task中のreye_opennessとleye_opennessが{approx:.2f}程度になっており、{trend}。"
        except Exception:
            io_summary = None

        if expected_label == '連続閉眼あり' and detected_label == '連続閉眼なし':
            possible_causes = [
                '被験者が閉眼していない。',
                '被験者が正しく閉眼しているが、コアの検出機能に問題がある。'
            ]

        # 相対パスに変換（レポートからの相対: instance_reports/配下のMDから見て charts/..）
        def to_relative(p: Optional[Path]) -> Optional[str]:
            if p is None:
                return None
            try:
                base = (self.output_base_dir / 'instance_reports') if self.output_base_dir is not None else charts_dir.parent
                return str(Path(os.path.relpath(p, base)))
            except Exception:
                return str(p)

        return {
            'algo_chart_path': to_relative(algo_chart_path if isinstance(algo_chart_path, Path) else Path(str(algo_chart_path)) if algo_chart_path else None),
            'core_chart_path': to_relative(core_chart_path if isinstance(core_chart_path, Path) else Path(str(core_chart_path)) if core_chart_path else None),
            'conclusion': conclusion,
            'video_display': video_display,
            'video_link_path': video_link_path,
            'frame_range': frame_range,
            'expected_label': expected_label,
            'detected_label': detected_label,
            'io_summary': io_summary,
            'possible_causes': possible_causes,
        }

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
