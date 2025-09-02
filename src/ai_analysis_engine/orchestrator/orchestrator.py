"""Orchestratorクラス

AI分析エンジンの全体統制を行うクラスです。
PerformanceAnalyzerとInstanceAnalyzerを管理し、分析フローを制御します。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from ..config.settings import Settings
from ..performance_analyzer.performance_analyzer import PerformanceAnalyzer
from ..instance_analyzer.instance_analyzer import InstanceAnalyzer
from ..utils.data_loader import DataWareHouseConnector


class Orchestrator:
    """AI分析エンジンのオーケストレータクラス"""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Args:
            settings: 設定オブジェクト（Noneの場合はデフォルト設定を使用）
        """
        self.settings = settings or Settings()
        self.logger = self._setup_logger()

        # 各エージェントの初期化
        self.performance_analyzer = PerformanceAnalyzer(self.settings)
        self.instance_analyzer = InstanceAnalyzer(self.settings)

        # DataWareHouse接続
        db_path = self.settings.get('global.database_path')
        self.datawarehouse = DataWareHouseConnector(db_path)

        self.logger.info("Orchestrator initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(__name__)

        # 既にハンドラーが設定されている場合はスキップ
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # ファイルハンドラー（logsディレクトリに保存）
        log_file = Path(__file__).parent.parent.parent.parent / "logs" / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def run_analysis(self, algorithm_output_id: Optional[int] = None) -> Dict[str, Any]:
        """分析実行

        Args:
            algorithm_output_id: 分析対象のアルゴリズム出力ID（Noneの場合は最新を使用）

        Returns:
            分析結果辞書
        """
        try:
            self.logger.info(f"Starting analysis for algorithm_output_id: {algorithm_output_id}")

            # 1. 評価データを取得
            evaluation_data = self._load_evaluation_data(algorithm_output_id)
            self.logger.info(f"Loaded evaluation data: {len(evaluation_data)} records")

            # 2. 全体性能分析を実行
            self.logger.info("Running performance analysis...")
            performance_results = self.performance_analyzer.analyze_performance(evaluation_data)
            self.logger.info(f"Performance analysis completed: {performance_results.get('summary', {})}")

            # 3. 個別データ分析を実行
            self.logger.info("Running instance analysis...")
            instance_results = self.instance_analyzer.analyze_instances(evaluation_data)
            self.logger.info(f"Instance analysis completed: {len(instance_results)} instances analyzed")

            # 4. 結果を統合
            integrated_results = self._integrate_results(performance_results, instance_results)

            # 5. 最終レポートを生成
            report_path = self._generate_final_report(integrated_results)

            # 6. 結果をDataWareHouseに保存
            self._save_results_to_datawarehouse(integrated_results, evaluation_data)

            final_result = {
                'status': 'success',
                'performance_results': performance_results,
                'instance_results': instance_results,
                'integrated_results': integrated_results,
                'report_path': str(report_path),
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info("Analysis completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_analysis_by_evaluation_result(self, evaluation_result_id: int) -> Dict[str, Any]:
        """evaluation_result_ID を指定して分析実行"""
        try:
            self.logger.info(f"Starting analysis for evaluation_result_id: {evaluation_result_id}")

            # 1. 評価データを取得（evaluation_result_ID単位で全個別データを集約）
            evaluation_data = self._load_evaluation_data_by_result(evaluation_result_id)
            self.logger.info(f"Loaded evaluation data: {len(evaluation_data.get('data', [])) if hasattr(evaluation_data.get('data'), '__len__') else 'unknown'} records")

            # 2. 全体性能分析
            self.logger.info("Running performance analysis...")
            performance_results = self.performance_analyzer.analyze_performance(evaluation_data)
            self.logger.info(f"Performance analysis completed: {performance_results.get('summary', {})}")

            # 3. 個別データ分析
            self.logger.info("Running instance analysis...")
            instance_results = self.instance_analyzer.analyze_instances(evaluation_data)
            self.logger.info(f"Instance analysis completed: {len(instance_results)} instances analyzed")

            # 4. 統合
            integrated_results = self._integrate_results(performance_results, instance_results)

            # 5. レポート生成
            report_path = self._generate_final_report(integrated_results)

            # 5.5 JSON成果物も出力
            self._export_data_artifacts(integrated_results)

            # 6. 結果保存（evaluation_result_idで保存APIに渡すために互換引数名で流用）
            try:
                self._save_results_to_datawarehouse(integrated_results, {
                    'algorithm_output_id': evaluation_result_id
                })
            except Exception as e:
                self.logger.error(f"Failed to save results to DataWareHouse: {str(e)}")

            final_result = {
                'status': 'success',
                'performance_results': performance_results,
                'instance_results': instance_results,
                'integrated_results': integrated_results,
                'report_path': str(report_path),
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info("Analysis completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _load_evaluation_data(self, algorithm_output_id: Optional[int] = None) -> Dict[str, Any]:
        """評価データを読み込み"""
        try:
            # DataWareHouseから評価データを取得（辞書形式が返る）
            eval_payload = self.datawarehouse.get_evaluation_data(algorithm_output_id)

            # 互換用に期待キーを整形
            metadata = eval_payload.get('metadata', {})
            evaluation_df = eval_payload.get('data')
            algorithm_output_id = eval_payload.get('algorithm_output_id')

            if evaluation_df is None or getattr(evaluation_df, 'empty', True):
                raise ValueError("評価データが空です")

            return {
                'metadata': metadata,
                'data': evaluation_df,
                'algorithm_output_id': algorithm_output_id,
            }

        except Exception as e:
            self.logger.error(f"Failed to load evaluation data: {str(e)}")
            raise

    def _load_evaluation_data_by_result(self, evaluation_result_id: int) -> Dict[str, Any]:
        """evaluation_result_IDから評価データを読み込み"""
        try:
            # タスク単位の評価に合わせ、タグ区間に対する検出有無を構築
            task_df = self.datawarehouse.build_task_level_dataframe(evaluation_result_id)
            if task_df is None or getattr(task_df, 'empty', True):
                # フォールバック: 既存のevaluation_data_path連結
                payload = self.datawarehouse.get_evaluation_data_by_result_id(evaluation_result_id)
                if payload.get('data') is None or getattr(payload.get('data'), 'empty', True):
                    raise ValueError("評価データが空です")
                return {
                    'metadata': payload.get('metadata', {}),
                    'data': payload.get('data'),
                    'evaluation_result_id': evaluation_result_id,
                }
            return {
                'metadata': {'evaluation_result_id': evaluation_result_id},
                'data': task_df,
                'evaluation_result_id': evaluation_result_id,
            }
        except Exception as e:
            self.logger.error(f"Failed to load evaluation data by result: {str(e)}")
            raise

    def _integrate_results(self, performance_results: Dict[str, Any],
                          instance_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """結果を統合"""
        integrated = {
            'performance_summary': performance_results.get('summary', {}),
            'instance_summaries': [result.get('summary', {}) for result in instance_results],
            'recommendations': self._generate_recommendations(performance_results, instance_results),
            'analysis_timestamp': datetime.now().isoformat()
        }

        return integrated

    def _generate_recommendations(self, performance_results: Dict[str, Any],
                                instance_results: List[Dict[str, Any]]) -> List[str]:
        """改善提案を生成"""
        recommendations = []

        # 性能分析結果に基づく提案
        perf_summary = performance_results.get('summary', {})
        accuracy = perf_summary.get('accuracy', 0)

        if accuracy < 0.8:
            recommendations.append("正解率が80%未満です。アルゴリズムの閾値調整を検討してください。")
        elif accuracy < 0.9:
            recommendations.append("正解率が90%未満です。さらに精度向上の余地があります。")

        # 個別分析結果に基づく提案
        error_instances = [result for result in instance_results
                          if result.get('summary', {}).get('has_errors', False)]

        if error_instances:
            recommendations.append(f"{len(error_instances)}件の異常データが検出されました。詳細分析が必要です。")

        return recommendations

    def _generate_final_report(self, integrated_results: Dict[str, Any]) -> Path:
        """最終レポートを生成"""
        try:
            # レポートテンプレートの読み込み
            template_path = Path(__file__).parent.parent.parent.parent / "templates" / "report_template.md.j2"
            # 現段階ではテンプレート未対応。存在しなくても必ずファイル出力する
            if not template_path.exists():
                report_content = self._generate_simple_report(integrated_results)
            else:
                # TODO: Jinja2テンプレートを使用したレポート生成を実装
                report_content = self._generate_simple_report(integrated_results)

            # レポート保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"analysis_report_{timestamp}.md"
            report_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / report_filename

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            self.logger.info(f"Final report generated: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Failed to generate final report: {str(e)}")
            raise

    def _generate_simple_report(self, integrated_results: Dict[str, Any]) -> str:
        """簡易レポートを生成"""
        perf_summary = integrated_results.get('performance_summary', {})
        instance_summaries = integrated_results.get('instance_summaries', [])
        recommendations = integrated_results.get('recommendations', [])

        accuracy_val = perf_summary.get('accuracy', 'N/A')
        try:
            accuracy_str = f"{float(accuracy_val):.1%}" if isinstance(accuracy_val, (int, float)) or (isinstance(accuracy_val, str) and accuracy_val.replace('.', '', 1).isdigit()) else str(accuracy_val)
        except Exception:
            accuracy_str = str(accuracy_val)

        report = f"""# AI分析エンジン 最終レポート

## 実行概要
- 実行日時: {integrated_results.get('analysis_timestamp', 'Unknown')}
- 分析対象: drowsy_detectionアルゴリズム

## 性能分析結果
- 正解率: {accuracy_str}
- 総サンプル数: {perf_summary.get('total_samples', 'N/A')}

## 個別分析結果
- 分析インスタンス数: {len(instance_summaries)}
"""

        if recommendations:
            report += "\n## 改善提案\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"

        return report

    def _export_data_artifacts(self, integrated_results: Dict[str, Any]) -> Dict[str, str]:
        """JSON成果物をoutputs/data配下に出力"""
        try:
            data_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            def dump_json(name: str, payload: Any) -> str:
                path = data_dir / f"{name}.json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self._to_serializable(payload), f, ensure_ascii=False, indent=2)
                return str(path)

            paths = {}
            paths['analysis_summary'] = dump_json('analysis_summary', integrated_results)
            if isinstance(integrated_results.get('performance_summary'), dict):
                paths['performance_metrics'] = dump_json('performance_metrics', integrated_results['performance_summary'])
            if isinstance(integrated_results.get('recommendations'), list):
                paths['improvement_suggestions'] = dump_json('improvement_suggestions', integrated_results['recommendations'])
            return paths
        except Exception as e:
            self.logger.error(f"Failed to export data artifacts: {str(e)}")
            return {}

    def _to_serializable(self, obj: Any) -> Any:
        """JSONシリアライズ可能なPython型に変換"""
        try:
            import numpy as np
        except Exception:
            np = None  # 型判定をスキップ

        if isinstance(obj, dict):
            return {self._to_serializable(k): self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_serializable(v) for v in obj]
        if np is not None:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        return obj

    def _save_results_to_datawarehouse(self, integrated_results: Dict[str, Any],
                                     evaluation_data: Dict[str, Any]) -> None:
        """結果をDataWareHouseに保存"""
        try:
            algorithm_output_id = evaluation_data.get('algorithm_output_id')

            # 結果を保存し、DataWareHouseに登録
            analysis_result_id = self.datawarehouse.save_analysis_results(
                results=integrated_results,
                algorithm_output_id=algorithm_output_id,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

            self.logger.info(f"Results saved to DataWareHouse: analysis_result_id={analysis_result_id}")

        except Exception as e:
            self.logger.error(f"Failed to save results to DataWareHouse: {str(e)}")
            # 保存失敗時は処理を継続（ログに記録済み）
