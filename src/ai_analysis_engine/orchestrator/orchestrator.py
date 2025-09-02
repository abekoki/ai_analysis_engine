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

    def _load_evaluation_data(self, algorithm_output_id: Optional[int] = None) -> Dict[str, Any]:
        """評価データを読み込み"""
        try:
            # DataWareHouseからメタデータを取得
            metadata_df = self.datawarehouse.get_evaluation_data(algorithm_output_id)
            metadata = metadata_df.iloc[0].to_dict()

            # 評価結果ファイルを読み込み
            algorithm_output_dir = metadata['algorithm_output_dir']
            evaluation_df = self.datawarehouse.load_evaluation_results(algorithm_output_dir)

            return {
                'metadata': metadata,
                'data': evaluation_df,
                'algorithm_output_id': metadata.get('algorithm_output_ID')
            }

        except Exception as e:
            self.logger.error(f"Failed to load evaluation data: {str(e)}")
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

            if not template_path.exists():
                # テンプレートが存在しない場合は簡易レポートを生成
                return self._generate_simple_report(integrated_results)

            # TODO: Jinja2テンプレートを使用したレポート生成を実装
            report_content = self._generate_simple_report(integrated_results)

            # レポート保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"analysis_report_{timestamp}.md"
            report_path = Path(__file__).parent.parent.parent.parent / "outputs" / "reports" / report_filename
            report_path.parent.mkdir(parents=True, exist_ok=True)

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

        report = f"""# AI分析エンジン 最終レポート

## 実行概要
- 実行日時: {integrated_results.get('analysis_timestamp', 'Unknown')}
- 分析対象: drowsy_detectionアルゴリズム

## 性能分析結果
- 正解率: {perf_summary.get('accuracy', 'N/A'):.1%}
- 総サンプル数: {perf_summary.get('total_samples', 'N/A')}

## 個別分析結果
- 分析インスタンス数: {len(instance_summaries)}
"""

        if recommendations:
            report += "\n## 改善提案\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"

        return report

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
