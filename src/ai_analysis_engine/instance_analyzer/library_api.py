"""
ライブラリAPI - 上位システム統合向けのシンプルなインターフェース
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .main import AIAnalysisEngine as _InternalEngine
from .config.library_config import AnalysisConfig
from .models.result import AnalysisResult, Hypothesis, AnalysisMetrics
from .exceptions import (
    AIAnalysisError,
    ConfigurationError,
    ValidationError,
    AnalysisError,
    TimeoutError,
    InitializationError
)


class AIAnalysisEngine:
    """
    AI分析エンジンライブラリ - 上位システム統合向け

    シンプルで使いやすいAPIを提供し、複雑な内部処理を隠蔽
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        エンジン初期化

        Args:
            config: 分析設定（Noneの場合はデフォルト設定を使用）
        """
        self.config = config or AnalysisConfig.from_env()
        self._internal_engine = None
        self._rag_initialized = False
        self._algorithm_specs = []
        self._algorithm_codes = []
        self._evaluation_specs = []
        self._evaluation_codes = []

        # 設定検証
        try:
            self.config.validate()
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration: {e}")

    def initialize(
        self,
        algorithm_specs: List[str],
        algorithm_codes: List[str],
        evaluation_specs: List[str],
        evaluation_codes: List[str]
    ) -> bool:
        """
        エンジンの初期化とRAGベクトル化

        Args:
            algorithm_specs: アルゴリズム仕様Markdownファイルパスのリスト
            algorithm_codes: アルゴリズム実装コードファイルパスのリスト
            evaluation_specs: 評価仕様Markdownファイルパスのリスト
            evaluation_codes: 評価環境コードファイルパスのリスト

        Returns:
            bool: 初期化成功の場合True

        Raises:
            InitializationError: 初期化失敗時
            ValidationError: 入力検証エラー
        """
        # 入力検証
        self._validate_initialization_inputs(
            algorithm_specs, algorithm_codes, evaluation_specs, evaluation_codes
        )

        try:
            # 内部エンジンの初期化
            if self._internal_engine is None:
                self._internal_engine = _InternalEngine()

            if not self._internal_engine.initialize():
                raise InitializationError("Failed to initialize internal engine")

            # RAGドキュメントの保存
            self._algorithm_specs = algorithm_specs
            self._algorithm_codes = algorithm_codes
            self._evaluation_specs = evaluation_specs
            self._evaluation_codes = evaluation_codes

            # RAGベクトル化の実行
            self._initialize_rag_vector_stores()
            self._rag_initialized = True

            return True

        except Exception as e:
            raise InitializationError(f"Engine initialization failed: {e}")

    def is_initialized(self) -> bool:
        """
        初期化状態を確認

        Returns:
            bool: 初期化済みの場合True
        """
        return (self._internal_engine is not None and
                self._rag_initialized and
                bool(self._algorithm_specs) and
                bool(self._algorithm_codes) and
                bool(self._evaluation_specs) and
                bool(self._evaluation_codes))

    def analyze(
        self,
        algorithm_outputs: List[str],
        core_outputs: List[str],
        expected_results: List[str],
        output_dir: str,
        dataset_ids: List[str],
        timeout: Optional[int] = None
    ) -> List[AnalysisResult]:
        """
        複数データセットの一括分析を実行

        Args:
            algorithm_outputs: アルゴリズム出力CSVファイルパスのリスト
            core_outputs: コアライブラリ出力CSVファイルパスのリスト
            expected_results: 期待される結果の自然言語記述のリスト
            output_dir: 結果出力ディレクトリ
            dataset_ids: データセットIDのリスト
            timeout: タイムアウト時間（秒、Noneの場合は設定値を使用）

        Returns:
            List[AnalysisResult]: 分析結果のリスト

        Raises:
            ValidationError: 入力検証エラー
            AnalysisError: 分析実行エラー
            TimeoutError: タイムアウト
        """
        if not self.is_initialized():
            raise InitializationError("Engine not initialized. Call initialize() first.")

        # 入力検証
        self._validate_analysis_inputs(
            algorithm_outputs, core_outputs, expected_results, output_dir, dataset_ids
        )

        # タイムアウト設定
        actual_timeout = timeout or self.config.timeout
        results = []

        try:
            # 各データセットの処理
            for i, (algo_csv, core_csv, expected, dataset_id) in enumerate(zip(
                algorithm_outputs, core_outputs, expected_results, dataset_ids
            )):
                try:
                    print(f"📊 データセット {i+1}/{len(dataset_ids)} を処理中: {dataset_id}")

                    # 内部エンジン用のリクエスト作成
                    state = self._internal_engine.create_analysis_request(
                        algorithm_outputs=[algo_csv],
                        core_outputs=[core_csv],
                        algorithm_specs=self._algorithm_specs,
                        evaluation_specs=self._evaluation_specs,
                        expected_results=[expected],
                        algorithm_codes=self._algorithm_codes,
                        evaluation_codes=self._evaluation_codes,
                        dataset_ids=[dataset_id],
                        output_dir=output_dir
                    )

                    # 分析実行（タイムアウト付き）
                    start_time = time.time()
                    result = self._run_with_timeout(
                        self._internal_engine.run_analysis(state),
                        actual_timeout
                    )
                    execution_time = time.time() - start_time

                    # 結果の変換
                    analysis_result = self._convert_to_library_result(
                        result, dataset_id, execution_time
                    )
                    results.append(analysis_result)

                    print(f"✅ データセット {dataset_id} の分析完了")

                except Exception as e:
                    print(f"❌ データセット {dataset_id} の分析失敗: {e}")
                    # エラーが発生しても処理を継続
                    error_result = AnalysisResult.error_result(
                        dataset_id=dataset_id,
                        error=str(e),
                        error_details={"batch_index": i, "traceback": str(e)}
                    )
                    results.append(error_result)

            print(f"📊 一括分析完了: {len(results)}/{len(dataset_ids)} 件処理")
            return results

        except asyncio.TimeoutError:
            raise TimeoutError(f"Batch analysis timed out after {actual_timeout} seconds")
        except Exception as e:
            raise AnalysisError(f"Analysis failed: {e}")

    def _initialize_rag_vector_stores(self) -> None:
        """
        RAGベクトルストアの初期化

        初期化時に提供された仕様書とコードファイルをベクトル化
        """
        try:
            from .tools.rag_tool import RAGTool

            print("🔍 RAGベクトル化を開始...")

            # RAGツールの初期化
            rag_tool = RAGTool()

            # ドキュメントの準備
            documents = {
                "algorithm_specs": self._algorithm_specs,
                "algorithm_codes": self._algorithm_codes,
                "evaluation_specs": self._evaluation_specs,
                "evaluation_codes": self._evaluation_codes
            }

            # ベクトルストアの初期化
            if rag_tool.initialize_vector_stores(documents):
                print("✅ RAGベクトル化完了")
            else:
                print("⚠️ RAGベクトル化で警告が発生しましたが、処理を継続します")

        except Exception as e:
            print(f"⚠️ RAGベクトル化でエラーが発生しましたが、処理を継続します: {e}")

    def _validate_initialization_inputs(
        self,
        algorithm_specs: List[str],
        algorithm_codes: List[str],
        evaluation_specs: List[str],
        evaluation_codes: List[str]
    ) -> None:
        """
        初期化入力の検証

        Args:
            algorithm_specs: アルゴリズム仕様ファイルパスのリスト
            algorithm_codes: アルゴリズムコードファイルパスのリスト
            evaluation_specs: 評価仕様ファイルパスのリスト
            evaluation_codes: 評価コードファイルパスのリスト

        Raises:
            ValidationError: 検証エラー
        """
        # 必須パラメータのチェック
        if not algorithm_specs:
            raise ValidationError("algorithm_specs cannot be empty")
        if not algorithm_codes:
            raise ValidationError("algorithm_codes cannot be empty")
        if not evaluation_specs:
            raise ValidationError("evaluation_specs cannot be empty")
        if not evaluation_codes:
            raise ValidationError("evaluation_codes cannot be empty")

        # ファイル存在チェック
        all_files = algorithm_specs + algorithm_codes + evaluation_specs + evaluation_codes
        for file_path in all_files:
            if not Path(file_path).exists():
                raise ValidationError(f"File does not exist: {file_path}")

        # ファイル拡張子チェック
        for spec_file in algorithm_specs + evaluation_specs:
            if not spec_file.lower().endswith(('.md', '.txt', '.markdown')):
                raise ValidationError(f"Specification file must be .md, .txt, or .markdown: {spec_file}")

        print(f"✅ 初期化入力検証完了: {len(all_files)} ファイル確認済み")

    def _validate_analysis_inputs(
        self,
        algorithm_outputs: List[str],
        core_outputs: List[str],
        expected_results: List[str],
        output_dir: str,
        dataset_ids: List[str]
    ) -> None:
        """
        分析入力の検証

        Args:
            algorithm_outputs: アルゴリズム出力ファイルパスのリスト
            core_outputs: コア出力ファイルパスのリスト
            expected_results: 期待結果のリスト
            output_dir: 出力ディレクトリ
            dataset_ids: データセットIDのリスト

        Raises:
            ValidationError: 検証エラー
        """
        # リスト長の一致チェック
        lengths = [
            len(algorithm_outputs),
            len(core_outputs),
            len(expected_results),
            len(dataset_ids)
        ]

        if len(set(lengths)) != 1:
            raise ValidationError(
                f"All input lists must have the same length. "
                f"Got lengths: algorithm_outputs={len(algorithm_outputs)}, "
                f"core_outputs={len(core_outputs)}, "
                f"expected_results={len(expected_results)}, "
                f"dataset_ids={len(dataset_ids)}"
            )

        # ファイル存在チェック
        for file_path in algorithm_outputs + core_outputs:
            if not Path(file_path).exists():
                raise ValidationError(f"File does not exist: {file_path}")

        # ファイル拡張子チェック
        for csv_file in algorithm_outputs + core_outputs:
            if not csv_file.lower().endswith('.csv'):
                raise ValidationError(f"Output file must be .csv: {csv_file}")

        # 出力ディレクトリチェック
        if not output_dir.strip():
            raise ValidationError("output_dir cannot be empty")

        # データセットIDチェック
        for dataset_id in dataset_ids:
            if not dataset_id.strip():
                raise ValidationError("dataset_ids cannot contain empty strings")

        # 期待結果チェック
        for expected in expected_results:
            if not expected.strip():
                raise ValidationError("expected_results cannot contain empty strings")

        print(f"✅ 分析入力検証完了: {len(algorithm_outputs)} 件のデータセットを確認")

    async def analyze_async(
        self,
        algorithm_outputs: List[str],
        core_outputs: List[str],
        expected_results: List[str],
        output_dir: str,
        dataset_ids: List[str],
        timeout: Optional[int] = None
    ) -> List[AnalysisResult]:
        """
        複数データセットの一括分析を非同期実行

        Args:
            algorithm_outputs: アルゴリズム出力CSVファイルパスのリスト
            core_outputs: コアライブラリ出力CSVファイルパスのリスト
            expected_results: 期待される結果の自然言語記述のリスト
            output_dir: 結果出力ディレクトリ
            dataset_ids: データセットIDのリスト
            timeout: タイムアウト時間（秒、Noneの場合は設定値を使用）

        Returns:
            List[AnalysisResult]: 分析結果のリスト
        """
        # 現在の実装は同期処理なので、非同期ラッパーを提供
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyze,
            algorithm_outputs,
            core_outputs,
            expected_results,
            output_dir,
            dataset_ids,
            timeout
        )


    def get_status(self) -> Dict[str, Any]:
        """
        エンジンの現在の状態を取得

        Returns:
            Dict[str, Any]: 状態情報
        """
        if not self._internal_engine:
            return {
                "initialized": False,
                "rag_initialized": False,
                "config_valid": self.config.validate() if self.config else False,
                "version": "1.0.0"
            }

        internal_status = self._internal_engine.get_status()
        return {
            "initialized": self.is_initialized(),
            "rag_initialized": self._rag_initialized,
            "config_valid": self.config.validate() if self.config else False,
            "internal_engine_ready": internal_status.get("initialized", False),
            "documents_loaded": {
                "algorithm_specs": len(self._algorithm_specs),
                "algorithm_codes": len(self._algorithm_codes),
                "evaluation_specs": len(self._evaluation_specs),
                "evaluation_codes": len(self._evaluation_codes)
            },
            "version": "1.0.0",
            "config": {
                "model": self.config.model,
                "timeout": self.config.timeout,
                "output_dir": self.config.output_dir
            }
        }

    def shutdown(self) -> None:
        """
        エンジンのクリーンアップ

        内部リソースの解放とクリーンアップを行う
        """
        if self._internal_engine:
            # 内部エンジンのクリーンアップ（必要に応じて）
            pass

        # RAG関連の状態クリア
        self._rag_initialized = False
        self._algorithm_specs = []
        self._algorithm_codes = []
        self._evaluation_specs = []
        self._evaluation_codes = []

        self._internal_engine = None


    def _run_with_timeout(self, func_result, timeout: int):
        """タイムアウト付きで関数を実行"""
        import threading
        import queue

        result_queue = queue.Queue()

        def run_func():
            try:
                result_queue.put(func_result)
            except Exception as e:
                result_queue.put(e)

        thread = threading.Thread(target=run_func)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Analysis timed out after {timeout} seconds")

        result = result_queue.get()
        if isinstance(result, Exception):
            raise result

        return result

    def _convert_to_library_result(
        self,
        internal_result: Dict[str, Any],
        dataset_id: str,
        execution_time: float
    ) -> AnalysisResult:
        """内部結果をライブラリ結果に変換"""
        if "error" in internal_result:
            return AnalysisResult.error_result(
                dataset_id=dataset_id,
                error=internal_result["error"],
                error_details=internal_result
            )

        # 仮説の抽出
        hypotheses = []
        if "datasets" in internal_result and internal_result["datasets"]:
            dataset = internal_result["datasets"][0]

            # 仮説の抽出（内部形式から変換）
            if hasattr(dataset, 'hypotheses') and dataset.hypotheses:
                for hyp in dataset.hypotheses:
                    if isinstance(hyp, dict):
                        hypotheses.append(Hypothesis(
                            text=hyp.get('hypothesis', ''),
                            confidence=hyp.get('confidence', 0.5),
                            evidence=hyp.get('evidence', []),
                            category=hyp.get('category', 'general')
                        ))

        # レポートの取得
        report = None
        report_path = None
        if "datasets" in internal_result and internal_result["datasets"]:
            dataset = internal_result["datasets"][0]
            if hasattr(dataset, 'report_content') and dataset.report_content:
                report = dataset.report_content
                # レポートファイルパスの推定
                report_path = f"{self.config.output_dir}/results/reports/{dataset_id}_report.md"

        # プロットの取得
        plots = []
        if "datasets" in internal_result and internal_result["datasets"]:
            # プロットファイルの検索（簡易実装）
            import glob
            plot_pattern = f"{self.config.output_dir}/results/reports/plots/{dataset_id}/*.png"
            plots = glob.glob(plot_pattern)

        # メトリクスの作成
        metrics = AnalysisMetrics(
            execution_time=execution_time,
            data_points_processed=len(internal_result.get("datasets", [])),
            hypotheses_generated=len(hypotheses),
            plots_generated=len(plots)
        )

        return AnalysisResult.success_result(
            dataset_id=dataset_id,
            report=report,
            hypotheses=hypotheses,
            plots=plots,
            metrics=metrics,
            report_path=report_path,
            summary=f"Analysis completed for {dataset_id}"
        )
