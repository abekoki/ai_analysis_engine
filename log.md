# 2025-09-29
- InstanceAnalyzer 個別課題分析フローの改善項目対応を実施。図生成・代表値算出の評価区間適用、代表値／画像のプロンプト連携、出力ディレクトリ構造統一、LangGraph対話履歴保存、ログ配置変更などを完了。
- 主な修正: `utils/file_utils.py` に共通ユーティリティ追加、`core/nodes.py` / `agents/reporter_agent.py` でのフィルタリングと統計計算修正、`logger.py` でラン毎ログ設定、`main.py` の成果物書き出し整理、テンプレート更新等。
- LangGraph MemorySaver を導入し、実行コンテキストを `logs/langgraph_context/` に保存。テンプレート・プロンプトが代表値／画像リンクを参照するよう調整。
# 2025-09-30

- 個別課題成果物の出力構造を仕様に合わせて全面改修。
  - `utils/file_utils.get_report_paths()` により `report_<dataset>/images|logs|langgraph_context` を生成し、`InstanceAnalyzer`・各ノード・エージェントで利用。
  - `core/nodes.py` と `agents/reporter_agent.py` でプロット保存・レポート内リンクを `report_<dataset>/images/` 基準へ更新。
  - `instance_analyzer/main.py` で成果物を `summary/analysis_results.json` と各 `report_<dataset>/report_*.md` に書き出すよう変更。
  - `library_api` のメタパス、`orchestrator` の `summary/summary.md` 出力位置、Jinjaテンプレートの個別レポートリンクを新構成に追随。
- 既存成果物は未変更。次回以降の実行で仕様通りの階層が生成されることを期待（実行検証は未実施）。
# 2025-10-01

- `scripts/run_analysis.py` に `--max-instances` オプションを追加し、テスト用途で個別課題分析の件数を制限可能に。
- `Orchestrator`・`InstanceAnalyzer`・LangGraph `AnalysisGraph` 連携部分を改修し、`max_instances` を伝播して初回1件のみ処理できるように調整。
- 併せて `core/nodes`・`reporter_agent` のプロット生成コードで未定義変数を修正し、新ディレクトリ構成 (`outputs/analysis_report_<id>/...`) に全成果物が出力されることを確認。
- 評価ID 3 を `--max-instances 1` で実行し、`outputs/analysis_report_20251001_122809` に想定どおり `report_1_4` 以下のレポート・画像・ログ、および `summary` 配下の集計ファイルが生成されることを確認。
- `config.output_dir` が旧 `./output` を指していたため、`InstanceAnalyzer.set_output_base_dir` で `config.output_dir` をラン専用パス (`outputs/analysis_report_<timestamp>`) に上書きするよう修正。再実行により画像・ログ・LangGraph context が `outputs/analysis_report_20251001_130503/report_1_4/...` に正しく生成されることを確認。
- 評価ID 3 を `--max-instances 1` で再テスト（`outputs/analysis_report_20251001_135157`）。プロット列の動的推論は実行されるものの、閾値描画処理で `threshold_name` 参照の NameError が発生しアルゴリズム/コアプロット生成が失敗。閾値処理ロジックの修正が今後の対応課題。
# AI分析エンジン開発ログ

## 2025-09-03

- 仕様乖離点の棚卸しを実施（_input/ と docs/detailed_design_spec.md に対する実装差分を確認）。
- 主なギャップ: テンプレート未配置、ベースライン取得未実装、RAG/LLM未統合、期待値パーサ未実装、並列・リトライ未実装、DWH成果物出力の未整備、JSONLログ未対応、tests未整備 など。
- TODOリストを作成し、優先度の高い項目（テンプレート導入、エントリポイント修正、ベースライン取得、RAG/LLMフック、期待値パーサ、DWH成果物出力、テスト整備）を追加。

## 2025-09-02

- プロジェクト開始: _input/内の仕様書を確認し、詳細設計検討を開始。
- 仕様書内容:
  - 3つのエージェント: 制御・最終レポート、全体性能差分分析、個別データ分析。
  - エージェント間通信不要、各クラス化設計。
  - Pandas AIとLangchainを使用。
  - 環境管理: uv。
- uv環境セットアップ完了: pandasai, langchain-openai, matplotlib, seaborn, plotly, jinja2 をインストール。
- Pandas AIとLangchainの使用方法を確認: SmartDataframe, Agent, ChatOpenAIなどのコンポーネントを使用。
- 詳細設計仕様書作成: detailed_design_spec.md に各エージェントのクラス設計を定義。
- システム構成図とファイル構造を追加: Mermaidグラフとプロジェクトディレクトリ構造を詳細設計に追加。
- DataWareHouse連携仕様を追加: 期待値生成方法、成果物格納仕様、DB連携APIを定義。
- drowsy_detectionアルゴリズム仕様を確認: https://github.com/abekoki/drowsy_detection
- プロジェクト構造作成: src/, templates/, config/, tests/, outputs/ ディレクトリを作成。
- 基本実装開始: __init__.py, 設定管理クラス、DataWareHouse連携ユーティリティ、期待値生成ユーティリティを実装。
- Orchestratorクラス実装完了: 全体統制クラスを実装。
- PerformanceAnalyzerクラス実装完了: 全体性能分析クラスを実装。
- InstanceAnalyzerクラス実装完了: 個別データ分析クラスを実装。
- 実行スクリプト作成: scripts/run_analysis.py を作成。
- 設定ファイル作成: config/default_config.yaml を作成。
- READMEファイル作成: プロジェクト概要と使用方法を記載。
- サンプルコード実装完了: 基本的なクラス構造と実行フローが完成。

- DataWareHouse API統合完了: sqlite3から公式APIに移行。
  - algorithm_api.py: アルゴリズム出力管理
  - analysis_api.py: 分析結果・問題点管理
  - evaluation_api.py: 評価データ管理
- **実装完了**: DataWareHouse API統合により、本格的な分析システムが完成しました。

## 2025-01-27

- **マージコンフリクト解決完了**: リモートブランチとのコンフリクトを解決。
  - 状況: リモートブランチでファイル削除（プロジェクト整理）vs ローカルブランチでDataWareHouse API統合機能追加
  - 解決方針: ローカルブランチの機能追加を保持、リモートブランチの削除を無視
  - 結果: DataWareHouse API統合機能が正常に保持され、プロジェクトの機能性を維持
  - コミット: "resolve: マージコンフリクト解決 - DataWareHouse API統合機能を保持"

## 2025-09-04

- Jinja2テンプレート導入（最終・個別レポート）、`Orchestrator`連携、DWH出力同期を実装。
- LLM/RAG統合（langchain + pandasai フォールバック）とRAG参照強化（仕様/コード/外部リポ参照の自動取得）。
- 可視化拡張（ROC/PR追加、混同行列の標準化、棒グラフの値ラベル修正）。
- インスタンス分析フィルタリング追加（課題なしのデータは除外可能）。
- 実行: `scripts/run_analysis.py -e 3 -v` 正常終了。レポート・図表・JSON生成およびDWH登録完了。
- 外部仕様参照: `https://github.com/abekoki/drowsy_detection` を `external/drowsy_detection` に取得/更新して参照。

2025-10-01 15:26 作業内容:
- reporter_agent と core.nodes の `get_report_paths` 呼び出しを `report_<dataset>` に統一し、旧 `dataset_id` ディレクトリの生成を防止
- `_save_results` を修正して各データセットごとの `analysis_results.json` をレポートディレクトリへ保存するよう変更し、summary 側で集約
- LangGraph コンテキストおよびログの保存先を `report_<dataset>` 配下に揃え、viz 画像も同様に統合
- `reporter_agent.py` のインデント不備を修正し、テストとして `uv run python scripts/run_analysis.py -e 3 --max-instances 1` を実行