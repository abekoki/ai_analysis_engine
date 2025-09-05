2025-09-05: 個別データ分析レポート生成をサンプル仕様に合わせて更新。
- `DataWareHouse`タグ単位の行に出力ディレクトリ等を付与（`build_task_level_dataframe`）。
- タスクレベルのレポートで、アルゴリズム出力とコア出力の区間グラフを自動生成。
- テンプレートをサンプルのレイアウト（結論/動画リンク/フレーム区間/期待・検知/確認結果）に対応。
- IOサマリと考えられる原因の自動記述を追加。
2025-09-05: LLM強制モードを導入（ヒューリスティック解析を無効化）。
- `config/default_config.yaml` に `instance_analyzer.require_llm: true` を追加。
- `RAGSystem` を更新し、LLM未利用時は例外を送出。pandasai未導入時はlangchain+ChatOpenAIで要約を実施。
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