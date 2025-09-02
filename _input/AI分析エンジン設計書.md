# AI分析エンジン設計書


## 📋 概要

AI分析エンジンは、アルゴリズム評価結果を自動分析し分析レポートを生成するシステムです。

本エンジンは次の3つのエージェントの組み合わせで構成されます。

1. 制御・最終レポートエージェント（オーケストレータ）
   - エージェント2/3を統括し、実行順序・入出力・失敗時リトライ・要約を管理
   - 最終的な統合レポート（Markdown）の生成・体裁調整・確認

2. 全体性能の確認・差分分析エージェント（パフォーマンスアナライザ）
   - 評価データセット全体の正解率・過検知率・未検知率・速度などを集計
   - 前回実行やベースラインとの比較（差分）を可視化し、傾向・回帰の有無を判定

3. 個別データの分析エージェント（インスタンスアナライザ）
   - 課題データ（個票）ごとの詳細分析を実施
   - 仕様書・ソースコード・期待値を参照しつつ、仮説→検証→ループバックで原因特定を支援

## 🎯 機能要件

### 主要機能
1. **自動評価結果分析**
2. **レポート**
3. **課題の優先度付け**
4. **全体性能差分レポート**（エージェント2）
5. **個別データ詳細レポート**（エージェント3）

### レポート構成
1. **サマリセクション**
   - 全体評価スコア（データセット全体の正解率、過検知数(未実装)）
   - 主要課題の要約
   - 改善優先度ランキング
   - 推奨アクション

2. **個別分析結果**
   - 時系列分析結果（対象データの時系列グラフ（閾値や、検知フレームにマークを追加するなどわかりやすい図を生成））
   - エラー分析結果（誤検知・未検知の原因を特定）


## 🏗️ 技術仕様

### 使用技術
- **AI API**: OpenAI GPT
- **データ処理**: Python + pandas + numpy
- **可視化**: matplotlib + seaborn + plotly
- **データ管理**: データセットはDataWareHouseで管理
- **データ分析**: データ分析はlangchainを使用。必要なツール類は詳細仕様で定義。
- **レポート形式**: マークダウン+jpgによる図


### システム構成

```mermaid
graph TB
    subgraph "AI分析エンジン アーキテクチャ"
        subgraph "入力層"
            I1[評価結果データ<br/>CSV/JSON]
            I2[仕様書<br/>Markdown]
            I3[ソースコード<br/>Python]
            I4[期待値<br/>自然言語]
        end
        
        subgraph "制御・最終レポートエージェント（オーケストレータ）"
            ORC[orchestrator.py<br/>実行順序管理]
            SCH[scheduler<br/>並列実行]
            RTY[retry_manager<br/>リトライ制御]
            ART[artifact_registry<br/>成果物管理]
        end
        
        subgraph "全体性能差分分析エージェント"
            PA1[summary_metrics.py<br/>指標計算]
            PA2[diff_analyzer.py<br/>ベースライン比較]
            PA3[visualizations.py<br/>図表生成]
            PA4[exporter.py<br/>JSON/PNG出力]
        end
        
        subgraph "個別データ分析エージェント"
            IA1[期待値エージェント<br/>自然言語処理]
            IA2[解析エージェント<br/>仮説生成・検証]
            IA3[レポートエージェント<br/>Markdown生成]
        end
        
        subgraph "共通基盤"
            DB[(DataWareHouse<br/>database.db)]
            RAG[RAG<br/>FAISS+OpenAI]
            AI[AI API<br/>OpenAI GPT]
            VIS[可視化<br/>matplotlib+plotly]
        end
        
        subgraph "出力層"
            O1[最終レポート<br/>Markdown]
            O2[図表<br/>charts/]
            O3[データ<br/>JSON]
            O4[ログ<br/>実行履歴]
        end
    end
    
    %% データフロー
    I1 --> ORC
    I2 --> RAG
    I3 --> RAG
    I4 --> IA1
    
    ORC --> PA1
    ORC --> IA1
    
    PA1 --> PA2
    PA2 --> PA3
    PA3 --> PA4
    
    IA1 --> IA2
    IA2 --> IA3
    
    %% 共通基盤への接続
    PA1 -.-> DB
    PA2 -.-> VIS
    PA3 -.-> VIS
    IA1 -.-> AI
    IA2 -.-> RAG
    IA2 -.-> AI
    IA3 -.-> VIS
    
    %% 出力
    PA4 --> O2
    PA4 --> O3
    IA3 --> O1
    ORC --> O1
    ORC --> O4
    
    %% スタイル
    classDef inputClass fill:#e3f2fd
    classDef orchestratorClass fill:#f3e5f5
    classDef performanceClass fill:#e8f5e8
    classDef instanceClass fill:#fff3e0
    classDef commonClass fill:#fce4ec
    classDef outputClass fill:#f1f8e9
    
    class I1,I2,I3,I4 inputClass
    class ORC,SCH,RTY,ART orchestratorClass
    class PA1,PA2,PA3,PA4 performanceClass
    class IA1,IA2,IA3 instanceClass
    class DB,RAG,AI,VIS commonClass
    class O1,O2,O3,O4 outputClass
```

### データセット・出力の保管ポリシー
- 本プロジェクトで扱うデータセットおよび各エンジンの出力は、`database.db` を基準とした相対パスで一元管理します。
- 評価・分析・可視化で生成される成果物（CSV/JSON/md/HTML 等）も、原則 `(database.dbの格納ディレクトリ)/05_analysis_output/` 配下に相対パスで保存します。


## 🔄 処理フロー

1. **評価結果データ読み込み**
2. **全体性能の確認・差分分析（エージェント2）**
3. **個別データの詳細分析（エージェント3）**
4. **全体性能中間レポート生成**
5. **個別データ中間レポート生成**
6. **最終レポートにまとめて統合出力**

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant ORC as オーケストレータ
    participant PA as 全体性能分析
    participant IA as 個別データ分析
    participant DB as DataWareHouse
    participant AI as AI API
    participant VIS as 可視化エンジン
    
    Note over User, VIS: 1. 初期化・設定読み込み
    User->>ORC: 分析実行要求
    ORC->>ORC: 設定読み込み（並列度、しきい値）
    ORC->>DB: 評価結果データ取得
    
    Note over User, VIS: 2. 全体性能差分分析（エージェント2）
    ORC->>PA: 全体分析開始
    PA->>DB: 評価データ・ベースライン取得
    PA->>PA: 指標計算・差分算出
    PA->>VIS: 図表生成（時系列、混同行列、ROC）
    PA->>ORC: 集計結果・図表・差分レポート
    
    Note over User, VIS: 3. 個別データ分析（エージェント3）並列実行
    ORC->>IA: 個別分析タスク配布（並列）
    
    loop 各データに対して並列実行
        IA->>AI: 期待値解釈（自然言語→構造化）
        IA->>DB: 個別データ取得
        IA->>AI: 仮説生成（RAG活用）
        IA->>AI: 動的解析実行
        
        alt 仮説検証成功
            IA->>VIS: 個別図表生成
            IA->>IA: 個別レポート作成
        else 仮説検証失敗
            IA->>AI: 仮説再生成（最大3回）
            IA->>AI: 再解析実行
        end
        
        IA->>ORC: 個別分析結果
    end
    
    Note over User, VIS: 4. 統合・最終レポート生成
    ORC->>ORC: 結果統合・優先度付け
    ORC->>VIS: 最終レポート生成（Markdown）
    ORC->>DB: 分析結果保存
    ORC->>User: 最終レポート配信
    
    Note over User, VIS: 5. エラーハンドリング・リトライ
    alt 処理失敗時
        ORC->>ORC: リトライ管理（指数バックオフ）
        ORC->>User: エラー通知・部分結果提供
    end
```

---

## 3. 個別データの分析エージェント（詳細仕様）

本章の詳細仕様はボリュームが大きいため、専用ドキュメントに分離。

- 参照: `projects/詳細設計資料/AI分析エンジン/個別データ分析エージェント仕様.md`
- 参照: `projects/詳細設計資料/AI分析エンジン/制御・最終レポートエージェント仕様.md`
- 参照: `projects/詳細設計資料/AI分析エンジン/全体性能差分分析エージェント仕様.md`

### 3.1 目的と特徴
- 自然言語の期待値・仕様書・ソースコードを参照し、個票データの異常箇所を特定
- 仮説→検証→ループバックの自律サイクルで原因特定を支援（最大試行回数を設定）
- 解析は AI を用いた動的コード生成で実施し、根拠を出力

### 3.2 入力
- データ: アルゴリズムの出力結果が記載された時系列のCSV （例: `frame`, `predict_result` など）
- データ: アルゴリズムの入力データが記載された時系列のCSV （例: `frame`, `input_data` など）
- 仕様書: Markdown（アルゴリズム仕様）
- ソースコード: Python 等（必要に応じて AST 解析）
- 期待値: 自然言語（テンプレート＋LLM で構造化 JSON に変換）（未実装）

### 3.3 出力
- 個別レポート（解析データ１つにつき１つのマークダウンファイル）
  - 異常箇所（場所・検出値・期待値・差分）
  - 仮説（例: 条件分岐ミス、パラメータ不整合、データ不備）
  - 可視化（jpgによる図をマークダウンにリンク）

### 3.4 処理の流れ

```mermaid
flowchart TD
    Start([個別データ分析開始]) --> A1[期待値エージェント]
    
    subgraph "期待値エージェント"
        A1 --> A2{自然言語期待値}
        A2 -->|定型句| A3[テンプレート解析]
        A2 -->|非定型| A4[LLM解析]
        A3 --> A5[構造化JSON生成]
        A4 --> A5
    end
    
    A5 --> B1[解析エージェント]
    
    subgraph "解析エージェント（仮説・検証ループ）"
        B1 --> B2[1. 結果確認]
        B2 --> B3[仕様書・コード参照<br/>RAG検索]
        B3 --> B4[2. データ収集]
        B4 --> B5[入力データ特定・抽出<br/>時系列可視化]
        B5 --> B6[3. 仮説検討]
        B6 --> B7[input-output整合確認]
        B7 --> B8[4. 仮説検証]
        B8 --> B9[境界条件・欠損処理<br/>座標変換解析]
        B9 --> B10{仮説検証}
        
        B10 -->|成功| B11[5. 課題出力]
        B10 -->|失敗| B12{試行回数<br/>< 3回?}
        B12 -->|Yes| B6
        B12 -->|No| B13[最終仮説採用]
        B13 --> B11
    end
    
    B11 --> C1[レポートエージェント]
    
    subgraph "レポートエージェント"
        C1 --> C2[異常箇所特定]
        C2 --> C3[可視化図表生成<br/>Plotly/matplotlib]
        C3 --> C4[個別レポートMD生成]
        C4 --> C5[仮説・根拠記載]
    end
    
    C5 --> End([個別分析完了])
    
    %% スタイル
    classDef agentClass fill:#e3f2fd,stroke:#1976d2
    classDef processClass fill:#e8f5e8,stroke:#388e3c
    classDef decisionClass fill:#fff3e0,stroke:#f57c00
    classDef outputClass fill:#f1f8e9,stroke:#689f38
    
    class A1,B1,C1 agentClass
    class A3,A4,A5,B2,B3,B4,B5,B6,B7,B8,B9,B11,B13,C2,C3,C4,C5 processClass
    class A2,B10,B12 decisionClass
    class Start,End outputClass
```

---

## 📈 出力例

### レポートファイル構成
```
[analysis_ID]_[date]/
├── analysis_report_YYYYMMDD_HHMMSS.md
├── data1_[data_ID].md
├── data2_[data_ID].md
├── ...
├── images/
│   ├── time_series.jpg
│   ├── confusion_matrix.jpg
│   ├── roc_curve.jpg
│   └── error_heatmap.jpg
└── data/
    ├── analysis_summary.json
    └── improvement_suggestions.json
```


### サマリレポート出力サンプル
```markdown
# 分析サマリー

## 実行概要
- **分析日時**: 2025-07-25 14:30:00
- **対象アルゴリズム**: eye_detection
- **アルゴリズムバージョン**: v1.0
- **評価回数**: 20回（4人×5回）
- **比較対象**: v0.9

## 主要結果
- **正解率**: 85.2% (前回比 +5.3%))
- **過検知数**: xx回/h (前回比 -0.2回/h)

## 主要課題
| 課題 | 原因 | 詳細 |
|------|------|------|
| 過検知数が多い | 閾値設定が不適切 | 眼の閉じ具合の判定閾値が低すぎるため、軽微な瞬きも居眠りとして検出 |
| 検出遅延が発生 | フレーム処理の遅延 | 連続フレーム解析時のバッファリング処理が重く、リアルタイム性が低下 |
| 精度のばらつき | 環境光の影響 | 明暗差が大きい環境下で瞳孔検出精度が不安定 |


## 個別データ分析結果
(各個別データ分析結果のマークダウンファイルへのリンクを記載)


```

## 🔧 設定項目

### AI分析パラメータ
- 分析深度レベル（Basic/Standard/Detailed）
- 可視化の詳細度
- レポートの出力形式

### カスタマイズ項目
- 分析項目の追加・削除
- 可視化チャートのカスタマイズ
- レポートテンプレートの変更



## 📝 更新履歴

- **2025-07-25**: 初版作成（v1.0）
- **2025-08-08**: 3エージェント構成を定義し、個別データ分析エージェントの詳細仕様を統合（v1.1）
- **2025-08-17**: 3.6 入出力例を「フレーム x~y は項目 z が w であるべき」に基づく現実的ワークフローへ更新（v1.2）
- **2025-08-29**: システム構成図・処理フロー図を追加、改善提案を実装重視に整理して資料をv1.3に完成（v1.3）