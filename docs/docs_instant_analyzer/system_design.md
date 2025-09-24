# 時系列処理アルゴリズムの課題分析システム 仕様書 (v3.0)

## 変更履歴
このセクションでは、仕様書のバージョン履歴を記載。変更は日付、バージョン、変更内容、理由を明記。

| 日付          | バージョン | 変更内容                                                                 | 理由/追加要求                                                                 |
|---------------|------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 2025-09-10   | v1.0      | 初期仕様書作成（概要、アーキテクチャ、プロセスフロー、報告フォーマット等）。 | 初回クエリに基づくシステム設計。                                              |
| 2025-09-10   | v1.1      | 入力形式明確化（CSV/Markdown）、期待値自然言語、報告フォーマット指定、RAG事前インデックス、REPLセキュリティなし、スケーラビリティ不要等。評価環境input追加（Markdown仕様/Pythonコード）。複数データ前提、RAGベクトル化タイミング/セグメント化。解析具体例統合。報告フォーマット更新。 | ユーザーの質問事項回答と追加要求（評価環境、複数データ、解析例、レポートフォーマット）。 |
| 2025-09-10   | v1.2      | Supervisorの複数データ処理を逐次に変更（並列不要）。 | ユーザー追加要求（並列不要）。                                                |
| 2025-09-10   | v1.3      | 全要求統合、齟齬解消。 | ユーザー再作成要求（要求整理）。                                              |
| 2025-09-11   | v3.0      | 変更履歴欄追加、全履歴まとめ。内容はv1.3を基に統合/確認。仕様書と詳細設計書を分離。 | ユーザー要求（変更履歴欄作成、仕様書と詳細設計書分離）。                      |

## 1. 概要
この仕様書は、時系列処理アルゴリズムの課題分析をAIを用いて実施するシステムの設計を定義する。ユーザーの全追加要求を整理・統合して再作成。主なポイント:
- 入力データの形式: 時系列データはCSV (エンコーディング: UTF-8)、仕様書はMarkdown。
- 期待値: 自然言語入力 (例: "フレームxからyまでに列zに1が存在すること")。
- 報告フォーマット: 指定テンプレートを使用。プロットの複数種類 (アルゴリズム出力/コア出力の時系列グラフ) 対応。追加セクション不要。
- RAG知識ベース: 事前インデックス (初期化時)。外部DB不使用。ベクトル化タイミング: 初期化時またはベクトルデータ不存在時のみ。アルゴリズム・エンジン種類ごとにセグメント化ベクトル化。
- REPLセキュリティ: 制限なし、uv環境内実行。
- スケーラビリティ/エラー通知/国際化: 不要。
- 新規入力追加: アルゴリズム評価用環境の仕様 (複数Markdown) と対応ソースコード (Python)。内容: コアライブラリ出力の入力方法、CSV出力仕様。
- 複数データ分析前提: 複数のアルゴリズム出力結果を解析。処理は逐次 (並列不要)。
- 解析具体(mathjax_start)例: 統合し、実現可能。入力受け取り、初期化、解析開始 (入出力確認、課題検証ループ)、レポート作成をフローに反映。RAG/REPL/mermaidを活用。

目的: 複数時系列データ (例: 動画フレームベース) のアルゴリズム課題を自動分析。バグ/不整合/性能問題を検出・報告。

適用範囲: 時系列データ限定。複数データセット逐次対応。

前提条件:
- 入力: CSV (UTF-8, 列: timestamp, value 等)。Markdown: UTF-8。
- 期待値: 自然言語文字列。
- システム: Python、LangGraph v0.1.x+、LangChain v0.2.x+。
- 複数データ: リスト形式 (例: [{"output_csv": "file1.csv", "core_csv": "core1.csv", ...}])。

## 2. システムアーキテクチャ
LangGraphのSupervisorアーキテクチャ基盤。複数データ逐次処理。RAGベクトル化条件付き。

### 2.1 高レベルアーキテクチャ図
<xaiArtifact artifact_id="0a66ee7f-cca9-4e98-8bb6-30caa4c777fc" artifact_version_id="fb761d9e-277c-4e9d-8614-4b82f3e0e466" title="architecture.mmd" contentType="text/mermaid">
graph TD
    START[Start] --> Init[Initialization: Vectorize if needed]
    Init --> Supervisor[Supervisor Agent]
    Supervisor -->|Sequential for each Dataset| DataChecker[Data Checker Agent]
    Supervisor -->|Route to| ConsistencyChecker[Consistency Checker Agent]
    Supervisor -->|Route to| HypothesisGenerator[Hypothesis Generator Agent]
    Supervisor -->|Route to| Verifier[Verifier Agent]
    Supervisor -->|Route to| Reporter[Reporter Agent]
    DataChecker -->|Update State| Supervisor
    ConsistencyChecker -->|Update State| Supervisor
    HypothesisGenerator -->|Update State| Supervisor
    Verifier -->|Success? Yes| Supervisor
    Verifier -->|Success? No| HypothesisGenerator[Loop Back]
    Reporter --> END[End]
    subgraph Tools
        RAG[RAG Tool<br>Segmented by Algo/Engine]
        REPL[REPL Tool<br>uv Env, Data Plot]
    end
    DataChecker --> RAG
    DataChecker --> REPL
    ConsistencyChecker --> RAG
    Verifier --> REPL
    Verifier --> RAG
    subgraph Multi-Data
        Supervisor -->|Sequential| Dataset1
        Supervisor -->|Sequential| DatasetN
    end
</xaiArtifact>

- **説明**: Initでベクトル化チェック。Supervisor逐次ループ。Tools複数データ対応。

### 2.2 詳細コンポーネント
#### 2.2.1 Initialization Node
- **役割**: RAG構築。存在チェック後、不存在時ベクトル化。
- **入力**: ドキュメント (仕様書/コード/評価環境仕様/コード)。
- **出力**: セグメントベクトルストア (例: {"engine_A": VectorStore})。
- **実装**: FAISS/Chroma、OpenAI Embeddings。メタデータ分類。
- **図: 初期化フロー**
<xaiArtifact artifact_id="14192e1d-abbe-4153-b8df-be575652b408" artifact_version_id="f01bf29b-3737-43e0-9321-3110643d87a7" title="initialization_flow.mmd" contentType="text/mermaid">
flowchart TD
    A[Start] --> B[Check Exists?]
    B -->|Yes| C[Load]
    B -->|No| D[Segment by Algo/Engine]
    D --> E[Embed & Index]
    E --> F[Save]
    C --> G[Update State]
    F --> G
</xaiArtifact>

#### 2.2.2 Supervisor Agent
- **役割**: 制御。複数データ逐次 (for ds in datasets: sub_state; route; wait)。
- **入力**: State (入力リスト)。
- **出力**: Command (goto with ds_id)。
- **プロンプト**: "状態: {state}. DS{ds_id}次ステップ決定。"
- **ツール**: なし。
- **図: 逐次処理**
<xaiArtifact artifact_id="416c5f19-7a75-41e5-aa14-746f82bb41d5" artifact_version_id="83ef646d-50b7-402e-a132-8a95cdd3efd7" title="sequential_processing.mmd" contentType="text/mermaid">
sequenceDiagram
    Supervisor->>DS1: Process
    DS1-->>Supervisor: Complete
    Supervisor->>DS2: Process
    DS2-->>Supervisor: Complete
</xaiArtifact>

#### 2.2.3 Data Checker Agent
- **役割**: 入出力確認。評価環境から列詳細抽出。CSV概要 (describe/info)。プロット。
- **入力**: DSごとCSV、評価環境、期待値。
- **出力**: 結果/プロット。
- **ツール**: REPL (pd.read_csv; df.describe; plt.plot.savefig), RAG (列クエリ)。
- **プロンプト**: "環境: {env}. データ: {csv}. 確認。"

#### 2.2.4 Consistency Checker Agent
- **役割**: 整合調査。自然言語期待値解析 (正規表現/LLM)。
- **入力**: 出力、期待値、環境。
- **出力**: レポート。
- **ツール**: RAG, REPL (df.query)。

#### 2.2.5 Hypothesis Generator Agent
- **役割**: 仮説設定 (例: 仕様不整合, パラメータ不適切, 想定外動作)。
- **入力**: 結果、不整合。
- **出力**: JSONリスト。
- **ツール**: RAG。
- **プロンプト**: "不整合: {issues}. 仮説: 不整合/パラメータ/想定外。"

#### 2.2.6 Verifier Agent
- **役割**: 検証。REPLテスト (コード実行/コア比較)。ループ (max設定)。
- **入力**: 仮説, コード, データ。
- **出力**: 結果 (成功/詳細)。
- **ツール**: REPL, RAG。
- **図: ループ**
<xaiArtifact artifact_id="885c9896-456b-4dde-925a-404d7e4c1f66" artifact_version_id="8f5a98be-a552-4bad-beba-16f1bdc43887" title="verification_loop.mmd" contentType="text/mermaid">
graph TD
    Start --> Set[Set Hypo]
    Set --> Verify[REPL Test]
    Verify --> Check[Success?]
    Check -->|Yes| Report
    Check -->|No| Update[Update Hypo]
    Update -->| < Max| Set
    Update -->|Max| Report
</xaiArtifact>

#### 2.2.7 Reporter Agent
- **役割**: DSごとレポート。mermaid図/REPLグラフ/説明追加。
- **入力**: State。
- **出力**: Markdown。
- **ツール**: REPL。

### 2.3 データフロー図
<xaiArtifact artifact_id="2c3110fd-31fb-482d-93a5-a1ba5a197974" artifact_version_id="db3825e7-0a3a-42a5-b2d7-9e45a158c033" title="data_flow.mmd" contentType="text/mermaid">
graph TD
    Inputs[Inputs: Multi Outputs, Core, etc.] --> Init[Vectorize]
    Init --> State[State]
    State -->|Sequential DS| DataChecker
    State --> ConsistencyChecker
    State --> HypothesisGenerator
    State --> Verifier
    State --> Reporter
    Reporter --> Outputs[Reports]
    subgraph Loop
        HypothesisGenerator <--> Verifier
    end
</xaiArtifact>

## 3. プロセスフロー (具体例統合: 実現可能)
1. **入力受け取り**: データロード (仕様書/コード/環境仕様/コード/出力/コア出力/期待値)。
2. **初期化**: ベクトル化 (仕様書/コード/環境仕様/コード)。
3. **解析開始** (逐次DS):
   - **入出力確認**: 環境から列確認。REPL概要/プロット (対象区間/課題再現/コア)。
   - **課題検証**: 仮説設定 (結果/仕様照合)。検証ループ (REPL判断、成功→レポート、失敗→更新、maxループ)。
4. **レポート作成**: フォーマット従い、mermaid (原因図)/REPL (説明) 使用。

- エラー: ループ制限。
- 状態: Persistent。

## 4. 報告Markdownフォーマット
<xaiArtifact artifact_id="7149f839-17ac-43ad-8c18-cb77cb810d23" artifact_version_id="5c145bf8-3a3d-45de-b7c9-69bb15bf1cf0" title="report_format.md" contentType="text/markdown">
<!-- これは個別データ分析レポートのサンプルです。 -->

# 個別データ分析レポート

## 概要

- 結論 : <!--結論は仮説検証にて確認した結果を記載-->
- 解析対象動画： <!-- videoデータが格納されているフォルダをリンク表記 -->
- フレーム区間: <!--評価区間を記載-->
- 期待値：<!--評価結果から取得-->
- 検知結果： <!--評価結果から取得-->

## 確認結果

![アルゴリズム出力結果の当該タスク区間における時系列グラフ](images/xxx.png)
アルゴリズム出力結果

![コア出力結果の当該タスク区間における時系列グラフ](images/xxx.png)
コア出力結果

- 入出力の確認結果：<!-- 入出力の確認結果を記載 -->

- 考えられる原因1 : <!-- 考えられる課題要因を図や背景を交えて列挙 -->

## 推奨事項

- <!-- 推奨事項を記載 -->

## 参照した仕様/コード（抜粋）
... <!-- 仮説検証にて参照した仕様/コードを記載-->
</xaiArtifact>

- **拡張**: 原因にmermaid (例: mindmap原因ツリー)。グラフ: REPL PNG。
- **図: 構造**
<xaiArtifact artifact_id="0331d3d2-abc0-4d98-aa7a-1fa2cd44cf8f" artifact_version_id="c8a0b70e-20f3-4583-a3ba-8e40411a6a5d" title="report_structure.mmd" contentType="text/mermaid">
mindmap
    root((レポート))
      概要[概要]
        結論
        動画
        区間
        期待
        検知
      確認[確認結果]
        グラフ1
        グラフ2
        確認
        原因
      推奨[推奨事項]
      参照[参照抜粋]
</xaiArtifact>

## 5. 実装詳細
- **LangGraph例** (擬似コード):
<xaiArtifact artifact_id="7609b1f1-060e-4a63-9f63-24c3d21b138f" artifact_version_id="6cf14e40-c8cd-4b80-b510-8288694e0c98" title="langgraph_example.py" contentType="text/python">
from langgraph.graph import StateGraph, MessagesState
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

model = ChatOpenAI()
embeddings = OpenAIEmbeddings()
rag_tool = Tool(name="RAG", func=lambda q, seg: vectorstores[seg].similarity_search(q))
repl_tool = Tool(name="REPL", func=code_execution)  # uv

def init(state):
    if not exists():
        docs = segment(state['inputs'])
        for seg, dl in docs.items():
            vectorstores[seg] = FAISS.from_documents(dl, embeddings)
        save()
    return state

def supervisor(state):
    for ds in state['datasets']:  # 逐次
        sub = state.copy()
        sub['ds'] = ds
        state['results'].append(process(sub))  # wait
    return state

builder = StateGraph(MessagesState)
builder.add_node("init", init)
builder.add_node("supervisor", supervisor)
# 他...
builder.set_entry_point("init")
builder.add_conditional_edges("verifier", lambda s: "hypothesis_generator" if not s["success"] else "reporter")
graph = builder.compile(checkpointer=True)
</xaiArtifact>

- **RAG**: セグメントクエリ。
- **REPL**: pd.concat 等。

## 6. テストと検証
- ユニット: エージェント/初期化。
- E2E: 複数データ逐次。
- 辺境: ベクトル不存在、ループmax。

## 7. 追加の疑問点
なし。全要求統合。追加時再作成可能。
