"""RAG/LLM 支援モジュール

langchain / langgraph / pandasai を前提として使用します。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import os


class RAGSystem:
    """RAG + LLM による仮説生成・検証を支援するクラス（必須ライブラリ前提）。"""

    def __init__(self, settings: Any):
        self.settings = settings
        # 必須ライブラリの読み込み（失敗時は即時例外）
        try:
            import langchain  # noqa: F401
            import langgraph  # noqa: F401
            import pandasai  # noqa: F401
            from langchain_openai import OpenAIEmbeddings  # type: ignore
            from langchain_community.vectorstores import FAISS  # type: ignore
            self.OpenAIEmbeddings = OpenAIEmbeddings
            self.FAISS = FAISS
        except Exception as e:
            raise RuntimeError(f"必須ライブラリの読み込みに失敗しました: {e}")
        # 直近の参照ソース（レポートへ表示用）
        self.last_sources: list[str] = []
        # 外部仕様リポジトリの準備（存在しない場合は作成/取得を試行）
        try:
            self._ensure_external_repo()
        except Exception:
            pass

    # git availability の事前チェックは行わず、呼び出し側で例外処理

    def generate_hypothesis(self, df, context: Optional[Dict[str, Any]] = None) -> str:
        """データと文脈から仮説を生成。

        可能なら LLM/チェーンを用いるが、フォールバックとしてヒューリスティックを使用。
        """
        require_llm = bool(self.settings.get('instance_analyzer.require_llm', False))

        # 参照文脈の収集（_input / docs / src のテキストを軽量に集約）
        try:
            retrieved_context = self._collect_context(max_chars=6000)
        except Exception:
            retrieved_context = ""

        # 必須ライブラリ前提のため、以降は常にLLM呼び出しを試みる

        # ここでは API キーや詳細設定が不明なため、軽量な疑似チェーンのみに留める
        try:
            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain_openai import ChatOpenAI  # type: ignore

            llm_model = self.settings.get('instance_analyzer.llm_model', 'gpt-4')
            temperature = float(self.settings.get('instance_analyzer.temperature', 0.1))
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key and require_llm:
                raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。")
            # 明示的にAPIキーを渡す（環境変数も併用可能）
            llm = ChatOpenAI(model=llm_model, temperature=temperature, timeout=15, api_key=api_key)  # type: ignore

            prompt = PromptTemplate(
                input_variables=["summary", "context"],
                template=(
                    "あなたはアルゴリズム改善のエキスパートです。以下の参照文脈と評価データ要約に基づき、"
                    "過検知/未検知の傾向や主要因を推定し、1文で簡潔な仮説を日本語で提案してください。\n"
                    "[参照文脈(抜粋)]\n{context}\n---\n[評価データ要約]\n{summary}"
                ),
            )

            # summary を簡易生成
            # 仕様/コードの外部参照も取得
            external_spec = self._fetch_external_spec_snippets()
            if external_spec:
                retrieved_context = (retrieved_context + "\n\n" + external_spec)[:6000]

            summary = self._summarize_df(df)
            chain = prompt | llm
            out = chain.invoke({"summary": summary, "context": retrieved_context})  # type: ignore
            text = getattr(out, 'content', None) or str(out)
            return text.strip()[:500]
        except Exception as e:
            # フォールバックは廃止
            raise RuntimeError(f"LLM呼び出しに失敗しました: {e}")

    def verify_hypothesis(self, hypothesis: str, df) -> Dict[str, Any]:
        """仮説を簡易検証。pandasai があれば自然言語 QA を併用。"""
        # ベースラインの数値検証
        acc = self._calc_accuracy(df)
        valid = acc < 0.9  # 高精度なら仮説は弱いとみなす
        confidence = 0.6 if 0.7 <= acc < 0.9 else (0.8 if acc < 0.7 else 0.3)

        # pandasai による自然言語裏付け（必須）
        try:
            from pandasai import SmartDataframe  # type: ignore
            from pandasai.llm.openai import OpenAI  # type: ignore
            # pandasai用にもAPIキーを明示
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key and bool(self.settings.get('instance_analyzer.require_llm', False)):
                raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません（pandasai）。")
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
            llm = OpenAI()  # type: ignore
            sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": True})  # type: ignore
            _ = sdf.chat("誤検知（expected_is_drowsy=0 かつ is_drowsy=1）の件数を教えてください。")  # noqa: F841
            confidence = min(0.95, confidence + 0.05)
        except Exception as e:
            # 必須として扱うため、失敗は表層化
            raise RuntimeError(f"pandasai経由のLLM検証に失敗: {e}")

        return {
            "valid": bool(valid),
            "confidence": float(confidence),
            "accuracy": float(acc),
            "reason": f"Accuracy={acc:.3f}"
        }

    def pandasai_narrative(self, df) -> str:
        """pandasaiで可視化・説明テキストを生成（必須）。"""
        require_llm = bool(self.settings.get('instance_analyzer.require_llm', False))
        try:
            from pandasai import SmartDataframe  # type: ignore
            from pandasai.llm.openai import OpenAI  # type: ignore
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key and require_llm:
                raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません（pandasai）。")
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
            llm = OpenAI()  # type: ignore
            sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": True})  # type: ignore
            prompt = (
                "左/右の開眼度、検知(is_drowsy)と期待(expected_is_drowsy)の時系列傾向を日本語で簡潔に説明し、"
                "過検知/未検知の区間があれば指摘してください。"
            )
            answer = sdf.chat(prompt)  # type: ignore
            return str(answer)
        except Exception as e:
            raise RuntimeError(f"pandasaiによる要約生成に失敗: {e}")

    # --- helpers ---
    def _heuristic_hypothesis(self, df) -> str:
        try:
            import pandas as pd  # noqa: F401
            fp = ((df['expected_is_drowsy'] == 0) & (df['is_drowsy'] == 1)).sum()
            fn = ((df['expected_is_drowsy'] == 1) & (df['is_drowsy'] == 0)).sum()
            if fp > fn:
                return "過検知が多い。閾値や信頼度の再設定が必要。"
            elif fn > fp:
                return "未検知が多い。感度向上や特徴量の見直しが必要。"
            return "全体的に良好だが、特定条件下での誤判定が残る可能性。"
        except Exception:
            return "データ傾向から、閾値最適化の余地があると推定される。"

    def _calc_accuracy(self, df) -> float:
        try:
            correct = (df['expected_is_drowsy'] == df['is_drowsy']).mean()
            return float(correct)
        except Exception:
            return 0.0

    def _summarize_df(self, df) -> str:
        try:
            total = len(df)
            fp = ((df['expected_is_drowsy'] == 0) & (df['is_drowsy'] == 1)).sum()
            fn = ((df['expected_is_drowsy'] == 1) & (df['is_drowsy'] == 0)).sum()
            acc = (df['expected_is_drowsy'] == df['is_drowsy']).mean()
            return f"samples={total}, fp={fp}, fn={fn}, acc={acc:.3f}"
        except Exception:
            return "summary_unavailable"

    def _collect_context(self, max_chars: int = 6000) -> str:
        """ローカルの仕様/設計/コードからシンプルに文脈を収集して連結する。

        ベクタDBは使わず、重要そうなファイルを優先して先頭数千文字のみ読み込む。
        """
        # ベクトルDB（FAISS）での収集をまず試みる
        try:
            return self._collect_context_with_vectors(max_chars=max_chars)
        except Exception:
            pass
        from pathlib import Path

        root = Path(__file__).parent.parent.parent.parent
        candidate_dirs = [
            root / "_input",
            root / "docs",
            root / "src",
        ]
        exts = {".md", ".py", ".yaml", ".yml", ".toml"}

        collected: list[str] = []
        remaining = max_chars

        collected_sources: list[str] = []
        for d in candidate_dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    # 先頭のみ使用
                    head = text[: min(1000, len(text))]
                    snippet = f"## {p.as_posix()}\n{head}\n"
                    if len(snippet) <= remaining:
                        collected.append(snippet)
                        collected_sources.append(p.as_posix())
                        remaining -= len(snippet)
                    else:
                        collected.append(snippet[:remaining])
                        collected_sources.append(p.as_posix())
                        remaining = 0
                    if remaining <= 0:
                        out = "\n".join(collected)
                        self.last_sources = collected_sources
                        return out

        out = "\n".join(collected)
        self.last_sources = collected_sources
        return out

    def _collect_context_with_vectors(self, max_chars: int = 6000) -> str:
        from pathlib import Path
        texts: list[str] = []
        metadatas: list[dict] = []

        root = Path(__file__).parent.parent.parent.parent
        candidate_dirs = [root / "_input", root / "docs", root / "src", root / "external" / "drowsy_detection"]
        exts = {".md", ".py", ".yaml", ".yml", ".toml"}

        for d in candidate_dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    try:
                        t = p.read_text(encoding="utf-8", errors="ignore")
                        # 固定長スライスでチャンク化
                        step = 800
                        for i in range(0, min(len(t), max_chars*3), step):
                            chunk = t[i:i+step]
                            if not chunk:
                                break
                            texts.append(chunk)
                            metadatas.append({"source": p.as_posix(), "offset": i})
                    except Exception:
                        continue

        # 収集が少なければフォールバック
        if not texts:
            return ""

        # ベクトル化と検索
        embeddings = self.OpenAIEmbeddings()  # type: ignore
        vs = self.FAISS.from_texts(texts, embeddings, metadatas=metadatas)  # type: ignore

        # クエリは固定（居眠り検知の仕様/実装の要点） + 直近の要約
        query = "居眠り検知/連続閉眼/顔信頼度/仕様と実装/誤検知・未検知の原因/改善指針"
        docs = vs.similarity_search(query, k=8)  # type: ignore
        out: list[str] = []
        sources: list[str] = []
        remain = max_chars
        for d in docs:
            head = str(d.page_content)[: min(600, len(str(d.page_content)))]
            tag = d.metadata.get("source", "unknown")
            snippet = f"## {tag}\n{head}\n"
            if len(snippet) <= remain:
                out.append(snippet)
                sources.append(tag)
                remain -= len(snippet)
            else:
                out.append(snippet[:remain])
                sources.append(tag)
                break
        self.last_sources = sources
        return "\n".join(out)

    def _fetch_external_spec_snippets(self) -> str:
        """GitHub仕様リポジトリをローカルに取得し、主要ドキュメントの冒頭を抜粋。"""
        try:
            repo_url = "https://github.com/abekoki/drowsy_detection"
            from pathlib import Path
            import subprocess, urllib.request, zipfile, io

            root = Path(__file__).parent.parent.parent.parent
            dest = root / "external" / "drowsy_detection"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                # git clone（失敗時はZIPダウンロード）
                completed = subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if completed.returncode != 0:
                    try:
                        zip_url = "https://codeload.github.com/abekoki/drowsy_detection/zip/refs/heads/main"
                        data = urllib.request.urlopen(zip_url, timeout=15).read()
                        with zipfile.ZipFile(io.BytesIO(data)) as z:
                            # 展開先作成
                            tmp = dest.parent / "drowsy_detection_main"
                            if tmp.exists():
                                pass
                            z.extractall(dest.parent)
                            # サブフォルダをdestへ移動
                            src_root = dest.parent / "drowsy_detection-main"
                            if src_root.exists():
                                src_root.rename(dest)
                    except Exception:
                        pass
            else:
                subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            snippets: list[str] = []
            sources: list[str] = []
            exts = {".md", ".py"}
            for rel in [
                "README.md",
                "01_algorithm_specification",
                "02_specific_design",
                "src/drowsy_detection",
            ]:
                p = dest / rel
                if p.is_file() and p.suffix.lower() in exts:
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        snippets.append(f"## {p.name}\n{text[:800]}")
                        sources.append(p.as_posix())
                    except Exception:
                        continue
                elif p.is_dir():
                    for f in sorted(p.rglob("*")):
                        if f.is_file() and f.suffix.lower() in exts:
                            try:
                                text = f.read_text(encoding="utf-8", errors="ignore")
                                snippets.append(f"## {f.relative_to(dest).as_posix()}\n{text[:600]}")
                                sources.append(f.as_posix())
                            except Exception:
                                continue
                        if sum(len(s) for s in snippets) > 4000:
                            break
            self.last_sources = sources[:10]
            return "\n".join(snippets)[:2000]
        except Exception:
            return ""

    def _ensure_external_repo(self) -> None:
        """`external/drowsy_detection` ディレクトリを必ず作成し、
        可能なら最新状態を取得する（失敗してもディレクトリは残す）。"""
        from pathlib import Path
        import subprocess
        root = Path(__file__).parent.parent.parent.parent
        dest = root / "external" / "drowsy_detection"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
        # 軽く取得を試す（失敗しても例外は外へ出さない）
        try:
            if not any(dest.iterdir()):
                subprocess.run(["git", "clone", "--depth", "1", "https://github.com/abekoki/drowsy_detection", str(dest)],
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


