"""
Supervisor Agent - Controls the overall analysis workflow
Generic workflow controller supporting multiple algorithms
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import config
from ..models.state import AnalysisState, DatasetInfo
from ..utils.logger import get_logger
from ..utils.exploration_utils import extract_json_with_llm

logger = get_logger(__name__)


class SupervisorAgent:
    """
    Supervisor agent that controls the analysis workflow
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

        self.prompt = ChatPromptTemplate.from_template("""
あなたは汎用時系列データ分析システムの監督者です。解析手順詳細に基づいて、アルゴリズム仕様に応じた最適な分析ワークフローを制御します。

【解析手順詳細に基づくワークフロー】
1. **分析の準備**: 評価指標の明確化、CSV構造確認、対象区間特定
2. **未検知データの抽出**: 評価区間内データフィルタリング、未検知ケース分類
3. **未検知データの特性分析**: 時系列可視化、特徴量分析、コンテキスト確認
4. **アルゴリズムの挙動分析**: 検知ロジック検証、時間窓影響、特徴量有効性
5. **未検知の原因特定**: データ関連、アルゴリズム関連、コンテキスト関連の原因分析
6. **改善案の提案**: データ改善、アルゴリズム改良、コンテキスト考慮
7. **原因レポートの作成**: 構造化レポート生成

【アルゴリズム仕様】
- 名称: {algorithm_name}
- 入力特徴量: {input_columns}
- 出力特徴量: {output_columns}
- 閾値設定: {thresholds}
- 前提条件: 検知結果が有効となるための品質・信頼度条件

現在の状態:
{state_summary}

利用可能なアクション:
- data_checker: データ構造確認と基本分析（ステップ1）
- consistency_checker: 仕様準拠確認とデータ整合性チェック
- hypothesis_generator: 未検知原因の仮説生成（ステップ5）
- verifier: 仮説検証とアルゴリズム挙動分析（ステップ4）
- reporter: 原因レポート作成と改善案提示（ステップ7）

【判断基準】
- データ未確認の場合 → data_checker
- 基本分析完了で不整合がある場合 → consistency_checker
- データ整合性が確認できた場合 → hypothesis_generator
- 仮説生成済みの場合 → verifier
- 検証完了の場合 → reporter

次のステップを決定し、アルゴリズム仕様と解析手順に基づいて理由を説明してください。

応答フォーマット:
```json
{{
    "next_action": "アクション名",
    "reason": "アルゴリズム仕様と解析手順に基づく決定理由",
    "dataset_id": "対象データセットID",
    "analysis_phase": "現在の解析フェーズ（1-7）"
}}
```
""")

    def decide_next_action(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Decide the next action based on current state and algorithm configuration

        Args:
            state: Current analysis state

        Returns:
            Dictionary with next action decision
        """
        try:
            # Prepare state summary
            state_summary = self._create_state_summary(state)

            # Get algorithm configuration for context
            algorithm_context = self._get_algorithm_context(state)

            # Get LLM decision
            chain = self.prompt | self.llm

            response = chain.invoke({
                "state_summary": state_summary,
                **algorithm_context
            })

            # Parse response
            decision = self._parse_llm_response(response.content)

            logger.info(f"Supervisor decided: {decision}")
            return decision

        except Exception as e:
            logger.error(f"Supervisor decision failed: {e}")
            # Fallback decision
            return {
                "next_action": "data_checker",
                "reason": "Error in decision making, defaulting to data_checker",
                "dataset_id": state.get_current_dataset().id if state.get_current_dataset() else None,
                "analysis_phase": "1"
            }

    def _get_algorithm_context(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Get algorithm configuration context for the current dataset

        Args:
            state: Current analysis state

        Returns:
            Dictionary with algorithm context information
        """
        context = {
            "algorithm_name": "Unknown Algorithm",
            "input_columns": [],
            "output_columns": [],
            "thresholds": {}
        }

        try:
            current_dataset = state.get_current_dataset()
            if current_dataset and current_dataset.algorithm_spec_md:
                # Try to load algorithm configuration from spec file
                try:
                    algo_config = config.load_algorithm_config_from_file(current_dataset.algorithm_spec_md)
                    context.update({
                        "algorithm_name": algo_config.name,
                        "input_columns": algo_config.input_columns,
                        "output_columns": algo_config.output_columns,
                        "thresholds": algo_config.thresholds
                    })
                except Exception as e:
                    logger.warning(f"Failed to load algorithm config: {e}")

        except Exception as e:
            logger.warning(f"Failed to get algorithm context: {e}")

        return context

    def _create_state_summary(self, state: AnalysisState) -> str:
        """Create a summary of the current state for the LLM"""
        summary_lines = []

        summary_lines.append(f"ワークフローステップ: {state.workflow_step}")
        summary_lines.append(f"データセット数: {len(state.datasets)}")
        summary_lines.append(f"現在のデータセットインデックス: {state.current_dataset_index}")

        # Current dataset info
        current_dataset = state.get_current_dataset()
        if current_dataset:
            summary_lines.append(f"現在のデータセット: {current_dataset.id}")
            summary_lines.append(f"ステータス: {current_dataset.status}")
            summary_lines.append(f"期待値: {current_dataset.expected_result}")

            if current_dataset.error_message:
                summary_lines.append(f"エラー: {current_dataset.error_message}")

        # Processing summary
        processing = state.get_processing_summary()
        summary_lines.append(f"完了: {processing['completed']}/{processing['total_datasets']}")
        summary_lines.append(f"失敗: {processing['failed']}")

        # Errors
        if state.errors:
            summary_lines.append(f"システムエラー数: {len(state.errors)}")

        return "\n".join(summary_lines)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract decision"""
        try:
            # Use LLM-based JSON extraction only
            import json
            decision = extract_json_with_llm(response)
            if decision is None:
                # Return fallback decision
                decision = {
                    "next_action": "data_checker",
                    "reason": "No valid decision found in LLM response",
                    "dataset_id": None
                }

            # Validate required fields
            required_fields = ["next_action", "reason"]
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"Missing required field: {field}")

            return decision

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Return fallback
            return {
                "next_action": "data_checker",
                "reason": f"Failed to parse response: {e}",
                "dataset_id": None
            }
