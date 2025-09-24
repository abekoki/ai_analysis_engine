"""
Hypothesis Generator Agent - Generates hypotheses about potential issues
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import config
from ..models.state import DatasetInfo
from ..models.types import Hypothesis, HypothesisType
from ..tools.rag_tool import RAGTool
from ..utils.logger import get_logger
from ..utils.exploration_utils import extract_json_with_llm

logger = get_logger(__name__)


class HypothesisGeneratorAgent:
    """
    Agent for generating hypotheses about potential issues in the data
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

        self.rag_tool = RAGTool()

        self.prompt = ChatPromptTemplate.from_template("""
あなたは汎用仮説生成エージェントです。解析手順詳細.mdの「5. 未検知の原因特定」に基づいて、アルゴリズム仕様を理解し、データ分析結果から未検知の根本原因を特定します。

【解析手順詳細に基づく原因特定アプローチ】
1. **データ関連原因分析**: センサー値の特性、ノイズの影響、データ品質
2. **アルゴリズム関連原因分析**: 閾値設定、時間窓不適切さ、特徴量不足
3. **コンテキスト関連原因分析**: 被験者個人差、環境要因
4. **総合的因果関係特定**: 複数要因の相互作用の評価

【アルゴリズム仕様に基づく仮説生成項目】
- **入力パラメータ確認**: {input_columns}の妥当性と分布
- **出力形式検証**: {output_columns}の仕様準拠
- **閾値設定検証**: {thresholds}とデータ分布の整合性
- **値範囲検証**: {value_ranges}の遵守状況
- **前提条件分析**: 検知結果が有効となるための信頼度・品質条件の影響

データセット情報:
{dataset_info}

アルゴリズム仕様:
{algorithm_spec}

評価環境仕様:
{evaluation_spec}

データ分析結果:
{data_analysis}

整合性チェック結果:
{consistency_check}

期待値:
{expected_result}

【原因カテゴリ別分析（解析手順詳細準拠）】

**データ関連原因**:
- センサー値の特性（微小変化、緩やかな変化）
- ノイズの影響（センサーノイズ、外乱）
- データ品質問題（欠損値、キャリブレーションエラー）
- サンプリング間隔の問題

**アルゴリズム関連原因**:
- 閾値設定の不適切さ（厳しすぎる/緩すぎる）
- 時間窓サイズの問題（短すぎる/長すぎる）
- 特徴量の不足（評価指標を捉えきれていない）
- モデル限界（学習データ不足、複雑パターン未学習）

**コンテキスト関連原因**:
- 被験者個人差（ベースラインの違い）
- 環境要因（センサー装着ズレ、外部ノイズ）
- 評価条件の不一致

【仮説生成の優先順位】
1. **高確度仮説**: データ分析結果と仕様書の明確な矛盾
2. **中確度仮説**: 仕様の曖昧さや境界条件の問題
3. **低確度仮説**: 推測ベースの潜在的問題

各仮説について：
- 解析手順詳細の該当ステップとの関連を明確に
- 具体的な証拠とアルゴリズム仕様書の該当箇所
- 原因カテゴリ（データ/アルゴリズム/コンテキスト）の分類
- 仕様に基づいた修正提案と予想効果

仮説をJSON形式で生成してください。

応答フォーマット:
```json
{{
    "hypotheses": [
        {{
            "id": "unique_id",
            "type": "hypothesis_type",
            "category": "data|algorithm|context",
            "description": "解析手順詳細に基づく詳細な原因説明",
            "confidence_score": 0.8,
            "evidence": ["データ分析結果の該当箇所", "仕様書の関連部分"],
            "spec_reference": "アルゴリズム仕様書の関連部分",
            "analysis_step": "解析手順詳細の該当ステップ番号",
            "suggested_fix": "仕様に基づいた具体的な修正提案",
            "expected_impact": "修正による予想効果"
        }}
    ]
}}
```
""")

    def generate_hypotheses(self, dataset: DatasetInfo,
                          data_analysis: Dict[str, Any],
                          consistency_check: Dict[str, Any]) -> List[Hypothesis]:
        """
        Generate hypotheses based on analysis results and algorithm configuration

        Args:
            dataset: Dataset information
            data_analysis: Results from data analysis
            consistency_check: Results from consistency checking

        Returns:
            List of generated hypotheses
        """
        try:
            logger.info(f"Generating hypotheses for dataset {dataset.id}")

            # Load algorithm configuration dynamically
            algorithm_config = self._load_algorithm_config(dataset)

            # Load specifications
            algorithm_spec = ""
            if dataset.algorithm_spec_md:
                try:
                    with open(dataset.algorithm_spec_md, 'r', encoding='utf-8') as f:
                        algorithm_spec = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load algorithm spec: {e}")

            evaluation_spec = ""
            if dataset.evaluation_spec_md:
                try:
                    with open(dataset.evaluation_spec_md, 'r', encoding='utf-8') as f:
                        evaluation_spec = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load evaluation spec: {e}")

            # Prepare input data
            dataset_info = self._prepare_dataset_info(dataset)
            analysis_summary = self._summarize_analysis(data_analysis)
            consistency_summary = self._summarize_consistency(consistency_check)

            # Prepare algorithm-specific context
            algorithm_context = self._prepare_algorithm_context(algorithm_config)

            # Use LLM to generate hypotheses with algorithm context
            chain = self.prompt | self.llm

            response = chain.invoke({
                "dataset_info": dataset_info,
                "algorithm_spec": algorithm_spec[:3000],
                "evaluation_spec": evaluation_spec[:2000],
                "data_analysis": analysis_summary,
                "consistency_check": consistency_summary,
                "expected_result": dataset.expected_result,
                **algorithm_context
            })

            # Parse hypotheses from response
            hypotheses = self._parse_hypotheses(response.content)

            # Add rule-based hypotheses with algorithm context
            rule_based = self._generate_rule_based_hypotheses(
                dataset, data_analysis, consistency_check, algorithm_config
            )
            hypotheses.extend(rule_based)

            # Remove duplicates and sort by confidence
            unique_hypotheses = self._deduplicate_hypotheses(hypotheses)
            unique_hypotheses.sort(key=lambda h: h.confidence_score, reverse=True)

            logger.info(f"Generated {len(unique_hypotheses)} hypotheses for dataset {dataset.id}")
            return unique_hypotheses

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            # Return fallback hypothesis
            return [
                Hypothesis(
                    id="error_fallback",
                    type=HypothesisType.ALGORITHM_BUG,
                    category="algorithm",
                    description="Analysis failed, potential algorithm issue",
                    confidence_score=0.5,
                    evidence=["Analysis error occurred"],
                    analysis_step="5",
                    suggested_fix="Review analysis pipeline",
                    expected_impact="Improved system stability"
                )
            ]

    def _prepare_dataset_info(self, dataset: DatasetInfo) -> str:
        """Prepare dataset information for LLM"""
        info_lines = [
            f"データセットID: {dataset.id}",
            f"期待値: {dataset.expected_result}",
            f"アルゴリズム出力: {dataset.algorithm_output_csv}",
            f"コア出力: {dataset.core_output_csv}"
        ]

        if dataset.error_message:
            info_lines.append(f"エラー: {dataset.error_message}")

        return "\n".join(info_lines)

    def _load_algorithm_config(self, dataset: DatasetInfo):
        """
        Load algorithm configuration from dataset specification

        Args:
            dataset: Dataset information

        Returns:
            AlgorithmConfig: Loaded configuration
        """
        if dataset.algorithm_spec_md:
            try:
                return config.load_algorithm_config_from_file(dataset.algorithm_spec_md)
            except Exception as e:
                logger.warning(f"Failed to load algorithm config from file: {e}")

        # Return default configuration
        from ..config.config import AlgorithmConfig
        return AlgorithmConfig()

    def _prepare_algorithm_context(self, algorithm_config) -> Dict[str, Any]:
        """
        Prepare algorithm-specific context for LLM

        Args:
            algorithm_config: AlgorithmConfig object

        Returns:
            Dictionary with algorithm context
        """
        return {
            "input_columns": algorithm_config.input_columns,
            "output_columns": algorithm_config.output_columns,
            "thresholds": algorithm_config.thresholds,
            "value_ranges": algorithm_config.value_ranges,
            "valid_values": algorithm_config.valid_values
        }

    def _summarize_analysis(self, data_analysis: Dict[str, Any]) -> str:
        """Summarize data analysis results"""
        if not data_analysis:
            return "データ分析結果なし"

        summary_lines = []

        # Basic analysis summary
        basic = data_analysis.get("basic_analysis", {})
        for name, analysis in basic.items():
            if isinstance(analysis, dict) and "shape" in analysis:
                shape = analysis["shape"]
                summary_lines.append(f"{name}: {shape[0]}行 x {shape[1]}列")

                if "missing_values" in analysis:
                    missing = analysis["missing_values"]
                    total_missing = sum(missing.values())
                    if total_missing > 0:
                        summary_lines.append(f"  欠損値: {total_missing}個")

        return "\n".join(summary_lines)

    def _summarize_consistency(self, consistency_check: Dict[str, Any]) -> str:
        """Summarize consistency check results"""
        if not consistency_check:
            return "整合性チェック結果なし"

        summary_lines = []

        if consistency_check.get("overall_consistent"):
            summary_lines.append("全体的に整合性あり")
        else:
            summary_lines.append("整合性の問題あり")

        issues = consistency_check.get("issues", [])
        if issues:
            summary_lines.append("問題点:")
            for issue in issues[:3]:  # Limit to top 3
                summary_lines.append(f"  - {issue}")

        return "\n".join(summary_lines)

    def _parse_hypotheses(self, response: str) -> List[Hypothesis]:
        """Parse hypotheses from LLM response with new fields"""
        hypotheses = []

        try:
            import json

            # Use LLM-based JSON extraction only
            data = extract_json_with_llm(response)
            if data is None:
                return hypotheses

                for hypo_data in data.get("hypotheses", []):
                    try:
                        hypothesis = Hypothesis(
                            id=hypo_data["id"],
                            type=HypothesisType(hypo_data["type"]),
                            category=hypo_data.get("category", "unknown"),
                            description=hypo_data["description"],
                            confidence_score=float(hypo_data["confidence_score"]),
                            evidence=hypo_data.get("evidence", []),
                            spec_reference=hypo_data.get("spec_reference"),
                            analysis_step=hypo_data.get("analysis_step"),
                            suggested_fix=hypo_data.get("suggested_fix"),
                            expected_impact=hypo_data.get("expected_impact")
                        )
                        hypotheses.append(hypothesis)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Invalid hypothesis data: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Failed to parse hypotheses: {e}")

        return hypotheses

    def _generate_rule_based_hypotheses(self, dataset: DatasetInfo,
                                      data_analysis: Dict[str, Any],
                                      consistency_check: Dict[str, Any],
                                      algorithm_config=None) -> List[Hypothesis]:
        """Generate rule-based hypotheses with algorithm context"""
        hypotheses = []

        # Rule 1: High missing values (Data-related)
        if data_analysis.get("basic_analysis"):
            for name, analysis in data_analysis["basic_analysis"].items():
                if isinstance(analysis, dict) and "missing_values" in analysis:
                    missing = analysis["missing_values"]
                    total_missing = sum(missing.values())
                    if total_missing > 0:
                        hypotheses.append(Hypothesis(
                            id=f"missing_values_{name}",
                            type=HypothesisType.DATA_QUALITY_ISSUE,
                            category="data",
                            description=f"データ品質問題: {name}に欠損値が存在 ({total_missing}個)",
                            confidence_score=0.7,
                            evidence=[f"欠損値: {missing}"],
                            analysis_step="5",
                            spec_reference="データ品質要件",
                            suggested_fix="欠損値処理の実装またはデータ収集の見直し",
                            expected_impact="データ完全性の向上"
                        ))

        # Rule 2: Consistency issues (Specification-related)
        if not consistency_check.get("overall_consistent", True):
            issues = consistency_check.get("issues", [])
            if issues:
                hypotheses.append(Hypothesis(
                    id="consistency_issue",
                    type=HypothesisType.SPECIFICATION_INCONSISTENCY,
                    category="algorithm",
                    description=f"仕様不整合: {issues[0]}",
                    confidence_score=0.8,
                    evidence=issues[:3],
                    analysis_step="5",
                    spec_reference="アルゴリズム仕様書",
                    suggested_fix="仕様書の確認と実装の修正",
                    expected_impact="仕様準拠性の向上"
                ))

        # Rule 3: Empty dataframes (Data-related)
        if data_analysis.get("basic_analysis"):
            for name, analysis in data_analysis["basic_analysis"].items():
                if isinstance(analysis, dict) and analysis.get("shape", [0, 0])[0] == 0:
                    hypotheses.append(Hypothesis(
                        id=f"empty_data_{name}",
                        type=HypothesisType.DATA_QUALITY_ISSUE,
                        category="data",
                        description=f"データ品質問題: {name}が空のデータ",
                        confidence_score=0.9,
                        evidence=["データフレームが空"],
                        analysis_step="3",
                        spec_reference="データ要件",
                        suggested_fix="データ生成プロセスの確認",
                        expected_impact="データ可用性の確保"
                    ))

        # Rule 4: Algorithm-specific validation (Algorithm-related)
        if algorithm_config:
            # Check threshold compliance
            if data_analysis.get("basic_analysis"):
                for name, analysis in data_analysis["basic_analysis"].items():
                    if isinstance(analysis, dict) and "algorithm_validation" in analysis:
                        validation = analysis["algorithm_validation"]
                        if validation.get("threshold_compliance"):
                            for col, compliance in validation["threshold_compliance"].items():
                                if compliance.get("percentage", 0) > 50:  # More than 50% above threshold
                                    hypotheses.append(Hypothesis(
                                        id=f"threshold_issue_{col}",
                                        type=HypothesisType.PARAMETER_INAPPROPRIATE,
                                        category="algorithm",
                                        description=f"閾値設定問題: {col}の{compliance['percentage']:.1f}%が閾値を超過",
                                        confidence_score=0.6,
                                        evidence=[f"閾値超過率: {compliance['percentage']:.1f}%"],
                                        analysis_step="5",
                                        spec_reference=f"{col}閾値設定",
                                        suggested_fix=f"{col}の閾値を見直し",
                                        expected_impact="検知精度の改善"
                                    ))

        # Rule 5: Quality condition issues (Context-related)
        if algorithm_config:
            # Check quality/confidence conditions
            quality_thresholds = [k for k in algorithm_config.thresholds.keys()
                                if 'confidence' in k.lower() or 'quality' in k.lower()]
            if quality_thresholds and data_analysis.get("basic_analysis"):
                for name, analysis in data_analysis["basic_analysis"].items():
                    if isinstance(analysis, dict) and "algorithm_validation" in analysis:
                        validation = analysis["algorithm_validation"]
                        for threshold_name in quality_thresholds:
                            compliance_key = f"{threshold_name}_compliance"
                            if compliance_key in validation:
                                compliance = validation[compliance_key]
                                if compliance.get("below_threshold", 0) > 10:  # More than 10 frames below quality threshold
                                    hypotheses.append(Hypothesis(
                                        id=f"quality_condition_{threshold_name}",
                                        type=HypothesisType.DATA_QUALITY_ISSUE,
                                        category="context",
                                        description=f"信頼度条件未充足: {threshold_name}が{compliance['below_threshold']}フレームで条件を満たしていない",
                                        confidence_score=0.8,
                                        evidence=[f"信頼度条件未充足フレーム数: {compliance['below_threshold']}"],
                                        analysis_step="5",
                                        spec_reference=f"{threshold_name}品質要件",
                                        suggested_fix="信頼度条件の見直しまたはデータ収集条件の改善",
                                        expected_impact="検知結果の信頼性向上"
                                    ))

            # Check value range compliance
            if data_analysis.get("basic_analysis"):
                for name, analysis in data_analysis["basic_analysis"].items():
                    if isinstance(analysis, dict) and "algorithm_validation" in analysis:
                        validation = analysis["algorithm_validation"]
                        if validation.get("value_range_compliance"):
                            for col, compliance in validation["value_range_compliance"].items():
                                if compliance.get("percentage", 0) > 10:  # More than 10% out of range
                                    hypotheses.append(Hypothesis(
                                        id=f"range_issue_{col}",
                                        type=HypothesisType.DATA_QUALITY_ISSUE,
                                        category="data",
                                        description=f"値範囲問題: {col}の{compliance['percentage']:.1f}%が有効範囲外",
                                        confidence_score=0.7,
                                        evidence=[f"範囲外データ率: {compliance['percentage']:.1f}%"],
                                        analysis_step="5",
                                        spec_reference=f"{col}有効範囲",
                                        suggested_fix=f"{col}のデータ範囲を修正",
                                        expected_impact="データ品質の向上"
                                    ))

        return hypotheses

    def _deduplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Remove duplicate hypotheses"""
        seen = set()
        unique = []

        for hypo in hypotheses:
            # Create a simple signature for deduplication
            signature = (hypo.type.value, hypo.description[:50])

            if signature not in seen:
                seen.add(signature)
                unique.append(hypo)

        return unique
