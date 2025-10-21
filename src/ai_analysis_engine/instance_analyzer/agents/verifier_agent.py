"""
Verifier Agent - Verifies hypotheses through testing
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import config
from ..models.state import DatasetInfo
from ..models.types import Hypothesis, VerificationResult
from ..tools.repl_tool import REPLTool
from ..tools.rag_tool import RAGTool
from ..utils.logger import get_logger
from ..utils.context_recorder import AgentInteractionLogger
from .reporting_mixins import PromptLoggingMixin

logger = get_logger(__name__)


class VerifierAgent(PromptLoggingMixin):
    """
    Agent for verifying hypotheses through testing and validation
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

        self.repl_tool = REPLTool()
        self.rag_tool = RAGTool()
        self.max_iterations = config.langgraph.max_iterations
        self.prompter = AgentInteractionLogger("verifier_agent")

        self.prompt = ChatPromptTemplate.from_template("""
あなたは汎用検証エージェントです。解析手順詳細.mdの「4. アルゴリズムの挙動分析」に基づいて、アルゴリズム仕様を理解し、仮説をテストして検証します。

【解析手順詳細に基づく検証アプローチ】
1. **検知ロジックの検証**: アルゴリズム仕様に基づき、検知条件を確認
2. **時間窓の影響分析**: 時間窓サイズが評価指標のパターンに合っているか確認
3. **特徴量の有効性評価**: 特徴量が評価指標のパターンを捉えるのに不十分でないか
4. **モデル内部挙動分析**: 出力スコアや確率値を確認し、未検知の理由を分析

【アルゴリズム仕様に基づく検証項目】
- **検知条件確認**: {thresholds}と実際のデータ分布の比較
- **入力特徴量検証**: {input_columns}の妥当性チェック
- **出力形式確認**: {output_columns}の仕様準拠
- **値範囲検証**: {value_ranges}の遵守状況
- **前提条件検証**: 検知結果が有効となるための信頼度・品質条件の確認

データセット情報:
{dataset_info}

アルゴリズム仕様:
{algorithm_spec}

評価環境仕様:
{evaluation_spec}

仮説:
{hypothesis_description}

仮説タイプ: {hypothesis_type}
信頼度: {confidence_score}
仕様関連: {spec_reference}

【検証時の注目点】
- アルゴリズムの判定閾値（{thresholds}）と実際のデータ分布の整合性
- 各種指標の閾値とエラーの関係
- アルゴリズムの詳細な判定ロジック
- 計算内容と閾値の妥当性
- エラーコードと仕様書のエラーハンドリングの一致
- 入力データの有効範囲（{value_ranges}）と実際の値の比較
- 検知結果が有効となる前提条件（信頼度・品質条件）の充足状況

Pythonコードを使って仕様に基づいた動的検証を行い、結果を解析手順詳細の次のステップ（未検知の原因特定）につなげる情報を提供してください。

検証結果を詳細に報告し、アルゴリズム仕様との関連を明確にしてください。
""")

    def verify_hypothesis(self, dataset: DatasetInfo, hypothesis: Hypothesis) -> VerificationResult:
        """
        Verify a hypothesis through testing using dynamic algorithm configuration

        Args:
            dataset: Dataset information
            hypothesis: Hypothesis to verify

        Returns:
            Verification result
        """
        try:
            logger.info(f"Verifying hypothesis {hypothesis.id} for dataset {dataset.id}")

            # Load algorithm configuration dynamically
            algorithm_config = self._load_algorithm_config(dataset)

            # Prepare verification context
            dataset_info = self._prepare_dataset_info(dataset)

            # Prepare algorithm-specific context
            algorithm_context = self._prepare_algorithm_context(algorithm_config)

            # Use LLM to generate verification plan with algorithm context
            verification_plan = self._generate_verification_plan(
                hypothesis, dataset, dataset_info, algorithm_context
            )

            # Generate dynamic verification code/tests based on plan
            verification_code = self._generate_dynamic_verification_code(
                hypothesis, dataset, algorithm_config, verification_plan
            )

            # Execute verification
            verification_result = self._execute_verification(verification_code, dataset)

            # Evaluate results with algorithm context
            evaluation = self._evaluate_verification_results(
                hypothesis, verification_result, dataset, algorithm_config
            )

            result = VerificationResult(
                hypothesis_id=hypothesis.id,
                success=evaluation["success"],
                result_details=evaluation["details"],
                code_executed=verification_code,
                output_data=verification_result.get("result"),
                error_message=verification_result.get("error")
            )

            logger.info(f"Verification completed for hypothesis {hypothesis.id}: {result.success}")
            return result

        except Exception as e:
            logger.error(f"Verification failed for hypothesis {hypothesis.id}: {e}")
            return VerificationResult(
                hypothesis_id=hypothesis.id,
                success=False,
                result_details=f"Verification failed: {e}",
                error_message=str(e)
            )

    def _prepare_dataset_info(self, dataset: DatasetInfo) -> str:
        """Prepare dataset information for verification"""
        info_lines = [
            f"データセットID: {dataset.id}",
            f"アルゴリズム出力: {dataset.algorithm_output_csv}",
            f"コア出力: {dataset.core_output_csv}",
            f"期待値: {dataset.expected_result}"
        ]

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
            "valid_values": algorithm_config.valid_values,
            "detection_patterns": algorithm_config.detection_patterns
        }

    def _generate_verification_plan(self, hypothesis: Hypothesis, dataset: DatasetInfo,
                                   dataset_info: str, algorithm_context: Dict[str, Any]) -> str:
        """
        Generate verification plan using LLM with algorithm context

        Args:
            hypothesis: Hypothesis to verify
            dataset: Dataset information
            dataset_info: Dataset information string
            algorithm_context: Algorithm context dictionary

        Returns:
            String containing verification plan
        """
        plan_prompt = f"""
以下の仮説を検証するための計画を作成してください。

仮説: {hypothesis.description}
仮説タイプ: {hypothesis.type.value}

アルゴリズム仕様コンテキスト:
- 入力特徴量: {algorithm_context.get('input_columns', [])}
- 出力特徴量: {algorithm_context.get('output_columns', [])}
- 閾値設定: {algorithm_context.get('thresholds', {})}
- 値範囲: {algorithm_context.get('value_ranges', {})}
- 有効値: {algorithm_context.get('valid_values', {})}

データセット: {dataset_info}

検証計画の構造:
1. 検証対象の特定（どのデータ/特徴量を検証するか）
2. 検証方法の決定（どのような分析/テストを行うか）
3. 成功/失敗の判定基準
4. 必要なPythonコードの概要

簡潔に計画を記述してください。
"""

        try:
            response = self.llm.invoke(plan_prompt)
            self._log_prompt(
                node="verifier_plan",
                dataset_id=getattr(dataset, "id", None),
                response=response,
                prompt_context={
                    "hypothesis_id": hypothesis.id,
                    "dataset_info": dataset_info,
                },
            )
            self._log_response(
                node="verifier_plan",
                dataset_id=getattr(dataset, "id", None),
                response=response,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate verification plan: {e}")
            return f"Basic verification of hypothesis: {hypothesis.description}"

    def _generate_dynamic_verification_code(self, hypothesis: Hypothesis, dataset: DatasetInfo,
                                           algorithm_config, verification_plan: str) -> str:
        """
        Generate dynamic Python code to verify the hypothesis based on algorithm specifications

        Args:
            hypothesis: Hypothesis to verify
            dataset: Dataset information
            algorithm_config: AlgorithmConfig object
            verification_plan: Verification plan from LLM

        Returns:
            String containing Python verification code
        """
        # Load data
        code_lines = [
            "# Dynamic verification code generated based on algorithm specifications",
            f"import pandas as pd",
            f"import numpy as np",
            f"from typing import Dict, Any",
            f"",
            f"algo_df = pd.read_csv('{dataset.algorithm_output_csv}')",
            f"core_df = pd.read_csv('{dataset.core_output_csv}')",
            f"",
            "# Algorithm configuration loaded from specification",
        ]

        # Add dynamic thresholds from algorithm config
        for key, value in algorithm_config.thresholds.items():
            code_lines.append(f"{key} = {value}")

        code_lines.append("")

        # Add hypothesis-specific verification based on type
        if hypothesis.type.value == "data_quality_issue":
            code_lines.extend(self._generate_data_quality_verification(algorithm_config))
        elif hypothesis.type.value == "specification_inconsistency":
            code_lines.extend(self._generate_specification_verification(algorithm_config))
        elif hypothesis.type.value == "algorithm_bug":
            code_lines.extend(self._generate_algorithm_bug_verification(algorithm_config))
        elif hypothesis.type.value == "parameter_inappropriate":
            code_lines.extend(self._generate_parameter_verification(algorithm_config))
        else:
            code_lines.extend(self._generate_generic_verification(algorithm_config))

        # Add result formatting
        code_lines.extend([
            "",
            "# Format results for analysis",
            "verification_results = {",
            "    'hypothesis_type': '" + hypothesis.type.value + "',",
            "    'verification_plan': '''" + verification_plan.replace("'", "\\'") + "''',",
            "    'data_quality_checks': data_quality_checks,",
            "    'algorithm_validation': algorithm_validation,",
            "    'threshold_analysis': threshold_analysis,",
            "    'pattern_analysis': pattern_analysis",
            "}",
            "",
            "print('Verification completed')",
            "print(f'Hypothesis: {hypothesis.description}')",
            "print(f'Results: {verification_results}')"
        ])

        return "\n".join(code_lines)

    def _generate_data_quality_verification(self, algorithm_config) -> List[str]:
        """Generate data quality verification code"""
        code_lines = [
            "# Data quality verification",
            "data_quality_checks = {}",
            "",
            "# Check missing values",
            "data_quality_checks['missing_values'] = {",
            "    'algorithm_output': algo_df.isnull().sum().to_dict(),",
            "    'core_output': core_df.isnull().sum().to_dict()",
            "}",
            "",
            "# Check data ranges and quality conditions"
        ]

        # Add range checks for all configured columns
        for col, ranges in algorithm_config.value_ranges.items():
            min_val = ranges.get('min', float('-inf'))
            max_val = ranges.get('max', float('inf'))
            code_lines.extend([
                f"# Check {col} range",
                f"if '{col}' in algo_df.columns or '{col}' in core_df.columns:",
                f"    df_to_check = algo_df if '{col}' in algo_df.columns else core_df",
                f"    out_of_range = ((df_to_check['{col}'] < {min_val}) | (df_to_check['{col}'] > {max_val})).sum()",
                f"    data_quality_checks['{col}_range'] = {{'out_of_range': int(out_of_range), 'total': len(df_to_check)}}"
            ])

        # Add quality condition checks (e.g., confidence thresholds)
        for threshold_name, threshold_value in algorithm_config.thresholds.items():
            if 'confidence' in threshold_name.lower() or 'quality' in threshold_name.lower():
                col_name = threshold_name.lower().replace('_threshold', '').replace('_quality', '_confidence')
                if col_name in algorithm_config.input_columns:
                    code_lines.extend([
                        f"# Check {threshold_name} condition",
                        f"if '{col_name}' in algo_df.columns or '{col_name}' in core_df.columns:",
                        f"    df_to_check = algo_df if '{col_name}' in algo_df.columns else core_df",
                        f"    below_threshold = (df_to_check['{col_name}'] < {threshold_value}).sum()",
                        f"    data_quality_checks['{threshold_name}_compliance'] = {{'below_threshold': int(below_threshold), 'threshold': {threshold_value}, 'total': len(df_to_check)}}"
                    ])

        return code_lines

    def _generate_specification_verification(self, algorithm_config) -> List[str]:
        """Generate specification compliance verification code"""
        code_lines = [
            "# Specification compliance verification",
            "spec_compliance = {}",
            "",
            "# Check required columns"
        ]

        # Check required columns
        for col in algorithm_config.required_columns:
            code_lines.extend([
                f"spec_compliance['{col}_exists'] = '{col}' in algo_df.columns or '{col}' in core_df.columns"
            ])

        # Check valid values
        for col, valid_vals in algorithm_config.valid_values.items():
            if col in ['is_drowsy']:  # Example output column
                vals_str = str(valid_vals)
                code_lines.extend([
                    f"if '{col}' in algo_df.columns:",
                    f"    invalid_count = (~algo_df['{col}'].isin({vals_str})).sum()",
                    f"    spec_compliance['{col}_valid'] = {{'invalid_count': int(invalid_count), 'total': len(algo_df)}}"
                ])

        return code_lines

    def _generate_algorithm_bug_verification(self, algorithm_config) -> List[str]:
        """Generate algorithm bug detection verification code"""
        code_lines = [
            "# Algorithm bug detection verification",
            "algorithm_validation = {}",
            "",
            "# Check threshold compliance"
        ]

        # Add threshold checks
        for threshold_name, threshold_value in algorithm_config.thresholds.items():
            if 'EYE' in threshold_name.upper():  # Eye-related thresholds
                col_name = threshold_name.lower().replace('_threshold', '').replace('left_', 'l').replace('right_', 'r') + '_openness'
                code_lines.extend([
                    f"if '{col_name}' in core_df.columns:",
                    f"    below_threshold = (core_df['{col_name}'] < {threshold_value}).sum()",
                    f"    algorithm_validation['{threshold_name}_compliance'] = {{'below_threshold': int(below_threshold), 'threshold': {threshold_value}}}"
                ])

        return code_lines

    def _generate_parameter_verification(self, algorithm_config) -> List[str]:
        """Generate parameter appropriateness verification code"""
        code_lines = [
            "# Parameter appropriateness verification",
            "parameter_analysis = {}",
            "",
            "# Analyze threshold effectiveness"
        ]

        # Add parameter analysis for all input columns
        for col in algorithm_config.input_columns:
            code_lines.extend([
                f"if '{col}' in core_df.columns:",
                f"    param_stats = core_df['{col}'].describe()",
                f"    parameter_analysis['{col}_distribution'] = param_stats.to_dict()"
                ])

        return code_lines

    def _generate_generic_verification(self, algorithm_config) -> List[str]:
        """Generate generic verification code"""
        return [
            "# Generic verification",
            "generic_checks = {}",
            "",
            "# Basic statistical analysis",
            "generic_checks['basic_stats'] = {",
            "    'algorithm_shape': algo_df.shape,",
            "    'core_shape': core_df.shape,",
            "    'algorithm_columns': list(algo_df.columns),",
            "    'core_columns': list(core_df.columns)",
            "}"
        ]


    def _execute_verification(self, code: str, dataset: DatasetInfo) -> Dict[str, Any]:
        """Execute verification code"""
        try:
            # Create execution context with data paths
            context = {
                "dataset": dataset,
                "algo_csv": dataset.algorithm_output_csv,
                "core_csv": dataset.core_output_csv
            }

            # Execute the code
            result = self.repl_tool.execute_code(code, context)

            return result

        except Exception as e:
            logger.error(f"Verification execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _evaluate_verification_results(self, hypothesis: Hypothesis,
                                     verification_result: Dict[str, Any],
                                     dataset: DatasetInfo, algorithm_config=None) -> Dict[str, Any]:
        """
        Evaluate the results of verification with algorithm context

        Args:
            hypothesis: Hypothesis that was verified
            verification_result: Results from verification execution
            dataset: Dataset information
            algorithm_config: AlgorithmConfig for context-aware evaluation

        Returns:
            Dictionary with evaluation results
        """
        try:
            if not verification_result.get("success"):
                return {
                    "success": False,
                    "details": f"Verification failed: {verification_result.get('error')}"
                }

            # Evaluate based on hypothesis type and algorithm context
            stdout = verification_result.get("stdout", "")
            result_data = verification_result.get("result", {})

            # Algorithm-aware evaluation
            evaluation_details = []

            if algorithm_config:
                # Check algorithm specification compliance
                if hypothesis.type.value == "data_quality_issue":
                    evaluation_details.extend(self._evaluate_data_quality_with_spec(
                        stdout, result_data, algorithm_config
                    ))

                elif hypothesis.type.value == "specification_inconsistency":
                    evaluation_details.extend(self._evaluate_specification_compliance(
                        stdout, result_data, algorithm_config
                    ))

                elif hypothesis.type.value == "algorithm_bug":
                    evaluation_details.extend(self._evaluate_algorithm_behavior(
                        stdout, result_data, algorithm_config
                    ))

                elif hypothesis.type.value == "parameter_inappropriate":
                    evaluation_details.extend(self._evaluate_parameter_effectiveness(
                        stdout, result_data, algorithm_config
                    ))

            # General evaluation
            if stdout and len(stdout.strip()) > 0:
                evaluation_details.append(f"Verification output: {stdout[:200]}...")

            success = len(evaluation_details) > 0
            details = "; ".join(evaluation_details) if evaluation_details else "No specific findings"

            return {
                "success": success,
                "details": details,
                "evaluation_points": evaluation_details
            }

        except Exception as e:
            logger.error(f"Result evaluation failed: {e}")
            return {
                "success": False,
                "details": f"Evaluation failed: {e}"
            }

    def _evaluate_data_quality_with_spec(self, stdout: str, result_data: Dict,
                                       algorithm_config) -> List[str]:
        """Evaluate data quality issues with algorithm specification context"""
        findings = []

        # Check for missing values
        if "Missing values" in stdout:
            if "algorithm" in stdout.lower() and "0" not in stdout:
                findings.append("Algorithm output contains missing values")

        # Check required columns
        for col in algorithm_config.required_columns:
            if f"missing_{col}" in stdout.lower():
                findings.append(f"Required column '{col}' is missing")

        # Check value ranges
        for col, ranges in algorithm_config.value_ranges.items():
            if f"{col}_range" in stdout.lower():
                findings.append(f"Column '{col}' has values outside specified range")

        return findings

    def _evaluate_specification_compliance(self, stdout: str, result_data: Dict,
                                         algorithm_config) -> List[str]:
        """Evaluate specification compliance"""
        findings = []

        # Check column alignment
        if "missing" in stdout.lower():
            findings.append("Missing required columns detected")

        # Check valid values
        for col, valid_vals in algorithm_config.valid_values.items():
            if f"{col}_valid" in stdout.lower():
                findings.append(f"Column '{col}' contains invalid values")

        return findings

    def _evaluate_algorithm_behavior(self, stdout: str, result_data: Dict,
                                   algorithm_config) -> List[str]:
        """Evaluate algorithm behavior against specifications"""
        findings = []

        # Check threshold compliance
        for threshold_name in algorithm_config.thresholds.keys():
            if threshold_name.lower() in stdout.lower():
                findings.append(f"Threshold '{threshold_name}' compliance analyzed")

        # Check for unrealistic values
        if "unrealistic" in stdout.lower():
            findings.append("Unrealistic values detected in algorithm output")

        return findings

    def _evaluate_parameter_effectiveness(self, stdout: str, result_data: Dict,
                                        algorithm_config) -> List[str]:
        """Evaluate parameter effectiveness"""
        findings = []

        # Check parameter distributions
        for col in algorithm_config.input_columns:
            if f"{col}_distribution" in stdout.lower():
                findings.append(f"Parameter '{col}' distribution analyzed")

        return findings
