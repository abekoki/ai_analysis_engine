"""
Consistency Checker Agent - Checks consistency between data and specifications
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import config
from ..models.state import DatasetInfo
from ..tools.rag_tool import RAGTool
from ..tools.repl_tool import REPLTool
from ..utils.logger import get_logger
from ..utils.text_utils import extract_frame_range, extract_expected_value
from ..utils.context_recorder import AgentInteractionLogger
from .reporting_mixins import PromptLoggingMixin

logger = get_logger(__name__)


class ConsistencyCheckerAgent(PromptLoggingMixin):
    """
    Agent for checking consistency between data and specifications
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

        self.rag_tool = RAGTool()
        self.repl_tool = REPLTool()
        self.prompter = AgentInteractionLogger("consistency_checker_agent")

        self.prompt = ChatPromptTemplate.from_template("""
あなたは整合性チェックエージェントです。アルゴリズム仕様に基づいてデータと期待値の整合性を確認してください。

【重要】アルゴリズム仕様を理解し、その仕様に基づいて整合性をチェックしてください。

データセット情報:
{dataset_info}

アルゴリズム仕様:
{algorithm_spec}

評価環境仕様:
{evaluation_spec}

期待値（自然言語）:
{expected_result}

データ分析結果:
{data_analysis}

【整合性チェックアプローチ】
1. **アルゴリズム仕様の理解**: 入力パラメータ、判定ロジック、出力形式を把握
2. **期待値の解釈**: 自然言語の期待値をアルゴリズム仕様に基づいて具体的な条件に変換
3. **データ整合性**: CSVデータの構造と値が仕様に準拠しているか確認
4. **アルゴリズム出力検証**: 出力が仕様通りのロジックで生成されているか確認
5. **エラーパターン分析**: エラーコードと仕様書のエラーハンドリングの整合性

【特に注目すべき点】
- アルゴリズムの判定閾値と実際のデータ分布の整合性
- 入力パラメータの有効範囲と実際の値の比較
- 出力値（is_drowsy, error_code）の妥当性
- 計算ロジックと実際のcsv出力内容の一致
- 各種指標の閾値とエラーの発生パターン

不整合点を具体的に指摘し、仕様に基づいた改善提案を行ってください。

整合性チェック結果を詳細に報告してください。
""")

    def check_consistency(self, dataset: DatasetInfo, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check consistency for the dataset

        Args:
            dataset: Dataset to check
            data_analysis: Results from data analysis

        Returns:
            Consistency check results
        """
        try:
            logger.info(f"Starting consistency check for dataset {dataset.id}")

            # Load algorithm spec
            algorithm_spec = ""
            if dataset.algorithm_spec_md:
                try:
                    with open(dataset.algorithm_spec_md, 'r', encoding='utf-8') as f:
                        algorithm_spec = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load algorithm spec: {e}")

            # Load evaluation spec
            evaluation_spec = ""
            if dataset.evaluation_spec_md:
                try:
                    with open(dataset.evaluation_spec_md, 'r', encoding='utf-8') as f:
                        evaluation_spec = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load evaluation spec: {e}")

            # Prepare dataset info
            dataset_info = self._prepare_dataset_info(dataset)

            # Parse expected result
            parsed_expectation = self._parse_expected_result(dataset.expected_result)

            # Perform consistency checks
            consistency_results = self._perform_consistency_checks(
                dataset, data_analysis, parsed_expectation
            )

            # Use LLM for final assessment
            chain = self.prompt | self.llm

            response = chain.invoke({
                "dataset_info": dataset_info,
                "algorithm_spec": algorithm_spec[:3000],  # Prioritize algorithm spec
                "evaluation_spec": evaluation_spec[:2000],
                "expected_result": dataset.expected_result,
                "data_analysis": str(data_analysis)[:2000]  # Limit size
            })

            self._log_prompt(
                node="consistency_checker",
                dataset_id=getattr(dataset, "id", None),
                response=response,
                prompt_context={
                    "dataset_info": dataset_info,
                    "parsed_expectation": parsed_expectation,
                    "data_analysis_keys": list(data_analysis.keys()) if isinstance(data_analysis, dict) else None,
                },
            )

            self._log_response(
                node="consistency_checker",
                dataset_id=getattr(dataset, "id", None),
                response=response,
            )

            consistency_results["llm_assessment"] = response.content

            self._log_result(
                node="consistency_checker",
                dataset_id=getattr(dataset, "id", None),
                result=consistency_results,
                description="Consistency checker aggregated results",
            )

            logger.info(f"Consistency check completed for dataset {dataset.id}")
            return consistency_results

        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {
                "error": str(e),
                "overall_consistent": False,
                "issues": ["Consistency check failed due to error"]
            }

    def _prepare_dataset_info(self, dataset: DatasetInfo) -> str:
        """Prepare dataset information"""
        info_lines = [
            f"データセットID: {dataset.id}",
            f"期待値: {dataset.expected_result}",
            f"アルゴリズム出力: {dataset.algorithm_output_csv}",
            f"コア出力: {dataset.core_output_csv}"
        ]

        return "\n".join(info_lines)

    def _parse_expected_result(self, expected_result: str) -> Dict[str, Any]:
        """Parse natural language expected result"""
        parsed = {
            "original": expected_result,
            "frame_range": extract_frame_range(expected_result),
            "expected_value": extract_expected_value(expected_result)
        }

        return parsed

    def _perform_consistency_checks(self, dataset: DatasetInfo,
                                  data_analysis: Dict[str, Any],
                                  parsed_expectation: Dict[str, Any]) -> Dict[str, Any]:
        """Perform various consistency checks"""
        results = {
            "overall_consistent": True,
            "checks_performed": [],
            "issues": [],
            "parsed_expectation": parsed_expectation
        }

        try:
            # Load data for checking
            dataframes = self.repl_tool.load_csv_data([
                dataset.algorithm_output_csv,
                dataset.core_output_csv
            ])

            # Check 1: Frame range consistency
            frame_check = self._check_frame_range(dataframes, parsed_expectation)
            results["checks_performed"].append(frame_check)

            if not frame_check["consistent"]:
                results["issues"].append(frame_check["issue"])
                results["overall_consistent"] = False

            # Check 2: Value pattern consistency
            value_check = self._check_value_patterns(dataframes, parsed_expectation)
            results["checks_performed"].append(value_check)

            if not value_check["consistent"]:
                results["issues"].append(value_check["issue"])
                results["overall_consistent"] = False

            # Check 3: Data completeness
            completeness_check = self._check_data_completeness(dataframes)
            results["checks_performed"].append(completeness_check)

            if not completeness_check["consistent"]:
                results["issues"].append(completeness_check["issue"])
                results["overall_consistent"] = False

        except Exception as e:
            results["issues"].append(f"Consistency check error: {e}")
            results["overall_consistent"] = False

        return results

    def _check_frame_range(self, dataframes: Dict[str, Any],
                          parsed_expectation: Dict[str, Any]) -> Dict[str, Any]:
        """Check frame range consistency"""
        frame_range = parsed_expectation.get("frame_range")

        if not frame_range:
            return {
                "check_type": "frame_range",
                "consistent": True,
                "message": "No specific frame range specified"
            }

        start_frame, end_frame = frame_range

        for name, df in dataframes.items():
            if len(df) == 0:
                continue

            # Check if dataframe has frame/timestamp column
            frame_col = None
            if 'frame' in df.columns:
                frame_col = 'frame'
            elif 'timestamp' in df.columns:
                # Try to infer frame numbers from timestamp
                frame_col = 'timestamp'

            if frame_col:
                min_frame = df[frame_col].min()
                max_frame = df[frame_col].max()

                if start_frame < min_frame or end_frame > max_frame:
                    return {
                        "check_type": "frame_range",
                        "consistent": False,
                        "issue": f"Expected frame range {start_frame}-{end_frame} not covered by data range {min_frame}-{max_frame} in {name}"
                    }

        return {
            "check_type": "frame_range",
            "consistent": True,
            "message": f"Frame range {start_frame}-{end_frame} is covered by data"
        }

    def _check_value_patterns(self, dataframes: Dict[str, Any],
                            parsed_expectation: Dict[str, Any]) -> Dict[str, Any]:
        """Check value patterns consistency"""
        expected_value = parsed_expectation.get("expected_value")

        if not expected_value:
            return {
                "check_type": "value_pattern",
                "consistent": True,
                "message": "No specific value pattern specified"
            }

        # This is a simplified check - in a real system you'd have more sophisticated NLP
        for name, df in dataframes.items():
            if len(df) == 0:
                continue

            # Look for patterns that might indicate issues
            numeric_cols = df.select_dtypes(include=['number']).columns

            for col in numeric_cols:
                # Check for unusual patterns (simplified)
                if df[col].isnull().all():
                    return {
                        "check_type": "value_pattern",
                        "consistent": False,
                        "issue": f"Column {col} is completely null in {name}"
                    }

        return {
            "check_type": "value_pattern",
            "consistent": True,
            "message": "Value patterns appear consistent"
        }

    def _check_data_completeness(self, dataframes: Dict[str, Any]) -> Dict[str, Any]:
        """Check data completeness"""
        for name, df in dataframes.items():
            if len(df) == 0:
                return {
                    "check_type": "data_completeness",
                    "consistent": False,
                    "issue": f"Dataframe {name} is empty"
                }

            # Check for excessive missing values
            missing_pct = df.isnull().mean().mean()
            if missing_pct > 0.5:  # More than 50% missing
                return {
                    "check_type": "data_completeness",
                    "consistent": False,
                    "issue": f"High missing value percentage ({missing_pct:.2%}) in {name}"
                }

        return {
            "check_type": "data_completeness",
            "consistent": True,
            "message": "Data completeness is acceptable"
        }
