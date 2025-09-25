"""
Data Checker Agent - Analyzes and validates input data
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from ..config import config
from ..models.state import AnalysisState, DatasetInfo
from ..tools.rag_tool import RAGTool
from ..tools.repl_tool import REPLTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataCheckerAgent:
    """
    Agent for checking and analyzing input data
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

        self.rag_tool = RAGTool()
        self.repl_tool = REPLTool()

        # Setup tools
        self.tools = [
            self._create_rag_search_tool(),
            self._create_data_analysis_tool(),
            self._create_plot_creation_tool()
        ]

        self.prompt = ChatPromptTemplate.from_template("""
あなたは汎用データ確認エージェントです。解析手順詳細.mdの「1. 分析の準備」に基づいて、アルゴリズム仕様を理解し、データセットを分析します。

【解析手順詳細に基づく分析アプローチ】
1. **評価指標の明確化**: 自然言語で定義された評価指標を具体化
2. **CSVデータの構造確認**: 列の確認とアルゴリズム仕様との対応
3. **対象評価区間の特定**: 解析対象区間をCSVから抽出
4. **未検知の定義**: 未検知（FN）の基準を設定

【アルゴリズム仕様に基づく分析項目】
- **入力パラメータ確認**: {input_columns}とCSV列の対応関係
- **出力形式検証**: {output_columns}の妥当性確認
- **閾値設定検証**: {thresholds}とデータ分布の整合性
- **データ品質評価**: 欠損値、異常値、範囲チェック
- **信頼度条件確認**: 検知結果が有効となるための前提条件の検証

データセット情報:
{dataset_info}

アルゴリズム仕様:
{algorithm_spec}

評価環境仕様:
{evaluation_spec}

期待値:
{expected_result}

【仕様準拠確認ポイント】
- 必須列の存在確認: {required_columns}
- データ値範囲の妥当性: {value_ranges}
- 有効値の確認: {valid_values}
- 時間軸の連続性と欠損確認

【分析結果の構造化】
1. **データ構造分析**: 列構成、データ型、サイズ
2. **品質メトリクス**: 欠損率、異常値分布、外れ値検出
3. **仕様準拠度**: 仕様との一致度合い
4. **潜在的問題**: 検知精度に影響する可能性のある問題点

分析結果を詳細に報告し、解析手順詳細の次のステップ（未検知データの抽出）につなげる情報を提供してください。

利用可能なツール:
- rag_search: 仕様書の検索（アルゴリズム仕様を優先）
- analyze_data: データ分析
- create_plot: グラフ作成
""")

    def analyze_data(self, dataset: DatasetInfo) -> Dict[str, Any]:
        """
        Analyze the given dataset using dynamic algorithm configuration

        Args:
            dataset: Dataset to analyze

        Returns:
            Analysis results
        """
        try:
            logger.info(f"Starting data analysis for dataset {dataset.id}")

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

            # Prepare dataset info
            dataset_info = self._prepare_dataset_info(dataset)

            # Prepare algorithm-specific context
            algorithm_context = self._prepare_algorithm_context(algorithm_config)

            # Use LLM with tools for analysis
            chain = self.prompt | self.llm.bind_tools(self.tools)

            response = chain.invoke({
                "dataset_info": dataset_info,
                "algorithm_spec": algorithm_spec[:3000],
                "evaluation_spec": evaluation_spec[:2000],
                "expected_result": dataset.expected_result,
                **algorithm_context
            })

            # Process tool calls if any
            if hasattr(response, 'tool_calls') and response.tool_calls:
                results = self._execute_tools(response.tool_calls, dataset)
            else:
                results = {"llm_analysis": response.content}

            # Add basic data analysis
            basic_analysis = self._perform_basic_analysis(dataset, algorithm_config)
            results.update(basic_analysis)

            # Add algorithm configuration info
            results["algorithm_config"] = algorithm_config.model_dump()

            logger.info(f"Data analysis completed for dataset {dataset.id}")
            return results

        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {"error": str(e)}

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
            "required_columns": algorithm_config.required_columns,
            "thresholds": algorithm_config.thresholds,
            "value_ranges": algorithm_config.value_ranges,
            "valid_values": algorithm_config.valid_values
        }

    def _prepare_dataset_info(self, dataset: DatasetInfo) -> str:
        """Prepare dataset information for LLM"""
        info_lines = []

        info_lines.append(f"データセットID: {dataset.id}")
        info_lines.append(f"アルゴリズム出力CSV: {dataset.algorithm_output_csv}")
        info_lines.append(f"コア出力CSV: {dataset.core_output_csv}")
        info_lines.append(f"評価環境仕様: {dataset.evaluation_spec_md}")
        info_lines.append(f"期待値: {dataset.expected_result}")

        return "\n".join(info_lines)

    def _perform_basic_analysis(self, dataset: DatasetInfo, algorithm_config=None) -> Dict[str, Any]:
        """
        Perform basic data analysis with algorithm-specific validation

        Args:
            dataset: Dataset to analyze
            algorithm_config: AlgorithmConfig for validation

        Returns:
            Dictionary with analysis results
        """
        try:
            # Load CSV files
            csv_files = [dataset.algorithm_output_csv, dataset.core_output_csv]
            dataframes = self.repl_tool.load_csv_data(csv_files)

            analysis = {}

            for name, df in dataframes.items():
                if len(df) > 0:
                    basic_info = {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.to_dict(),
                        "summary_stats": df.describe().to_dict(),
                        "missing_values": df.isnull().sum().to_dict(),
                        "sample_data": df.head(5).to_dict('records')
                    }

                    # Add algorithm-specific validation if config available
                    if algorithm_config:
                        validation = self._validate_algorithm_compliance(df, algorithm_config, name)
                        basic_info["algorithm_validation"] = validation

                    analysis[name] = basic_info
                else:
                    analysis[name] = {"error": "Empty or failed to load dataframe"}

            return {"basic_analysis": analysis}

        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
            return {"basic_analysis_error": str(e)}

    def _validate_algorithm_compliance(self, df, algorithm_config, file_type: str) -> Dict[str, Any]:
        """
        Validate dataframe compliance with algorithm specifications

        Args:
            df: DataFrame to validate
            algorithm_config: AlgorithmConfig for validation
            file_type: Type of file (algorithm/core)

        Returns:
            Dictionary with validation results
        """
        validation = {
            "column_compliance": {},
            "value_range_compliance": {},
            "valid_values_compliance": {},
            "threshold_compliance": {}
        }

        # Check required columns
        if file_type == "algorithm_output":
            required_cols = algorithm_config.output_columns + algorithm_config.required_columns
        else:
            required_cols = algorithm_config.input_columns + algorithm_config.required_columns

        for col in required_cols:
            if col in df.columns:
                validation["column_compliance"][col] = "present"
            else:
                validation["column_compliance"][col] = "missing"

        # Check value ranges
        for col, ranges in algorithm_config.value_ranges.items():
            if col in df.columns:
                min_val = ranges.get("min", float('-inf'))
                max_val = ranges.get("max", float('inf'))
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                validation["value_range_compliance"][col] = {
                    "out_of_range_count": int(out_of_range),
                    "total_count": len(df),
                    "percentage": float(out_of_range / len(df) * 100) if len(df) > 0 else 0
                }

        # Check valid values
        for col, valid_vals in algorithm_config.valid_values.items():
            if col in df.columns:
                invalid_count = (~df[col].isin(valid_vals)).sum()
                validation["valid_values_compliance"][col] = {
                    "invalid_count": int(invalid_count),
                    "total_count": len(df),
                    "percentage": float(invalid_count / len(df) * 100) if len(df) > 0 else 0
                }

        # Check threshold compliance for numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            if col in algorithm_config.thresholds:
                threshold = algorithm_config.thresholds[col]
                above_threshold = (df[col] > threshold).sum()
                validation["threshold_compliance"][col] = {
                    "above_threshold_count": int(above_threshold),
                    "threshold_value": threshold,
                    "percentage": float(above_threshold / len(df) * 100) if len(df) > 0 else 0
                }

        return validation

    def _create_rag_search_tool(self):
        """Create RAG search tool"""
        @tool
        def rag_search(query: str, segment: str = "algorithm_specs") -> str:
            """Search specification documents for relevant information, prioritizing algorithm specs"""
            try:
                # Try algorithm specs first, then evaluation specs
                results = []
                segments_to_search = ["algorithm_specs", "evaluation_specs"] if segment == "algorithm_specs" else [segment]

                for seg in segments_to_search:
                    try:
                        seg_results = self.rag_tool.search(query, seg, k=2)
                        results.extend(seg_results)
                    except:
                        continue

                if not results:
                    return "No relevant information found in specifications"

                # Sort by score and return top results
                results.sort(key=lambda x: x['score'], reverse=True)
                top_results = results[:5]

                return "\n\n".join([
                    f"Content: {r['content']}\nSource: {r['source']}\nScore: {r['score']:.3f}"
                    for r in top_results
                ])
            except Exception as e:
                return f"Search failed: {e}"

        return rag_search

    def _create_data_analysis_tool(self):
        """Create data analysis tool"""
        @tool
        def analyze_data(csv_path: str, analysis_type: str = "summary") -> str:
            """Analyze CSV data with pandas"""
            try:
                df = self.repl_tool.load_csv_data([csv_path]).get(csv_path.split('/')[-1].split('\\')[-1])

                if df is None or len(df) == 0:
                    return "Failed to load data or empty dataframe"

                if analysis_type == "summary":
                    result = df.describe().to_string()
                elif analysis_type == "info":
                    result = str(df.info())
                elif analysis_type == "missing":
                    result = df.isnull().sum().to_string()
                else:
                    result = df.head(10).to_string()

                return result

            except Exception as e:
                return f"Analysis failed: {e}"

        return analyze_data

    def _create_plot_creation_tool(self):
        """Create plot creation tool"""
        @tool
        def create_plot(csv_path: str, plot_type: str = "timeseries", column: str = None) -> str:
            """Create plots from CSV data"""
            try:
                df = self.repl_tool.load_csv_data([csv_path]).get(csv_path.split('/')[-1].split('\\')[-1])

                if df is None or len(df) == 0:
                    return "Failed to load data"

                # Determine column to plot
                if column is None and len(df.select_dtypes(include=['number']).columns) > 0:
                    column = df.select_dtypes(include=['number']).columns[0]

                if column not in df.columns:
                    return f"Column {column} not found"

                # Create plot code
                if plot_type == "timeseries" and 'timestamp' in df.columns:
                    code = f"""
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['{column}'])
plt.title('Time Series: {column}')
plt.xlabel('Timestamp')
plt.ylabel('{column}')
plt.xticks(rotation=45)
plt.tight_layout()
"""
                else:
                    code = f"""
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(df['{column}'].dropna(), bins=30, alpha=0.7)
plt.title('Distribution: {column}')
plt.xlabel('{column}')
plt.ylabel('Frequency')
plt.tight_layout()
"""

                result = self.repl_tool.create_plot(code, {"df": df})

                if result.get("success"):
                    return f"Plot created successfully: {result.get('plot_path')}"
                else:
                    return f"Plot creation failed: {result.get('error')}"

            except Exception as e:
                return f"Plot creation failed: {e}"

        return create_plot

    def _execute_tools(self, tool_calls, dataset: DatasetInfo) -> Dict[str, Any]:
        """Execute tool calls and collect results"""
        results = {}

        for tool_call in tool_calls:
            try:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                if tool_name == 'rag_search':
                    results['rag_results'] = self._create_rag_search_tool()(tool_args.get('query', ''), tool_args.get('segment', 'evaluation_specs'))
                elif tool_name == 'analyze_data':
                    results['data_analysis'] = self._create_data_analysis_tool()(tool_args.get('csv_path', ''), tool_args.get('analysis_type', 'summary'))
                elif tool_name == 'create_plot':
                    results['plot_results'] = self._create_plot_creation_tool()(
                        tool_args.get('csv_path', ''),
                        tool_args.get('plot_type', 'timeseries'),
                        tool_args.get('column')
                    )

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results[f'error_{tool_call["name"]}'] = str(e)

        return results
