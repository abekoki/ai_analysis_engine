"""
Reporter Agent - Generates final analysis reports
"""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..config.config import AlgorithmConfig

from ..config import config
from ..models.state import DatasetInfo
from ..models.types import Hypothesis
from ..tools.repl_tool import REPLTool
from ..tools.rag_tool import RAGTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FrameInterval(BaseModel):
    """Pydantic model for extracting frame intervals from natural language text"""
    start_frame: Optional[int] = Field(None, description="Start frame number of the evaluation interval")
    end_frame: Optional[int] = Field(None, description="End frame number of the evaluation interval")
    has_interval: bool = Field(False, description="Whether a frame interval was found in the text")


class PlotKeywords(BaseModel):
    """Pydantic model for extracting relevant keywords for plotting from specifications"""
    keywords: List[str] = Field(default_factory=list, description="List of relevant keywords for plotting")
    confidence_score: float = Field(0.0, description="Confidence score of the extracted keywords (0.0 to 1.0)")


class ReporterAgent:
    """
    Agent for generating final analysis reports
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.openai.model,
            temperature=config.openai.temperature,
            api_key=config.openai.api_key
        )

        self.repl_tool = REPLTool()
        self.rag_tool = RAGTool()
        self.__init_report_prompt()

    def _extract_plot_keywords_from_specs(self, dataset: DatasetInfo, algorithm_config: 'AlgorithmConfig') -> List[str]:
        """
        Extract relevant keywords for plotting from specifications using RAG
        """
        try:
            # Query RAG system to find relevant columns and metrics mentioned in specs
            spec_files = []
            if hasattr(dataset, 'algorithm_spec_md') and dataset.algorithm_spec_md:
                spec_files.append(dataset.algorithm_spec_md)
            if hasattr(dataset, 'evaluation_spec_md') and dataset.evaluation_spec_md:
                spec_files.append(dataset.evaluation_spec_md)

            if not spec_files:
                # Fallback to basic keywords if no specs available
                return ['confidence', 'score', 'result', 'detection']

            # Build RAG query to extract relevant plotting keywords
            rag_query = f"""
            このアルゴリズム仕様書から、プロットに適した重要な指標や列名を抽出してください。
            特に以下の点を考慮して抽出してください：
            1. 数値データとしてプロット可能な指標
            2. 信頼度や確率を示す指標
            3. 検知結果を示す指標
            4. 分析の主要な出力指標

            仕様書の内容に基づいて、プロットに適したキーワードをリスト形式で返してください。
            """

            # Get relevant information from specs
            relevant_info = ""
            for spec_file in spec_files:
                try:
                    with open(spec_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract relevant sections
                        if '出力仕様' in content or 'output' in content.lower():
                            relevant_info += content
                except Exception as e:
                    logger.warning(f"Failed to read spec file {spec_file}: {e}")

            if relevant_info:
                # Use LLM to extract keywords from the specifications
                extraction_prompt = f"""
                以下のアルゴリズム仕様書から、プロットに適した重要な指標や列名を抽出してください。

                仕様書内容:
                {relevant_info[:2000]}  # Limit content length

                抽出するキーワードの基準:
                1. 数値データとしてプロット可能な指標名
                2. 信頼度・確率を示す用語
                3. 検知結果を示す用語
                4. 分析の主要な出力指標

                結果はPythonリスト形式で返してください。
                例: ['confidence', 'score', 'detection_result', 'probability']
                """

                try:
                    response = self.llm.invoke(extraction_prompt)
                    # Parse the response to extract keywords
                    response_text = response.content.strip()

                    # Try to extract list from response
                    import re
                    list_match = re.search(r'\[([^\]]+)\]', response_text)
                    if list_match:
                        keywords_str = list_match.group(1)
                        keywords = [k.strip().strip("'\"") for k in keywords_str.split(',')]
                        keywords = [k for k in keywords if k]  # Remove empty strings
                        if keywords:
                            logger.info(f"Extracted plot keywords from specs: {keywords}")
                            return keywords

                except Exception as e:
                    logger.warning(f"Failed to extract keywords from specs: {e}")

            # Fallback keywords if extraction fails
            fallback_keywords = ['confidence', 'score', 'result', 'detection', 'probability']
            logger.info(f"Using fallback plot keywords: {fallback_keywords}")
            return fallback_keywords

        except Exception as e:
            logger.error(f"Error extracting plot keywords: {e}")
            return ['confidence', 'score', 'result', 'detection']

    def _extract_relevant_keywords_with_llm(self, spec_content: str, algorithm_config: 'AlgorithmConfig') -> List[str]:
        """
        Extract relevant keywords for plotting from specifications using LLM

        Args:
            spec_content: Specification content as string
            algorithm_config: Algorithm configuration

        Returns:
            List of relevant keywords for plotting
        """
        try:
            # Create prompt for keyword extraction
            keyword_extraction_prompt = f"""あなたはアルゴリズム仕様書からプロットに適したキーワードを抽出する専門家です。
以下の仕様書の内容から、プロット作成に適した重要なキーワードを抽出してください。

仕様書内容:
{spec_content[:3000]}

抽出するキーワードの基準:
1. 数値データとしてプロット可能な指標名（例: confidence, score, probability）
2. 検知結果を示す用語（例: detection, result, status）
3. 信頼度を示す用語（例: confidence, reliability, accuracy）
4. 分析の主要な出力指標名

以下のJSON形式で結果を返してください:
{{
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "confidence_score": 0.85
}}

例:
{{"keywords": ["confidence", "score", "detection", "probability"], "confidence_score": 0.9}}

キーワードは2文字以上20文字以内の英数字のみにしてください。"""

            # Call LLM
            response = self.llm.invoke(keyword_extraction_prompt)

            # Parse JSON response with better error handling
            import json
            response_text = response.content.strip()
            if not response_text:
                logger.warning("LLM returned empty response for keywords")
                return []

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                import re
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1).strip()
                else:
                    # Fallback: remove all ``` markers
                    response_text = re.sub(r'```\w*\n?', '', response_text).strip()

            try:
                result = json.loads(response_text)
                if result.get('keywords') and len(result['keywords']) > 0:
                    # Filter and clean keywords
                    clean_keywords = []
                    for keyword in result['keywords']:
                        # Clean the keyword
                        clean_keyword = str(keyword).lower().strip()
                        # Remove very short or very long keywords
                        if 2 <= len(clean_keyword) <= 20:
                            clean_keywords.append(clean_keyword)

                    if clean_keywords:
                        confidence = result.get('confidence_score', 0.0)
                        logger.info(f"LLM extracted keywords: {clean_keywords[:10]}... (confidence: {confidence})")
                        return clean_keywords

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM keyword response as JSON: {response_text[:200]}...")
                # Try to extract keywords using regex as fallback
                import re
                keyword_matches = re.findall(r'"([^"]+)"\s*:\s*"[^"]*"|"([^"]+)"', response_text)
                extracted_keywords = []
                for match in keyword_matches:
                    keyword = match[0] or match[1]
                    if keyword and len(keyword) > 1 and not keyword.isdigit():
                        extracted_keywords.append(keyword.lower())

                if extracted_keywords:
                    logger.info(f"Fallback regex extracted keywords: {extracted_keywords}")
                    return extracted_keywords

            # Fallback to regex-based extraction
            return self._extract_relevant_keywords_with_regex(spec_content, algorithm_config)

        except Exception as e:
            raise RuntimeError(f"LLM keyword extraction failed: {e}") from e

    def _extract_relevant_keywords_with_regex(self, spec_content: str, algorithm_config: 'AlgorithmConfig') -> List[str]:
        """
        Fallback method to extract keywords using regex patterns

        Args:
            spec_content: Specification content as string
            algorithm_config: Algorithm configuration

        Returns:
            List of relevant keywords
        """
        try:
            keywords = set()

            # Extract column names from tables (markdown format)
            import re

            # Look for table rows with column names
            table_rows = re.findall(r'\|([^\|]+)\|([^\|]+)\|([^\|]+)\|', spec_content)
            for row in table_rows:
                for cell in row:
                    cell = cell.strip()
                    if cell and not cell.startswith('-') and len(cell) > 1:
                        # Clean the cell content
                        clean_cell = re.sub(r'[*_`]', '', cell)
                        if clean_cell and not clean_cell.isdigit():
                            keywords.add(clean_cell.lower())

            # Extract words that look like column names or metrics
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', spec_content)
            for word in words:
                word_lower = word.lower()
                # Filter for likely column names or metrics
                if (len(word) > 2 and
                    not word_lower in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'with', 'will', 'have', 'this', 'that', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'] and
                    not word_lower.startswith(('spec', 'eval', 'test', 'data', 'file', 'path', 'name', 'desc', 'info', 'type', 'form', 'list', 'item', 'valu', 'rang', 'min', 'max', 'def', 'num', 'str', 'int', 'flo', 'bool'))):
                    keywords.add(word_lower)

            # Add algorithm config output columns
            if hasattr(algorithm_config, 'output_columns'):
                for col in algorithm_config.output_columns:
                    if col:
                        # Extract base name from column (remove prefixes/suffixes)
                        base_name = col.lower()
                        base_name = re.sub(r'^(left_|right_|leye_|reye_|face_)', '', base_name)
                        base_name = re.sub(r'(_openness|_closed|_confidence|_score|_result)$', '', base_name)
                        keywords.add(base_name)

            # Add threshold names
            if hasattr(algorithm_config, 'thresholds'):
                for threshold_name in algorithm_config.thresholds.keys():
                    threshold_base = threshold_name.lower().replace('_threshold', '').replace('_', '')
                    keywords.add(threshold_base)

            # Convert to list and filter
            keyword_list = list(keywords)
            # Remove very short or very long keywords
            keyword_list = [k for k in keyword_list if 2 <= len(k) <= 20]

            if keyword_list:
                logger.info(f"Regex extracted keywords: {keyword_list[:10]}...")
                return keyword_list
            else:
                # Fallback keywords
                return ['confidence', 'score', 'result', 'detection', 'probability']

        except Exception as e:
            logger.error(f"Error extracting keywords with regex: {e}")
            return ['confidence', 'score', 'result', 'detection', 'probability']

    def _extract_relevant_keywords_from_specs(self, dataset: DatasetInfo, algorithm_config: 'AlgorithmConfig') -> List[str]:
        """
        Extract relevant keywords for plotting from specifications using LLM with fallback to regex
        """
        try:
            # Read specification files
            spec_files = []
            if hasattr(dataset, 'algorithm_spec_md') and dataset.algorithm_spec_md:
                spec_files.append(dataset.algorithm_spec_md)
            if hasattr(dataset, 'evaluation_spec_md') and dataset.evaluation_spec_md:
                spec_files.append(dataset.evaluation_spec_md)

            if not spec_files:
                return ['confidence', 'score', 'result', 'detection', 'probability']

            # Combine all spec content
            combined_content = ""
            for spec_file in spec_files:
                try:
                    with open(spec_file, 'r', encoding='utf-8') as f:
                        combined_content += f.read() + "\n"
                except Exception as e:
                    logger.warning(f"Failed to read spec file {spec_file}: {e}")

            if combined_content:
                # Use LLM-based extraction only
                return self._extract_relevant_keywords_with_llm(combined_content, algorithm_config)
            else:
                return ['confidence', 'score', 'result', 'detection', 'probability']

        except Exception as e:
            raise RuntimeError(f"Keyword extraction from specs failed: {e}") from e

    def _extract_frame_interval_with_llm(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """
        Extract frame interval from natural language text using LLM

        Args:
            text: Natural language text containing frame interval information

        Returns:
            Tuple of (start_frame, end_frame) or (None, None) if not found
        """
        try:
            # Create prompt for frame interval extraction
            frame_extraction_prompt = f"""あなたはテキストからフレーム区間を抽出する専門家です。
以下のテキストから、評価対象となるフレーム区間を抽出してください。

テキスト内容:
{text}

以下のJSON形式で結果を返してください:
{{
    "start_frame": 開始フレーム番号（整数、見つからない場合はnull）,
    "end_frame": 終了フレーム番号（整数、見つからない場合はnull）,
    "has_interval": フレーム区間が見つかったかどうか（true/false）
}}

例:
{{"start_frame": 465, "end_frame": 593, "has_interval": true}}

テキストにフレーム区間が明記されていない場合は、start_frameとend_frameをnullにし、has_intervalをfalseにしてください。
フレーム区間は「フレーム区間XXX-YYY」や「XXX〜YYYフレーム」などの形式で表現されることが多いです。"""

            # Call LLM
            response = self.llm.invoke(frame_extraction_prompt)

            # Parse JSON response
            import json
            try:
                result = json.loads(response.content.strip())
                if result.get('has_interval') and result.get('start_frame') is not None and result.get('end_frame') is not None:
                    start_frame = int(result['start_frame'])
                    end_frame = int(result['end_frame'])
                    logger.info(f"LLM extracted frame interval: {start_frame}-{end_frame}")
                    return start_frame, end_frame
                else:
                    logger.info("No frame interval found in text using LLM")
                    return None, None
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return None, None

        except Exception as e:
            raise RuntimeError(f"LLM frame interval extraction failed: {e}") from e

    def _extract_frame_interval_with_regex(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """
        Fallback method to extract frame interval using regex patterns

        Args:
            text: Text to search for frame intervals

        Returns:
            Tuple of (start_frame, end_frame) or (None, None) if not found
        """
        import re

        # Extract frame range from expected result using regex
        frame_match = re.search(r'フレーム(?:区間)?\s*(\d+)\s*[-〜~]\s*(\d+)', text)
        if frame_match:
            start_frame = int(frame_match.group(1))
            end_frame = int(frame_match.group(2))
            logger.info(f"Regex extracted frame interval: {start_frame}-{end_frame}")
            return start_frame, end_frame

        return None, None

    def __init_report_prompt(self):
        """Initialize the report prompt template"""
        self.report_prompt = ChatPromptTemplate.from_template("""
あなたは汎用レポート作成エージェントです。このシステムは様々なアルゴリズムに対して適用可能な汎用AI分析エンジンです。





【レポート構造】
1. **概要**: 結論、解析対象動画、フレーム区間、期待値、検知結果
2. **確認結果**: グラフ表示、分析結果、考えられる原因
3. **推奨事項**: 具体的な改善提案
4. **参照した仕様/コード（抜粋）**: 使用した仕様書の参照

【結論作成のガイドライン】
- 「考えられる原因」項目の内容を端的に整理する
- ユーザーに直感的にわかりやすい表現を使用する
- 技術的な詳細は避け、問題の本質を明確に伝える
- 1-2文程度の簡潔なまとめとする

【アルゴリズム仕様に基づくレポート項目】
- **評価指標確認**: {thresholds}と実際のデータ分布の比較
- **入力特徴量検証**: {input_columns}の妥当性と分布分析
- **出力形式確認**: {output_columns}の仕様準拠状況
- **値範囲検証**: {value_ranges}の遵守状況確認
- **前提条件分析**: 検知結果が有効となるための信頼度・品質条件の充足状況

データセット情報:
{dataset_info}

アルゴリズム仕様:
{algorithm_spec}

評価環境仕様:
{evaluation_spec}

分析結果:
{analysis_results}

仮説と検証結果:
{hypotheses_results}

【レポート作成ガイドライン】

# 個別データ分析レポート - {dataset_id}

## 概要

- 結論: [考えられる原因を端的に整理したユーザーに直感的にわかりやすい内容]
- 解析対象動画: {dataset_id}
- フレーム区間: [データセットから取得したフレーム区間]
- 期待値: [データセットから取得した期待値]
- 検知結果: [分析結果に基づく検知結果]

## 確認結果

![アルゴリズム出力結果のグラフ]({algorithm_plot_link})
アルゴリズム出力結果
<!-- アルゴリズム出力データの時系列グラフ（閾値付き）-->

![コア出力結果のグラフ]({core_plot_link})
コア出力結果
<!-- コア出力データの時系列グラフ（閾値付き）-->

<!-- 上記のグラフを生成後、閉眼傾向があるかを仮説検証にて確認し、結果を以下に記載 -->
- 入出力の確認結果: [具体的な数値分析結果]

- 考えられる原因: [分析結果に基づき、1つ以上の原因を箇点で整理]


## 推奨事項

- [具体的な改善提案と次のステップ]

## 参照した仕様/コード（抜粋）
... <!-- 仮説検証にて参照した仕様/コードをすべて記載-->

---

""")

    def generate_report(self, dataset: DatasetInfo,
                       analysis_results: Dict[str, Any],
                       hypotheses: List[Hypothesis]) -> str:
        """
        Generate a comprehensive report for the dataset using algorithm configuration

        Args:
            dataset: Dataset information
            analysis_results: Results from all analyses
            hypotheses: Generated and verified hypotheses

        Returns:
            Markdown report content
        """
        try:
            logger.info(f"Generating report for dataset {dataset.id}")

            # Load algorithm configuration dynamically
            algorithm_config = self._load_algorithm_config(dataset)

            # Load specifications
            algorithm_spec = ""
            if dataset.algorithm_spec_md:
                try:
                    with open(dataset.algorithm_spec_md, 'r', encoding='utf-8') as f:
                        algorithm_spec = f.read()
                        logger.info(f"Loaded algorithm spec: {len(algorithm_spec)} characters")
                except Exception as e:
                    logger.warning(f"Failed to load algorithm spec: {e}")

            evaluation_spec = ""
            if dataset.evaluation_spec_md:
                try:
                    with open(dataset.evaluation_spec_md, 'r', encoding='utf-8') as f:
                        evaluation_spec = f.read()
                        logger.info(f"Loaded evaluation spec: {len(evaluation_spec)} characters")
                except Exception as e:
                    logger.warning(f"Failed to load evaluation spec: {e}")

            # Prepare algorithm-specific context
            algorithm_context = self._prepare_algorithm_context(algorithm_config)

            # Debug: Log specification content summaries
            logger.info(f"Algorithm spec preview: {algorithm_spec[:200] if algorithm_spec else 'None'}")
            logger.info(f"Evaluation spec preview: {evaluation_spec[:200] if evaluation_spec else 'None'}")

            # Generate visualization plots
            plot_files = self._generate_visualization_plots(dataset, algorithm_config)

            # Prepare report data
            dataset_info = self._prepare_dataset_info(dataset)
            analysis_summary = self._summarize_analysis_results(analysis_results)
            hypotheses_summary = self._summarize_hypotheses(hypotheses)
            hypotheses_results = hypotheses_summary  # Alias for template compatibility

            # Prepare plot links with relative path from reports directory
            algorithm_plot_link = f"../plots/{dataset.id}/{plot_files['algorithm']}" if plot_files['algorithm'] else ""
            core_plot_link = f"../plots/{dataset.id}/{plot_files['core']}" if plot_files['core'] else ""

            # Use LLM to generate report with algorithm context
            chain = self.report_prompt | self.llm

            # Log data context used for I/O confirmation result generation
            logger.info("Data context for I/O confirmation result generation:")
            logger.info(f"- Dataset ID: {dataset.id}")
            logger.info(f"- Algorithm spec length: {len(algorithm_spec)} chars (truncated to 3000)")
            logger.info(f"- Evaluation spec length: {len(evaluation_spec)} chars (truncated to 2000)")
            logger.info(f"- Analysis results: {len(analysis_results)} chars")
            logger.info(f"- Hypotheses results: {len(hypotheses_results)} chars")
            logger.info(f"- Algorithm plot link: {algorithm_plot_link}")
            logger.info(f"- Core plot link: {core_plot_link}")
            logger.info(f"- Algorithm context keys: {list(algorithm_context.keys())}")

            response = chain.invoke({
                "dataset_id": dataset.id,
                "dataset_info": dataset_info,
                "algorithm_spec": algorithm_spec[:3000],
                "evaluation_spec": evaluation_spec[:2000],
                "analysis_results": analysis_summary,
                "hypotheses_results": hypotheses_summary,
                "algorithm_plot_link": algorithm_plot_link,
                "core_plot_link": core_plot_link,
                **algorithm_context
            })

            # Process the LLM response
            report_content = response.content.strip()

            # Add algorithm configuration info to report
            if algorithm_config:
                report_content += f"\n\n## アルゴリズム設定情報\n"
                report_content += f"- アルゴリズム名: {algorithm_config.name}\n"
                report_content += f"- 閾値設定: {algorithm_config.thresholds}\n"
                report_content += f"- 必須列: {algorithm_config.required_columns}\n"

            logger.info(f"Generated report for dataset {dataset.id}: {len(report_content)} characters")
            return report_content

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            # Return basic error report
            return f"""# エラーレポート - {dataset.id}

## エラー発生
レポート生成中にエラーが発生しました: {e}

## 基本情報
- データセットID: {dataset.id}
- 期待値: {dataset.expected_result}
"""

    def _generate_visualization_plots(self, dataset: DatasetInfo, algorithm_config: 'AlgorithmConfig') -> Dict[str, str]:
        """Generate visualization plots for algorithm and core output data"""
        plot_files = {"algorithm": "", "core": ""}
        try:
            logger.info(f"Generating visualization plots for dataset {dataset.id}")
            logger.info("\n\n\n======\n\n\n")

            # Create plots directory in the same location as reports
            plots_dir = Path("output/results/reports/plots") / dataset.id
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Load data using REPL tool
            algorithm_data_path = dataset.algorithm_output_csv
            core_data_path = dataset.core_output_csv

            if algorithm_data_path and core_data_path:
                # Generate algorithm output plot with target interval filtering
                algo_plot_path = plots_dir / "algorithm_output_plot.png"
                self._generate_algorithm_output_plot(algorithm_data_path, algorithm_config, algo_plot_path, dataset)

                # Generate core output plot with target interval filtering
                core_plot_path = plots_dir / "core_output_plot.png"
                self._generate_core_output_plot(core_data_path, algorithm_config, core_plot_path, dataset)

                # Check actual generated files (they might have different names)
                actual_files = list(plots_dir.glob("*.png"))
                logger.info(f"Found plot files in {plots_dir}: {[f.name for f in actual_files]}")
                if actual_files:
                    # Sort files by name to ensure consistent ordering
                    actual_files.sort(key=lambda x: x.name)
                    # Assume first file is algorithm, second is core
                    plot_files["algorithm"] = actual_files[0].name if len(actual_files) > 0 else ""
                    plot_files["core"] = actual_files[1].name if len(actual_files) > 1 else ""
                    logger.info(f"Assigned plot files: algorithm='{plot_files['algorithm']}', core='{plot_files['core']}'")

                logger.info(f"Generated plots for dataset {dataset.id}: {plot_files}")
            else:
                logger.warning(f"Data paths not available for dataset {dataset.id}")

        except Exception as e:
            logger.error(f"Failed to generate plots for dataset {dataset.id}: {e}")

        return plot_files

    def _infer_x_axis_column(self, df_columns: List[str], spec_content: str = "", algorithm_config: 'AlgorithmConfig' = None) -> str:
        """
        Infer the most appropriate column for x-axis (time/frame axis) using LLM

        Args:
            df_columns: List of available columns in the dataframe
            spec_content: Algorithm specification content
            algorithm_config: Algorithm configuration

        Returns:
            Column name to use for x-axis, or empty string if none found
        """
        try:
            # First, try obvious candidates
            obvious_x_columns = ['frame', 'frame_num', 'timestamp', 'time', 'index', 'frame_number']
            for col in obvious_x_columns:
                if col in df_columns:
                    return col

            # If no obvious candidates, use LLM to infer
            if not spec_content and algorithm_config:
                # Try to get spec content from algorithm config
                try:
                    import os
                    if hasattr(algorithm_config, 'spec_file') and algorithm_config.spec_file:
                        if os.path.exists(algorithm_config.spec_file):
                            with open(algorithm_config.spec_file, 'r', encoding='utf-8') as f:
                                spec_content = f.read()[:1000]  # Limit content
                except:
                    pass

            if spec_content:
                x_axis_prompt = f"""以下のアルゴリズム仕様と利用可能な列名から、時系列プロットのx軸に最も適した列名を1つ選んでください。

利用可能な列名: {df_columns}

アルゴリズム仕様:
{spec_content[:1000]}

x軸に適した列の基準:
1. フレーム番号や時間情報を表す列
2. 連続した数値データである列
3. ソート可能な列（時系列データ）

最も適した列名のみを返してください。該当する列がない場合は空文字を返してください。

例: frame, timestamp, frame_num"""

                try:
                    response = self.llm.invoke(x_axis_prompt)
                    inferred_column = response.content.strip()

                    # Validate the inferred column exists in df_columns
                    if inferred_column in df_columns:
                        logger.info(f"LLM inferred x-axis column: {inferred_column}")
                        return inferred_column
                    else:
                        logger.warning(f"LLM inferred column '{inferred_column}' not found in available columns")
                except Exception as e:
                    logger.warning(f"Failed to infer x-axis column with LLM: {e}")

            # Fallback: look for numeric columns that might represent time/frame
            for col in df_columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['frame', 'time', 'index', 'number', 'num']):
                    return col

            # Last resort: use first numeric column
            import pandas as pd
            try:
                # We can't load the full dataframe here, so return empty and let caller handle
                return ""
            except:
                return ""

        except Exception as e:
            logger.error(f"Error inferring x-axis column: {e}")
            return ""

    def _generate_algorithm_output_plot(self, data_path: str, algorithm_config: 'AlgorithmConfig', output_path: Path, dataset: DatasetInfo = None):
        """Generate plot for algorithm output data with thresholds"""
        try:
            # Get column mapping from dataset
            column_mapping = {}
            if dataset and hasattr(dataset, 'data_summary') and dataset.data_summary:
                column_mapping = dataset.data_summary.get('column_mapping', {}).get('column_mapping', {})

            # Extract threshold information outside f-string
            thresholds_code = ""
            if hasattr(algorithm_config, 'thresholds') and algorithm_config.thresholds:
                thresholds_list = []
                for threshold_name, threshold_value in algorithm_config.thresholds.items():
                    thresholds_list.append(f"('{threshold_name}', {threshold_value})")
                thresholds_code = f"thresholds = [{', '.join(thresholds_list)}]"
            else:
                thresholds_code = "thresholds = []"

            # Determine columns to plot based on column mapping
            plot_columns_code = ""
            plot_columns = []

            # Priority 1: Use column mapping from specs
            if column_mapping and 'output' in column_mapping and column_mapping['output']:
                output_cols = [mapped_col for mapped_col in column_mapping['output'].values() if mapped_col]
                if output_cols:
                    plot_columns_code = f"plot_columns = {output_cols}"
                    plot_columns = output_cols
                    logger.info(f"Algorithm output plot - using mapped columns from specs: {plot_columns}")

            # Priority 2: Use algorithm config output columns
            elif hasattr(algorithm_config, 'output_columns') and algorithm_config.output_columns:
                output_cols = [col for col in algorithm_config.output_columns if col]
                if output_cols:
                    plot_columns_code = f"plot_columns = {output_cols}"
                    plot_columns = output_cols
                    logger.info(f"Algorithm output plot - using config columns: {plot_columns}")

            # Priority 3: Extract from spec using RAG keywords
            else:
                relevant_keywords = self._extract_relevant_keywords_from_specs(dataset, algorithm_config)
                if relevant_keywords:
                    # Load data to check columns
                    import pandas as pd
                    try:
                        df = pd.read_csv(data_path)
                        # Try to find columns that match the keywords
                        matched_cols = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in relevant_keywords)]
                        if matched_cols:
                            plot_columns_code = f"plot_columns = {matched_cols[:4]}"  # Limit to 4 columns
                            plot_columns = matched_cols[:4]
                            logger.info(f"Algorithm output plot - using RAG-matched columns: {plot_columns}")
                        else:
                            # Fallback to numeric columns that might match keywords
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            keyword_matched_numeric = [col for col in numeric_cols if any(keyword.lower() in col.lower() for keyword in relevant_keywords)]
                            if keyword_matched_numeric:
                                plot_columns_code = f"plot_columns = {keyword_matched_numeric[:4]}"
                                plot_columns = keyword_matched_numeric[:4]
                                logger.info(f"Algorithm output plot - using keyword-matched numeric columns: {plot_columns}")
                    except Exception as e:
                        logger.warning(f"Failed to load data for column matching: {e}")

            # Last resort: use numeric columns
            if not plot_columns:
                try:
                    import pandas as pd
                    df = pd.read_csv(data_path)
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        plot_columns_code = f"plot_columns = {numeric_cols[:4]}"
                        plot_columns = numeric_cols[:4]
                        logger.info(f"Algorithm output plot - fallback to numeric columns: {plot_columns}")
                except Exception as e:
                    logger.warning(f"Failed to load data for fallback: {e}")
                    plot_columns_code = "plot_columns = ['column1', 'column2']"  # Dummy fallback
                    plot_columns = ['column1', 'column2']

            logger.info(f"Algorithm output plot - final columns: {plot_columns}")

            # Log final columns used for plotting and analysis
            logger.info(f"PLOT_COLUMNS_USED - Algorithm: {plot_columns}")

            # Infer x-axis column using LLM
            spec_content = ""
            if dataset and hasattr(dataset, 'algorithm_spec_md') and dataset.algorithm_spec_md:
                try:
                    with open(dataset.algorithm_spec_md, 'r', encoding='utf-8') as f:
                        spec_content = f.read()[:1000]  # Limit content for LLM
                except Exception as e:
                    logger.warning(f"Failed to load spec for x-axis inference: {e}")

            # Load data to get column list for x-axis inference
            x_axis_column = ""
            try:
                import pandas as pd
                temp_df = pd.read_csv(data_path)
                x_axis_column = self._infer_x_axis_column(list(temp_df.columns), spec_content, algorithm_config)
                if x_axis_column:
                    logger.info(f"Using inferred x-axis column for algorithm plot: {x_axis_column}")
                else:
                    logger.info("No suitable x-axis column found, using default")
            except Exception as e:
                logger.warning(f"Failed to infer x-axis column: {e}")

            # Extract target interval from dataset if available
            target_interval_code = ""
            start_frame = None
            end_frame = None

            # First, try to get from consistency_check
            if dataset and hasattr(dataset, 'consistency_check') and dataset.consistency_check:
                cc = dataset.consistency_check
                if isinstance(cc, dict) and 'target_interval' in cc and cc['target_interval']:
                    interval = cc['target_interval']
                    start_frame = interval.get('start')
                    end_frame = interval.get('end')

            # If not found in consistency_check, extract from expected_result using LLM
            if start_frame is None and dataset and hasattr(dataset, 'expected_result'):
                expected = dataset.expected_result
                # Extract frame range from expected result using LLM
                start_frame, end_frame = self._extract_frame_interval_with_llm(expected)

            if start_frame is not None and end_frame is not None:
                target_interval_code = f"""
# Filter data to target interval
if 'frame' in df.columns:
    df = df[(df['frame'] >= {start_frame}) & (df['frame'] <= {end_frame})]
elif 'frame_num' in df.columns:
    df = df[(df['frame_num'] >= {start_frame}) & (df['frame_num'] <= {end_frame})]
# Reset index after filtering
df = df.reset_index(drop=True)
"""

            # Create plot generation code
            plot_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# X-axis column configuration
x_axis_column = '{x_axis_column}'

# Output path configuration
output_path = r'{str(output_path)}'

# Load data
df = pd.read_csv(r'{data_path}')

# Filter to target interval if available
{target_interval_code}

# Define frame range variables for axis limits
start_frame_val = {start_frame if start_frame is not None else 'None'}
end_frame_val = {end_frame if end_frame is not None else 'None'}

# Thresholds configuration
{thresholds_code}

# Determine columns to plot
{plot_columns_code}

# Filter to columns that exist in the data
plot_cols = [col for col in plot_columns if col in df.columns]
if not plot_cols:
    plot_cols = df.columns[:min(5, len(df.columns))]

# Create figure with subplots
fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, 4*len(plot_cols)))
if len(plot_cols) == 1:
    axes = [axes]

    # Determine x-axis data using inferred column
    x_data = df.index
    x_label = 'Index'
    if x_axis_column and x_axis_column in df.columns:
        x_data = df[x_axis_column]
        x_label = x_axis_column.replace('_', ' ').title()
    else:
        # Fallback to common patterns
        for col in ['frame', 'frame_num', 'timestamp', 'time']:
            if col in df.columns:
                x_data = df[col]
                x_label = col.replace('_', ' ').title()
                break

# Plot each column with thresholds
for i, col in enumerate(plot_cols):
    axes[i].plot(x_data, df[col], label=col, linewidth=2)

    # Add threshold lines if available and column matches threshold
    for threshold_name, threshold_value in thresholds:
        # Apply threshold if column name matches threshold name (case-insensitive partial match)
        if threshold_name.lower().replace('_threshold', '').replace('_', '') in col.lower():
            label_text = threshold_name + ': ' + str(threshold_value)
            axes[i].axhline(y=threshold_value, color='red', linestyle='--', label=label_text)

    axes[i].set_title(col + ' - Algorithm Output')
    axes[i].set_xlabel(x_label)
    axes[i].set_ylabel('Value')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
"""


            # Execute plot generation
            result = self.repl_tool.execute_code(plot_code)
            if result.get('success'):
                logger.info(f"Generated algorithm output plot: {output_path}")
            else:
                logger.error(f"Failed to generate algorithm output plot: {result.get('error')}")
                logger.error(f"Plot code snippet around error:\n{plot_code[-500:]}")

        except Exception as e:
            logger.error(f"Error generating algorithm output plot: {e}")

    def _generate_core_output_plot(self, data_path: str, algorithm_config: 'AlgorithmConfig', output_path: Path, dataset: DatasetInfo = None):
        """Generate plot for core output data with thresholds"""
        try:
            # Get column mapping from dataset
            column_mapping = {}
            if dataset and hasattr(dataset, 'data_summary') and dataset.data_summary:
                column_mapping = dataset.data_summary.get('column_mapping', {}).get('column_mapping', {})

            # Determine input columns to plot with multiple strategies
            input_cols = []

            # Priority 1: Use column mapping from specs
            if column_mapping and 'input' in column_mapping and column_mapping['input']:
                input_cols = [mapped_col for mapped_col in column_mapping['input'].values() if mapped_col]
                logger.info(f"Core output plot - using mapped input columns from specs: {input_cols}")

            # Priority 2: Use algorithm config input columns
            elif hasattr(algorithm_config, 'input_columns') and algorithm_config.input_columns:
                input_cols = algorithm_config.input_columns
                logger.info(f"Core output plot - using config input columns: {input_cols}")

            # Priority 3: Extract from spec using RAG keywords
            else:
                relevant_keywords = self._extract_relevant_keywords_from_specs(dataset, algorithm_config)
                if relevant_keywords:
                    # Load data to check columns
                    import pandas as pd
                    try:
                        df = pd.read_csv(data_path)
                        # Find columns that match the keywords
                        matched_cols = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in relevant_keywords)]
                        if matched_cols:
                            input_cols = matched_cols[:6]  # Limit to 6 columns for core plots
                            logger.info(f"Core output plot - using RAG-matched columns: {input_cols}")
                        else:
                            # Fallback to numeric columns that might match keywords
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            keyword_matched_numeric = [col for col in numeric_cols if any(keyword.lower() in col.lower() for keyword in relevant_keywords)]
                            if keyword_matched_numeric:
                                input_cols = keyword_matched_numeric[:6]
                                logger.info(f"Core output plot - using keyword-matched numeric columns: {input_cols}")
                    except Exception as e:
                        logger.warning(f"Failed to load data for column matching: {e}")

            # Last resort: use numeric columns
            if not input_cols:
                try:
                    import pandas as pd
                    df = pd.read_csv(data_path)
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        # Prefer common eye-tracking columns
                        preferred_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['confidence', 'openness', 'leye', 'reye', 'face'])]
                        input_cols = (preferred_cols + numeric_cols)[:6]  # Combine preferred and others
                        logger.info(f"Core output plot - fallback to numeric columns: {input_cols}")
                except Exception as e:
                    logger.warning(f"Failed to load data for fallback: {e}")
                    input_cols = ['face_confidence', 'leye_openness', 'reye_openness']  # Eye-tracking fallback

            input_cols_str = str(input_cols)
            logger.info(f"Core output plot - final input columns: {input_cols}")

            # Log final columns used for plotting and analysis
            logger.info(f"PLOT_COLUMNS_USED - Core: {input_cols}")

            # Infer x-axis column using LLM for core plot
            spec_content = ""
            if dataset and hasattr(dataset, 'algorithm_spec_md') and dataset.algorithm_spec_md:
                try:
                    with open(dataset.algorithm_spec_md, 'r', encoding='utf-8') as f:
                        spec_content = f.read()[:1000]  # Limit content for LLM
                except Exception as e:
                    logger.warning(f"Failed to load spec for core plot x-axis inference: {e}")

            # Load data to get column list for x-axis inference
            x_axis_column = ""
            try:
                import pandas as pd
                temp_df = pd.read_csv(data_path)
                x_axis_column = self._infer_x_axis_column(list(temp_df.columns), spec_content, algorithm_config)
                if x_axis_column:
                    logger.info(f"Using inferred x-axis column for core plot: {x_axis_column}")
                else:
                    logger.info("No suitable x-axis column found for core plot, using default")
            except Exception as e:
                logger.warning(f"Failed to infer x-axis column for core plot: {e}")

            thresholds_code = ""
            if column_mapping and 'thresholds' in column_mapping:
                # Use thresholds from column mapping
                thresholds_list = []
                for threshold_name, threshold_value in column_mapping['thresholds'].items():
                    thresholds_list.append(f"('{threshold_name}', {threshold_value})")
                thresholds_code = f"thresholds = [{', '.join(thresholds_list)}]"
            elif hasattr(algorithm_config, 'thresholds') and algorithm_config.thresholds:
                thresholds_list = []
                for threshold_name, threshold_value in algorithm_config.thresholds.items():
                    thresholds_list.append(f"('{threshold_name}', {threshold_value})")
                thresholds_code = f"thresholds = [{', '.join(thresholds_list)}]"
            else:
                thresholds_code = "thresholds = []"

            # Extract relevant keywords from specifications for dynamic column selection
            relevant_keywords = self._extract_relevant_keywords_from_specs(dataset, algorithm_config)

            # Extract target interval from dataset if available
            target_interval_code = ""
            start_frame = None
            end_frame = None

            # First, try to get from consistency_check
            if dataset and hasattr(dataset, 'consistency_check') and dataset.consistency_check:
                cc = dataset.consistency_check
                if isinstance(cc, dict) and 'target_interval' in cc and cc['target_interval']:
                    interval = cc['target_interval']
                    start_frame = interval.get('start')
                    end_frame = interval.get('end')

            # If not found in consistency_check, extract from expected_result using LLM
            if start_frame is None and dataset and hasattr(dataset, 'expected_result'):
                expected = dataset.expected_result
                # Extract frame range from expected result using LLM
                start_frame, end_frame = self._extract_frame_interval_with_llm(expected)

            if start_frame is not None and end_frame is not None:
                target_interval_code = f"""
# Filter data to target interval
if 'frame' in df.columns:
    df = df[(df['frame'] >= {start_frame}) & (df['frame'] <= {end_frame})]
elif 'frame_num' in df.columns:
    df = df[(df['frame_num'] >= {start_frame}) & (df['frame_num'] <= {end_frame})]
# Reset index after filtering
df = df.reset_index(drop=True)
"""

            # Create plot generation code
            plot_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# X-axis column configuration
x_axis_column = '{x_axis_column}'

# Output path configuration
output_path = r'{str(output_path)}'

# Load data
df = pd.read_csv(r'{data_path}')

# Filter to target interval if available
{target_interval_code}

# Define frame range variables for axis limits
start_frame_val = {start_frame if start_frame is not None else 'None'}
end_frame_val = {end_frame if end_frame is not None else 'None'}

# Input columns and thresholds configuration
input_cols = {input_cols_str}
relevant_keywords = {relevant_keywords}
{thresholds_code}

# Determine columns to plot from input columns with keyword prioritization
plot_cols = []
if input_cols:
    # Prioritize columns based on extracted keywords
    if relevant_keywords:
        prioritized_cols = []
        other_cols = []

        for col in input_cols:
            if col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in relevant_keywords):
                    prioritized_cols.append(col)
                else:
                    other_cols.append(col)

        # Combine prioritized and other columns
        plot_cols = prioritized_cols + other_cols[:max(0, 6 - len(prioritized_cols))]
    else:
        plot_cols = [col for col in input_cols if col in df.columns][:6]

# If no input columns specified or none exist, determine columns dynamically based on data types and threshold names
if not plot_cols:
    # First, try to find columns that match threshold names
    threshold_base_names = []
    for threshold_name in thresholds:
        base_name = threshold_name[0].lower().replace('_threshold', '').replace('_', '')
        threshold_base_names.append(base_name)

    # Look for columns that match threshold names
    for col in df.columns:
        col_lower = col.lower()
        if any(base_name in col_lower for base_name in threshold_base_names):
            plot_cols.append(col)

    # If still no matches, look for numeric columns (likely to be features)
    if not plot_cols:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col.lower() not in ['frame', 'index', 'timestamp']:
                plot_cols.append(col)
                if len(plot_cols) >= 6:  # Limit to 6 columns max
                    break

# If still no columns, use first few numeric columns
if not plot_cols:
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            plot_cols.append(col)
            if len(plot_cols) >= 6:  # Limit to 6 columns max
                break

# Ensure we have at least one column to plot
if len(plot_cols) == 0:
    print("No columns available for plotting")
else:
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, 4*len(plot_cols)))
    if len(plot_cols) == 1:
        axes = [axes]

    # Determine x-axis data using inferred column
    x_data = df.index
    x_label = 'Index'
    if x_axis_column and x_axis_column in df.columns:
        x_data = df[x_axis_column]
        x_label = x_axis_column.replace('_', ' ').title()
    else:
        # Fallback to common patterns
        for col in ['frame', 'frame_num', 'timestamp', 'time']:
            if col in df.columns:
                x_data = df[col]
                x_label = col.replace('_', ' ').title()
                break

    # Plot each input column with thresholds
    for i, col in enumerate(plot_cols):
        axes[i].plot(x_data, df[col], label=col, linewidth=2)

        # Add threshold lines if available and column matches threshold
        for threshold_name, threshold_value in thresholds:
            # Apply threshold if column name matches threshold name (case-insensitive partial match)
            if threshold_name.lower().replace('_threshold', '').replace('_', '') in col.lower():
                label_text = threshold_name + ': ' + str(threshold_value)
                axes[i].axhline(y=threshold_value, color='red', linestyle='--', label=label_text)

        axes[i].set_title(col + ' - Core Input Feature')
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
"""

            # Execute plot generation
            result = self.repl_tool.execute_code(plot_code)
            if result.get('success'):
                logger.info(f"Generated core output plot: {output_path}")
            else:
                logger.error(f"Failed to generate core output plot: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error generating core output plot: {e}")

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

    def _prepare_dataset_info(self, dataset: DatasetInfo) -> str:
        """Prepare dataset information for report generation"""
        info_lines = [
            f"データセットID: {dataset.id}",
            f"期待値: {dataset.expected_result}",
            f"アルゴリズム出力: {dataset.algorithm_output_csv}",
            f"コア出力: {dataset.core_output_csv}",
            f"評価環境仕様: {dataset.evaluation_spec_md}"
        ]

        # Add evaluation interval information if available
        if dataset.evaluation_interval:
            interval = dataset.evaluation_interval
            if interval.get('start') is not None and interval.get('end') is not None:
                info_lines.append(f"フレーム区間: {interval['start']} - {interval['end']}")
            else:
                info_lines.append("フレーム区間: 未指定")

        return "\n".join(info_lines)

    def _summarize_analysis_results(self, analysis_results: Dict[str, Any]) -> str:
        """Summarize analysis results"""
        if not analysis_results:
            return "分析結果なし"

        summary_lines = []

        # Data analysis summary
        if "basic_analysis" in analysis_results:
            basic = analysis_results["basic_analysis"]
            for name, analysis in basic.items():
                if isinstance(analysis, dict) and "shape" in analysis:
                    shape = analysis["shape"]
                    summary_lines.append(f"{name}: {shape[0]}行 x {shape[1]}列")

        # Consistency summary
        if "consistency_check" in analysis_results:
            consistency = analysis_results["consistency_check"]
            if consistency.get("overall_consistent"):
                summary_lines.append("整合性: 問題なし")
            else:
                issues = consistency.get("issues", [])
                summary_lines.append(f"整合性の問題: {len(issues)}個")

        return "\n".join(summary_lines)

    def _summarize_hypotheses(self, hypotheses: List[Hypothesis]) -> str:
        """Summarize hypotheses and their verification results"""
        if not hypotheses:
            return "仮説なし"

        summary_lines = []

        for hypo in hypotheses:
            status = "検証済み" if hypo.verification_status.value == "verified" else "未検証"
            summary_lines.append(f"- {hypo.type.value}: {hypo.description}")
            summary_lines.append(f"  信頼度: {hypo.confidence_score:.2f}, 状態: {status}")

            if hypo.suggested_fix:
                summary_lines.append(f"  提案修正: {hypo.suggested_fix}")

        return "\n".join(summary_lines)

    def _get_existing_plots(self, dataset: DatasetInfo) -> Dict[str, str]:
        """Get existing plots for the dataset"""
        plots = {}

        try:
            # Get plots directory
            plots_dir = config.output_dir / "plots" / dataset.id

            if plots_dir.exists():
                # Find plot files
                for png_file in plots_dir.glob("*.png"):
                    file_name = png_file.stem

                    # Categorize plots based on filename
                    if ('2' in file_name or '4' in file_name or 'algo' in file_name.lower() or
                        'is_drowsy' in file_name.lower() or file_name.endswith('_timeseries')):
                        plots['algorithm'] = str(png_file)
                    elif ('WIN_' in file_name or 'analysis' in file_name.lower() or
                          'leye' in file_name.lower() or 'reye' in file_name.lower() or
                          'confidence' in file_name.lower()):
                        plots['core'] = str(png_file)

            logger.info(f"Found existing plots for {dataset.id}: {list(plots.keys())}")

        except Exception as e:
            logger.error(f"Failed to get existing plots: {e}")

        return plots

    def _generate_report_plots(self, dataset: DatasetInfo) -> Dict[str, str]:
        """Generate plots for the report"""
        plots = {}

        try:
            # Create plots directory for this dataset
            plots_dir = config.output_dir / "reports" / dataset.id
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Load data
            dataframes = self.repl_tool.load_csv_data([
                dataset.algorithm_output_csv,
                dataset.core_output_csv
            ])

            # Generate time series plots
            for name, df in dataframes.items():
                if len(df) > 0 and len(df.select_dtypes(include=['number']).columns) > 0:
                    # Get first numeric column
                    numeric_col = df.select_dtypes(include=['number']).columns[0]

                    # Create plot
                    code = f"""
import matplotlib.pyplot as plt
import pandas as pd

# Create time series plot
plt.figure(figsize=(12, 6))

if 'timestamp' in df.columns:
    plt.plot(df['timestamp'], df['{numeric_col}'], marker='o', linestyle='-')
    plt.xlabel('Timestamp')
else:
    plt.plot(df.index, df['{numeric_col}'], marker='o', linestyle='-')
    plt.xlabel('Index')

plt.ylabel('{numeric_col}')
plt.title('{name} - {numeric_col} Time Series')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
"""

                    plot_path = str(plots_dir / f"{name}_{numeric_col}_timeseries.png")
                    result = self.repl_tool.create_plot(code, {"df": df}, plot_path)

                    if result.get("success"):
                        plots[name] = plot_path

        except Exception as e:
            logger.error(f"Failed to generate report plots: {e}")

        return plots

    def _insert_plots_into_report(self, report_content: str, plots: Dict[str, str], dataset_id: str) -> str:
        """Insert plot references into the report with correct relative paths"""
        # Insert algorithm output plot
        if "algorithm" in plots:
            # Convert absolute path to relative path from reports directory
            # reports/dataset_report.md -> ../../plots/dataset/filename.png
            relative_path = f"../../plots/{dataset_id}/{Path(plots['algorithm']).name}"
            plot_ref = f"\n![アルゴリズム出力結果の時系列グラフ]({relative_path})\n"
            report_content = report_content.replace(
                "### アルゴリズム出力結果の当該タスク区間における時系列グラフ",
                "### アルゴリズム出力結果の当該タスク区間における時系列グラフ" + plot_ref
            )

        # Insert core output plot
        if "core" in plots:
            # Convert absolute path to relative path from reports directory
            relative_path = f"../../plots/{dataset_id}/{Path(plots['core']).name}"
            plot_ref = f"\n![コア出力結果の時系列グラフ]({relative_path})\n"
            report_content = report_content.replace(
                "### コア出力結果の当該タスク区間における時系列グラフ",
                "### コア出力結果の当該タスク区間における時系列グラフ" + plot_ref
            )

        return report_content

    def _generate_fallback_report(self, dataset: DatasetInfo, error: str) -> str:
        """Generate a fallback report in case of errors"""
        return f"""# 個別データ分析レポート

## 概要

- 結論 : 分析中にエラーが発生しました
- 解析対象動画： {dataset.id}
- 期待値： {dataset.expected_result}
- 検知結果： エラーにより確認できませんでした

## エラー情報

{error}

## 推奨事項

- エラーの原因を調査してください
- ログを確認して問題を特定してください
"""
