from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool


def rag_search_tool(rag_tool):
    """Create a generic RAG search tool.

    Tool name: rag_search(query: str, segment: str = "algorithm_specs", k: int = 3) -> str
    """

    @tool
    def rag_search(query: str, segment: str = "algorithm_specs", k: int = 3) -> str:
        """Search RAG documents and return top snippets with source and score."""
        """Search RAG documents. segment in {algorithm_specs|evaluation_specs|algorithm_code|evaluation_code}."""
        try:
            results = rag_tool.search(query, segment, k=k)
            if not results:
                return "No relevant information found"
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            top = results[:5]
            return "\n\n".join([
                f"Content: {r.get('content','')}\nSource: {r.get('source','')}\nScore: {r.get('score',0):.3f}"
                for r in top
            ])
        except Exception as e:
            return f"Search failed: {e}"

    return rag_search


def analyze_data_tool(repl_tool, dataset):
    """Create dataset-bound data analysis tool.

    Tool name: analyze_data(target: str = "algorithm", analysis_type: str = "summary") -> str
    """

    @tool
    def analyze_data(target: str = "algorithm", analysis_type: str = "summary") -> str:
        """Analyze dataset CSV. analysis_type in {summary|info|missing|head}."""
        """Analyze CSV; target in {algorithm|core}; analysis_type in {summary|info|missing|head}."""
        try:
            csv = dataset.algorithm_output_csv if target == "algorithm" else dataset.core_output_csv
            df = repl_tool.load_csv_data([csv]).get(csv.split('/')[-1].split('\\')[-1])
            if df is None or len(df) == 0:
                return "Failed to load data or empty dataframe"
            if analysis_type == "summary":
                return df.describe().to_string()
            if analysis_type == "info":
                return str(df.info())
            if analysis_type == "missing":
                return df.isnull().sum().to_string()
            if analysis_type == "head":
                return df.head(10).to_string()
            return df.head(10).to_string()
        except Exception as e:
            return f"Analysis failed: {e}"

    return analyze_data


def create_plot_tool(repl_tool, dataset):
    """Create dataset-bound plot creation tool.

    Tool name: create_plot(target: str = "algorithm", plot_type: str = "timeseries", column: Optional[str] = None) -> str
    """

    @tool
    def create_plot(target: str = "algorithm", plot_type: str = "timeseries", column: Optional[str] = None) -> str:
        """Create and save a plot from dataset CSV. Returns output path or error string."""
        """Create plot from CSV; target in {algorithm|core}; plot_type in {timeseries|hist}."""
        try:
            csv = dataset.algorithm_output_csv if target == "algorithm" else dataset.core_output_csv
            df = repl_tool.load_csv_data([csv]).get(csv.split('/')[-1].split('\\')[-1])
            if df is None or len(df) == 0:
                return "Failed to load data"

            # Determine column to plot
            if column is None:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    return "No numeric columns"
                column = numeric_cols[0]

            if column not in df.columns:
                return f"Column {column} not found"

            # Build plotting code
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
            elif plot_type == "timeseries":
                code = f"""
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['{column}'])
plt.title('Time Series: {column}')
plt.xlabel('Index')
plt.ylabel('{column}')
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

            result = repl_tool.create_plot(code, {"df": df})
            if result.get("success"):
                return f"Plot created successfully: {result.get('plot_path')}"
            return f"Plot creation failed: {result.get('error')}"
        except Exception as e:
            return f"Plot creation failed: {e}"

    return create_plot


def check_frame_range_tool(repl_tool, dataset):
    """Create dataset-bound frame range coverage tool.

    Tool name: check_frame_range(start: int, end: int, target: str = "algorithm") -> str
    """

    @tool
    def check_frame_range(start: int, end: int, target: str = "algorithm") -> str:
        """Check whether [start,end] frames are covered in the target dataframe and return range info."""
        try:
            csv = dataset.algorithm_output_csv if target == "algorithm" else dataset.core_output_csv
            df = repl_tool.load_csv_data([csv]).get(csv.split('/')[-1].split('\\')[-1])
            if df is None or len(df) == 0:
                return "No data"
            frame_col = 'frame' if 'frame' in df.columns else ('frame_num' if 'frame_num' in df.columns else None)
            if frame_col is None:
                return "No frame column"
            min_f = int(df[frame_col].min())
            max_f = int(df[frame_col].max())
            ok = (start >= min_f and end <= max_f)
            return str({"covered": ok, "data_range": [min_f, max_f]})
        except Exception as e:
            return f"check_frame_range failed: {e}"

    return check_frame_range


