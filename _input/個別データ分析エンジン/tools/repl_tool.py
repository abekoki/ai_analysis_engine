"""
REPL (Read-Eval-Print Loop) Tool

Provides Python code execution capabilities for data analysis and processing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import io
import sys
import contextlib
from pathlib import Path
import traceback

from ..config import config
from ..utils.logger import get_logger
from ..utils.file_utils import read_csv_safe, ensure_directory

logger = get_logger(__name__)


class REPLTool:
    """
    REPL tool for executing Python code in a controlled environment
    """

    def __init__(self):
        self.allowed_modules = config.repl.allowed_modules
        self.timeout = config.repl.timeout
        self.max_output_length = config.repl.max_output_length

        # Setup matplotlib for non-interactive use
        plt.switch_backend('Agg')
        sns.set_style("whitegrid")

    def execute_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code and return results

        Args:
            code: Python code to execute
            context: Additional context variables

        Returns:
            Dictionary with execution results
        """
        if context is None:
            context = {}

        # Prepare execution environment
        exec_globals = self._create_execution_environment(context)

        try:
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, exec_globals)

            # Get the results
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            # Extract the last expression result if it's an expression
            result = None
            if code.strip() and not code.strip().endswith(';'):
                try:
                    result = eval(code.strip(), exec_globals)
                except:
                    pass  # Not an expression, ignore

            return {
                'success': True,
                'result': result,
                'stdout': stdout[:self.max_output_length],
                'stderr': stderr[:self.max_output_length],
                'has_more_output': len(stdout) > self.max_output_length or len(stderr) > self.max_output_length
            }

        except Exception as e:
            error_msg = traceback.format_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': error_msg[:self.max_output_length],
                'stdout': '',
                'stderr': ''
            }

    def load_csv_data(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load CSV data from files

        Args:
            file_paths: List of CSV file paths

        Returns:
            Dictionary mapping file names to DataFrames
        """
        dataframes = {}

        for file_path in file_paths:
            try:
                df = read_csv_safe(file_path)
                file_name = Path(file_path).stem
                dataframes[file_name] = df
                logger.info(f"Loaded CSV {file_path} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load CSV {file_path}: {e}")
                dataframes[Path(file_path).stem] = pd.DataFrame()

        return dataframes

    def analyze_dataframe(self, df: pd.DataFrame, name: str = "data") -> Dict[str, Any]:
        """
        Analyze DataFrame and return summary statistics

        Args:
            df: DataFrame to analyze
            name: Name for the DataFrame

        Returns:
            Dictionary with analysis results
        """
        try:
            # Safe dtype conversion
            dtypes_dict = {}
            try:
                for col in df.columns:
                    dtypes_dict[col] = str(df[col].dtype)
            except Exception as e:
                logger.warning(f"Failed to get dtypes: {e}")
                dtypes_dict = {}

            analysis = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': dtypes_dict,
                'missing_values': df.isnull().sum().to_dict(),
                'basic_stats': {}
            }

            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis['basic_stats'] = df[numeric_cols].describe().to_dict()

            # Sample data
            analysis['sample'] = df.head(5).to_dict('records')

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze DataFrame {name}: {e}")
            return {'error': str(e)}

    def create_plot(self, code: str, context: Optional[Dict[str, Any]] = None,
                   output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute plotting code and save the result

        Args:
            code: Python plotting code
            context: Context variables
            output_path: Path to save the plot

        Returns:
            Dictionary with plot creation results
        """
        if context is None:
            context = {}

        # Ensure output directory exists
        if output_path:
            ensure_directory(str(Path(output_path).parent))
        else:
            output_path = str(config.output_dir / "plot.png")

        try:
            # Add plt to context for saving
            plot_context = context.copy()
            plot_context['plt'] = plt

            # Execute the plotting code
            result = self.execute_code(code, plot_context)

            if result['success']:
                # Save the plot
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close('all')  # Clean up

                result['plot_path'] = output_path
                logger.info(f"Plot saved to {output_path}")
            else:
                plt.close('all')  # Clean up on error

            return result

        except Exception as e:
            plt.close('all')  # Clean up
            logger.error(f"Failed to create plot: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def query_dataframe(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Execute pandas query on DataFrame

        Args:
            df: DataFrame to query
            query: Pandas query string

        Returns:
            Dictionary with query results
        """
        try:
            # Execute the query
            result_df = df.query(query)

            return {
                'success': True,
                'result_shape': result_df.shape,
                'result_sample': result_df.head(10).to_dict('records'),
                'query': query
            }

        except Exception as e:
            logger.error(f"Failed to execute query '{query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }

    def compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame,
                          key_columns: List[str]) -> Dict[str, Any]:
        """
        Compare two DataFrames

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            key_columns: Columns to use as keys for comparison

        Returns:
            Dictionary with comparison results
        """
        try:
            # Basic comparison
            comparison = {
                'df1_shape': df1.shape,
                'df2_shape': df2.shape,
                'df1_columns': list(df1.columns),
                'df2_columns': list(df2.columns),
                'common_columns': list(set(df1.columns) & set(df2.columns)),
                'unique_to_df1': list(set(df1.columns) - set(df2.columns)),
                'unique_to_df2': list(set(df2.columns) - set(df1.columns))
            }

            # If key columns exist in both, do detailed comparison
            if all(col in df1.columns and col in df2.columns for col in key_columns):
                # Merge for comparison
                merged = pd.merge(df1, df2, on=key_columns, how='outer', suffixes=('_df1', '_df2'), indicator=True)

                comparison['merge_results'] = {
                    'total_rows': len(merged),
                    'only_in_df1': len(merged[merged['_merge'] == 'left_only']),
                    'only_in_df2': len(merged[merged['_merge'] == 'right_only']),
                    'in_both': len(merged[merged['_merge'] == 'both'])
                }

                # Sample differences
                differences = merged[merged['_merge'] != 'both']
                if len(differences) > 0:
                    comparison['sample_differences'] = differences.head(5).to_dict('records')

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare DataFrames: {e}")
            return {
                'error': str(e)
            }

    def _create_execution_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe execution environment"""
        # Base modules
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'Path': Path,
        }

        # Add context variables
        exec_globals.update(context)

        return exec_globals

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Basic validation of Python code

        Args:
            code: Code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
