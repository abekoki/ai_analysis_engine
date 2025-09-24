#!/usr/bin/env python3
"""使用例"""

import os
from ai_analysis_engine import AIAnalysisEngine

# APIキー設定
os.environ["OPENAI_API_KEY"] = "your-api-key"

# エンジン初期化
engine = AIAnalysisEngine()
engine.initialize()

# 分析実行
result = engine.analyze(
    algorithm_output="data/algorithm_output.csv",
    core_output="data/core_output.csv",
    algorithm_spec="docs/algorithm_spec.md",
    expected_result="フレーム100-200の間に検知結果が存在すること"
)

print(f"分析結果: {'成功' if result.success else '失敗'}")
