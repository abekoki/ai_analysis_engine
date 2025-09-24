# AI分析エンジンライブラリ

汎用AI分析エンジン - 時系列データ分析の自動化プラットフォーム

## インストール

```bash
pip install -e .
```

## 使用方法

```python
from ai_analysis_engine import AIAnalysisEngine

engine = AIAnalysisEngine()
engine.initialize()

result = engine.analyze(
    algorithm_output="data/algo.csv",
    core_output="data/core.csv",
    algorithm_spec="docs/spec.md",
    expected_result="期待される動作"
)
```
