# AI���̓G���W�����C�u����

�ėpAI���̓G���W�� - ���n��f�[�^���͂̎������v���b�g�t�H�[��

## �C���X�g�[��

```bash
pip install -e .
```

## �g�p���@

```python
from ai_analysis_engine import AIAnalysisEngine

engine = AIAnalysisEngine()
engine.initialize()

result = engine.analyze(
    algorithm_output="data/algo.csv",
    core_output="data/core.csv",
    algorithm_spec="docs/spec.md",
    expected_result="���҂���铮��"
)
```
