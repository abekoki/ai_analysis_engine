#!/usr/bin/env python3
"""�g�p��"""

import os
from ai_analysis_engine import AIAnalysisEngine

# API�L�[�ݒ�
os.environ["OPENAI_API_KEY"] = "your-api-key"

# �G���W��������
engine = AIAnalysisEngine()
engine.initialize()

# ���͎��s
result = engine.analyze(
    algorithm_output="data/algorithm_output.csv",
    core_output="data/core_output.csv",
    algorithm_spec="docs/algorithm_spec.md",
    expected_result="�t���[��100-200�̊ԂɌ��m���ʂ����݂��邱��"
)

print(f"���͌���: {'����' if result.success else '���s'}")
