from ai_analysis_engine.performance_analyzer.performance_analyzer import PerformanceAnalyzer
from ai_analysis_engine.config.settings import Settings
import pandas as pd


def main():
    settings = Settings()
    pa = PerformanceAnalyzer(settings)

    # フレームレベル風のテストデータ
    df = pd.DataFrame({
        'frame_num': list(range(1, 121)),
        'left_eye_open': [0.9]*60 + [0.1]*60,
        'right_eye_open': [0.9]*60 + [0.1]*60,
        'face_confidence': [0.95]*120,
        'is_drowsy': [0]*60 + [1]*60,
    })
    # 期待値: 後半が居眠り
    df['expected_is_drowsy'] = [0]*60 + [1]*60
    # ROC/PR用スコア（単純にis_drowsyと同傾向）
    df['score'] = [0.1]*60 + [0.9]*60

    result = pa.analyze_performance({'data': df, 'metadata': {}})
    print({
        'summary': result['summary'],
        'visualizations': result['visualizations']
    })


if __name__ == '__main__':
    main()

