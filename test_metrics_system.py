"""
新しい評価指標管理システムのテストスクリプト
"""
import os
import sys
import asyncio
from pathlib import Path

# パッケージのルートディレクトリをPYTHONPATHに追加
package_root = Path(__file__).parent.absolute()
sys.path.append(str(package_root))

from evaluator.metrics_factory import MetricFactory
from evaluator.metrics_loader import get_metrics_loader
from evaluator.metrics_db import get_metrics_db

# カスタム評価指標のテストコード
CUSTOM_METRIC_CODE = """
from evaluator.base import BaseMetric
import math

class WordCountSimilarity(BaseMetric):
    \"\"\"単語数の類似度に基づく評価指標\"\"\"
    
    def __init__(self):
        \"\"\"初期化メソッド\"\"\"
        super().__init__(name="word_count_similarity")
    
    def calculate(self, predicted: str, reference: str) -> float:
        \"\"\"
        単語数の類似度で評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        \"\"\"
        # 単語数をカウント
        pred_count = len(predicted.strip().split())
        ref_count = len(reference.strip().split())
        
        # 単語数の差を計算し、類似度に変換
        if pred_count == 0 and ref_count == 0:
            return 1.0
        
        if pred_count == 0 or ref_count == 0:
            return 0.0
        
        # 大きい方で割って、その差を1から引く
        similarity = 1.0 - abs(pred_count - ref_count) / max(pred_count, ref_count)
        return similarity
"""


async def test_metrics_system():
    """
    評価指標管理システムのテスト
    """
    print("=== 評価指標管理システムのテスト ===")
    
    # 利用可能な評価指標を表示
    print("\n1. 利用可能な評価指標:")
    metrics = MetricFactory.list_metrics()
    for i, metric_name in enumerate(metrics):
        print(f"  {i+1}. {metric_name}")
    
    # 評価指標のインスタンス化テスト
    print("\n2. 評価指標のインスタンス化テスト:")
    test_metrics = ["exact_match", "char_f1", "set_f1"]
    for metric_name in test_metrics:
        try:
            metric = MetricFactory.create(metric_name)
            print(f"  {metric_name}: インスタンス化成功")
        except Exception as e:
            print(f"  {metric_name}: インスタンス化失敗 - {e}")
    
    # 評価計算テスト
    print("\n3. 評価計算テスト:")
    test_cases = [
        {"metric": "exact_match", "pred": "Hello World", "ref": "Hello World"},
        {"metric": "exact_match", "pred": "Hello World", "ref": "Hello world"},
        {"metric": "char_f1", "pred": "Hello World", "ref": "Hello world"},
        {"metric": "set_f1", "pred": "A\nB\nC", "ref": "A\nC\nD"},
    ]
    
    for case in test_cases:
        try:
            metric = MetricFactory.create(case["metric"])
            score = metric.calculate(case["pred"], case["ref"])
            print(f"  {case['metric']}: \"{case['pred']}\" vs \"{case['ref']}\" = {score:.4f}")
        except Exception as e:
            print(f"  {case['metric']}: 計算失敗 - {e}")
    
    # カスタム評価指標の登録テスト
    print("\n4. カスタム評価指標の登録テスト:")
    try:
        success = MetricFactory.add_custom_metric(
            name="word_count_similarity",
            code=CUSTOM_METRIC_CODE,
            description="単語数の類似度に基づく評価指標",
            created_by="test_user",
            save_to_file=True
        )
        
        if success:
            print("  カスタム評価指標の登録成功")
            
            # カスタム評価指標のインスタンス化と計算テスト
            metric = MetricFactory.create("word_count_similarity")
            test_cases = [
                {"pred": "Hello World", "ref": "Hello World Test"},
                {"pred": "A B C D", "ref": "A B"},
                {"pred": "", "ref": "Test"},
            ]
            
            for i, case in enumerate(test_cases):
                score = metric.calculate(case["pred"], case["ref"])
                print(f"  テスト{i+1}: \"{case['pred']}\" vs \"{case['ref']}\" = {score:.4f}")
        else:
            print("  カスタム評価指標の登録失敗")
    except Exception as e:
        print(f"  カスタム評価指標の登録エラー: {e}")
    
    # 利用可能な評価指標を再表示
    print("\n5. 更新された評価指標リスト:")
    metrics = MetricFactory.list_metrics()
    for i, metric_name in enumerate(metrics):
        print(f"  {i+1}. {metric_name}")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    # テスト実行
    asyncio.run(test_metrics_system())
