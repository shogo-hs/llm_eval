"""
カスタム評価指標登録のテストスクリプト
"""
import os
import sys
from pathlib import Path

# パッケージのルートディレクトリをPYTHONPATHに追加
package_root = Path(__file__).parent.absolute()
sys.path.append(str(package_root))

from evaluator.metrics_factory import MetricFactory
from evaluator.metrics_loader import get_metrics_loader

# カスタム評価指標のコード
JACCARD_METRIC_CODE = """
from evaluator.base import BaseMetric

class JaccardSimilarity(BaseMetric):
    \"\"\"
    単語の集合ベースのJaccard類似度評価指標
    \"\"\"
    
    def __init__(self):
        \"\"\"
        初期化メソッド
        \"\"\"
        super().__init__(name="jaccard_similarity")
    
    def calculate(self, predicted: str, reference: str) -> float:
        \"\"\"
        Jaccard類似度で評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        \"\"\"
        # 単語の集合に変換
        pred_words = set(predicted.strip().lower().split())
        ref_words = set(reference.strip().lower().split())
        
        if not pred_words and not ref_words:
            return 1.0
        
        if not pred_words or not ref_words:
            return 0.0
        
        # 共通部分のサイズ / 和集合のサイズ
        intersection = len(pred_words.intersection(ref_words))
        union = len(pred_words.union(ref_words))
        
        return intersection / union
"""

def main():
    """
    メイン関数
    """
    print("=== カスタム評価指標登録のテスト ===")
    
    # 登録前の評価指標リスト
    print("\n1. 登録前の評価指標一覧:")
    metrics_before = MetricFactory.list_metrics()
    for i, metric_name in enumerate(metrics_before):
        print(f"  {i+1}. {metric_name}")
    
    # カスタム評価指標の登録
    print("\n2. カスタム評価指標の登録:")
    try:
        success = MetricFactory.add_custom_metric(
            name="jaccard_similarity",
            code=JACCARD_METRIC_CODE,
            description="単語の集合ベースのJaccard類似度評価指標",
            created_by="test_user",
            save_to_file=True
        )
        
        if success:
            print("  カスタム評価指標の登録成功")
        else:
            print("  カスタム評価指標の登録失敗")
    except Exception as e:
        print(f"  カスタム評価指標の登録エラー: {e}")
    
    # 登録後の評価指標リスト
    print("\n3. 登録後の評価指標一覧:")
    metrics_after = MetricFactory.list_metrics()
    for i, metric_name in enumerate(metrics_after):
        print(f"  {i+1}. {metric_name}")
    
    # 新しい評価指標のテスト
    print("\n4. 新しい評価指標のテスト:")
    if "jaccard_similarity" in metrics_after:
        try:
            metric = MetricFactory.create("jaccard_similarity")
            
            # テストケース
            test_cases = [
                {"pred": "これは テスト です", "ref": "これは テスト です"},
                {"pred": "これは テスト です", "ref": "これは 異なる テスト です"},
                {"pred": "全く 異なる 文", "ref": "まったく 違う 文章 です"},
                {"pred": "", "ref": "テスト"},
            ]
            
            for i, case in enumerate(test_cases):
                score = metric.calculate(case["pred"], case["ref"])
                print(f"  テスト{i+1}: \"{case['pred']}\" vs \"{case['ref']}\" = {score:.4f}")
        except Exception as e:
            print(f"  評価指標のテスト中にエラーが発生: {e}")
    else:
        print("  jaccard_similarity 評価指標が見つかりません")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()
