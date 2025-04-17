
from evaluator.base import BaseMetric
import math

class WordCountSimilarity(BaseMetric):
    """単語数の類似度に基づく評価指標"""
    
    def __init__(self):
        """初期化メソッド"""
        super().__init__(name="word_count_similarity")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        単語数の類似度で評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
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
