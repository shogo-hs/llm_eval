
from evaluator.base import BaseMetric

class JaccardSimilarity(BaseMetric):
    """
    単語の集合ベースのJaccard類似度評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="jaccard_similarity")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        Jaccard類似度で評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
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
