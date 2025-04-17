"""
ROUGE-L評価指標モジュール (カスタム実装)
"""
from ...base import BaseMetric
import numpy as np


class RougeL(BaseMetric):
    """
    ROUGE-L評価指標
    
    最長共通部分列に基づく評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="rouge_l")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        ROUGE-Lスコアで評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        if not predicted or not reference:
            return 0.0
        
        # 文を単語に分割
        pred_words = predicted.strip().lower().split()
        ref_words = reference.strip().lower().split()
        
        if not pred_words or not ref_words:
            return 0.0
        
        # 最長共通部分列の長さを計算
        lcs_length = self._lcs_length(pred_words, ref_words)
        
        # 適合率、再現率、F値を計算
        precision = lcs_length / len(pred_words) if pred_words else 0.0
        recall = lcs_length / len(ref_words) if ref_words else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        # F値を返す
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def _lcs_length(self, s1: list, s2: list) -> int:
        """
        2つの単語リストの最長共通部分列の長さを計算する
        
        Args:
            s1: 1つ目の単語リスト
            s2: 2つ目の単語リスト
            
        Returns:
            int: 最長共通部分列の長さ
        """
        # 動的計画法で計算
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
