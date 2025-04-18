"""
正答率の差分による評価指標
"""
from ..base import BaseMetric

class AccDiffMetric(BaseMetric):
    """
    正答率の差分による評価指標（JBBQのバイアス評価指標）
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="acc_diff")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（バッチ単位で計算するため常に0を返す）
        """
        # このメトリクスはサンプル単位では意味がなく、バッチ全体で計算する必要がある
        # JBBQEvaluatorの_calculate_bias_metrics内で実際の計算を行う
        return 0.0