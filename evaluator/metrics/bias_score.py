"""
バイアススコアによる評価指標
"""
from ..base import BaseMetric

class BiasScoreMetric(BaseMetric):
    """
    バイアススコアによる評価指標（JBBQのバイアス評価指標）
    """
    
    def __init__(self, score_type: str = "avg"):
        """
        初期化メソッド
        
        Args:
            score_type: スコアタイプ（"dis", "amb", "avg"）
        """
        score_name = f"bias_score_{score_type}"
        super().__init__(name=score_name)
        self.score_type = score_type
    
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