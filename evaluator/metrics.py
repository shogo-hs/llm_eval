"""
評価指標の具体実装モジュール
"""
import re
from typing import Optional, List, Dict, Any, Union
from abc import ABC
import math
import numpy as np
from fuzzywuzzy import fuzz
import difflib

from .base import BaseMetric


class ExactMatch(BaseMetric):
    """
    完全一致評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="exact_match")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        完全一致で評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（一致: 1.0, 不一致: 0.0）
        """
        return float(predicted.strip() == reference.strip())


class ExactMatchFigure(BaseMetric):
    """
    数値の完全一致評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="exact_match_figure")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        数値として解釈した場合の完全一致で評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（一致: 1.0, 不一致: 0.0）
        """
        try:
            pred_value = float(predicted.strip())
            ref_value = float(reference.strip())
            return float(pred_value == ref_value)
        except ValueError:
            return 0.0


class CharF1(BaseMetric):
    """
    文字ベースのF1スコア評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="char_f1")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        文字ベースのF1スコアで評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        return fuzz.token_sort_ratio(predicted.strip(), reference.strip()) / 100.0


class SetF1(BaseMetric):
    """
    集合ベースのF1スコア評価指標
    """
    
    def __init__(self, delimiter: str = "\n"):
        """
        初期化メソッド
        
        Args:
            delimiter: 項目の区切り文字
        """
        super().__init__(name="set_f1")
        self.delimiter = delimiter
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        集合ベースのF1スコアで評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        set_pred = {x.strip() for x in predicted.split(self.delimiter) if x.strip()}
        set_ref = {x.strip() for x in reference.split(self.delimiter) if x.strip()}
        
        if not set_pred and not set_ref:
            return 1.0
        if not set_pred or not set_ref:
            return 0.0
        
        true_positives = len(set_pred.intersection(set_ref))
        precision = true_positives / len(set_pred)
        recall = true_positives / len(set_ref)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


class BLEUScore(BaseMetric):
    """
    BLEUスコア評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="bleu")
        try:
            from sacrebleu import BLEU
            self.bleu = BLEU()
        except ImportError:
            raise ImportError("sacrebleu is required for BLEUScore metric")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        BLEUスコアで評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        return self.bleu.sentence_score(predicted.strip(), [reference.strip()]).score / 100.0


class ContainsAnswer(BaseMetric):
    """
    回答を含むかどうかの評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="contains_answer")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        予測文字列が正解を含んでいるかどうかで評価する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（含む: 1.0, 含まない: 0.0）
        """
        return float(reference.strip() in predicted.strip())


class MetricFactory:
    """
    評価指標ファクトリー
    
    指標名から評価指標インスタンスを生成する
    """
    
    _metric_map = {
        "exact_match": ExactMatch,
        "exact_match_figure": ExactMatchFigure,
        "char_f1": CharF1,
        "set_f1": SetF1,
        "bleu": BLEUScore,
        "contains_answer": ContainsAnswer
    }
    
    @classmethod
    def create(cls, metric_name: str, **kwargs) -> BaseMetric:
        """
        評価指標インスタンスを生成する
        
        Args:
            metric_name: 評価指標名
            **kwargs: 評価指標の初期化パラメータ
            
        Returns:
            BaseMetric: 評価指標インスタンス
        
        Raises:
            ValueError: 未サポートの評価指標名が指定された場合
        """
        if metric_name not in cls._metric_map:
            raise ValueError(f"Unsupported metric: {metric_name}")
        
        metric_class = cls._metric_map[metric_name]
        return metric_class(**kwargs)
    
    @classmethod
    def create_from_list(cls, metric_names: List[str], **kwargs) -> List[BaseMetric]:
        """
        評価指標インスタンスのリストを生成する
        
        Args:
            metric_names: 評価指標名のリスト
            **kwargs: 評価指標の初期化パラメータ
            
        Returns:
            List[BaseMetric]: 評価指標インスタンスのリスト
        """
        return [cls.create(name, **kwargs) for name in metric_names]
    
    @classmethod
    def register(cls, metric_name: str, metric_class: type):
        """
        評価指標クラスを登録する
        
        Args:
            metric_name: 評価指標名
            metric_class: 評価指標クラス
        """
        cls._metric_map[metric_name] = metric_class
