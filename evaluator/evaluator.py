"""
評価ツールの具体実装モジュール
"""
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from .base import BaseEvaluator, BaseDataset, BaseLLM, BaseMetric
from .metrics_factory import MetricFactory


class JasterEvaluator(BaseEvaluator):
    """
    Jaster評価ツールの実装
    
    Jasterベンチマークのタスクを評価する
    """
    
    def __init__(self, 
                 dataset: BaseDataset, 
                 llm: BaseLLM, 
                 metrics: Optional[List[BaseMetric]] = None,
                 few_shot_count: int = 0,
                 batch_size: int = 5):
        """
        初期化メソッド
        
        Args:
            dataset: 評価用データセット
            llm: 評価対象のLLM
            metrics: 評価指標のリスト（Noneの場合はデータセットの定義に従う）
            few_shot_count: Few-shotサンプル数
            batch_size: バッチサイズ
        """
        super().__init__(dataset, llm, metrics)
        self.few_shot_count = few_shot_count
        self.batch_size = batch_size
        
        # 評価指標がNoneの場合は、データセットの定義に従う
        if self._metrics is None:
            self._metrics = MetricFactory.create_from_list(dataset.metrics)
    
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        評価を実行する
        
        Args:
            **kwargs: 評価パラメータ
            
        Returns:
            Dict[str, Any]: 評価結果
        """
        # 評価開始時刻
        start_time = time.time()
        
        # サンプルを取得
        samples = self.dataset.get_samples()
        
        # プロンプトを生成
        prompts = []
        for sample in samples:
            prompt = self.dataset.get_prompt(sample["input"], self.few_shot_count)
            prompts.append(prompt)
        
        # モデルの予測を実行
        predictions = await self.llm.generate_batch(
            prompts, batch_size=self.batch_size, **kwargs
        )
        
        # 評価結果を計算
        metric_results = {}
        for metric in self.metrics:
            scores = []
            for i, sample in enumerate(samples):
                prediction = predictions[i]
                reference = sample["output"]
                score = metric.calculate(prediction, reference)
                scores.append(score)
            
            # 平均スコアを計算
            metric_results[metric.name] = {
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores)
            }
        
        # 評価終了時刻
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 結果をまとめる
        results = {
            "dataset": self.dataset.name,
            "model": self.llm.model_name,
            "num_samples": len(samples),
            "few_shot_count": self.few_shot_count,
            "metrics": {name: results["mean"] for name, results in metric_results.items()},
            "detailed_metrics": metric_results,
            "samples": [
                {
                    "input": sample["input"],
                    "reference": sample["output"],
                    "prediction": predictions[i],
                    "scores": {name: results["scores"][i] for name, results in metric_results.items()}
                }
                for i, sample in enumerate(samples)
            ],
            "elapsed_time": elapsed_time
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        評価結果を保存する
        
        Args:
            results: 評価結果
            output_path: 出力先パス
        """
        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSVファイルも保存
        csv_path = output_path.with_suffix(".csv")
        df = pd.DataFrame([
            {
                "dataset": results["dataset"],
                "model": results["model"],
                "num_samples": results["num_samples"],
                "few_shot_count": results["few_shot_count"],
                **results["metrics"],
                "elapsed_time": results["elapsed_time"]
            }
        ])
        df.to_csv(csv_path, index=False)


class EvaluatorFactory:
    """
    評価ツールファクトリー
    
    評価ツールタイプから評価ツールインスタンスを生成する
    """
    
    _evaluator_map = {
        "jaster": JasterEvaluator
    }
    
    @classmethod
    def create(cls, 
               evaluator_type: str, 
               dataset: BaseDataset, 
               llm: BaseLLM, 
               **kwargs) -> BaseEvaluator:
        """
        評価ツールインスタンスを生成する
        
        Args:
            evaluator_type: 評価ツールタイプ
            dataset: 評価用データセット
            llm: 評価対象のLLM
            **kwargs: 評価ツールの初期化パラメータ
            
        Returns:
            BaseEvaluator: 評価ツールインスタンス
        
        Raises:
            ValueError: 未サポートの評価ツールタイプが指定された場合
        """
        if evaluator_type not in cls._evaluator_map:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
        
        evaluator_class = cls._evaluator_map[evaluator_type]
        return evaluator_class(dataset, llm, **kwargs)
    
    @classmethod
    def register(cls, evaluator_type: str, evaluator_class: type):
        """
        評価ツールクラスを登録する
        
        Args:
            evaluator_type: 評価ツールタイプ
            evaluator_class: 評価ツールクラス
        """
        cls._evaluator_map[evaluator_type] = evaluator_class
