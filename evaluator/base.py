"""
評価指標の抽象基底クラスを定義するモジュール
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import json
import os
from pathlib import Path


class BaseMetric(ABC):
    """
    評価指標の抽象基底クラス
    
    全ての評価指標はこのクラスを継承して実装する必要がある
    """
    
    def __init__(self, name: str):
        """
        初期化メソッド
        
        Args:
            name: 評価指標の名前
        """
        self.name = name
    
    @abstractmethod
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name}"


class BaseDataset(ABC):
    """
    データセットの抽象基底クラス
    
    全てのデータセットはこのクラスを継承して実装する必要がある
    """
    
    def __init__(self, name: str, data_path: Union[str, Path], few_shot_path: Optional[Union[str, Path]] = None):
        """
        初期化メソッド
        
        Args:
            name: データセット名
            data_path: データセットファイルパス
            few_shot_path: Few-shotサンプルのファイルパス (Noneの場合は使用しない)
        """
        self.name = name
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.few_shot_path = Path(few_shot_path) if isinstance(few_shot_path, str) and few_shot_path else None
        self._data = None
        self._instruction = None
        self._metrics = None
        self._samples = None
        
    @property
    def data(self) -> Dict[str, Any]:
        """
        データセットの内容を取得する
        
        Returns:
            Dict[str, Any]: データセットの内容
        """
        if self._data is None:
            self._load_data()
        return self._data
    
    @property
    def instruction(self) -> str:
        """
        タスク指示を取得する
        
        Returns:
            str: タスク指示
        """
        if self._instruction is None:
            self._load_data()
        return self._instruction
    
    @property
    def metrics(self) -> List[str]:
        """
        評価指標のリストを取得する
        
        Returns:
            List[str]: 評価指標のリスト
        """
        if self._metrics is None:
            self._load_data()
        return self._metrics
    
    def _load_data(self):
        """
        データセットを読み込む
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        self._instruction = self._data.get("instruction", "")
        self._metrics = self._data.get("metrics", [])
    
    def get_samples(self, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        評価用サンプルを取得する（サンプリング機能付き）
        
        Args:
            max_samples: 最大サンプル数（Noneの場合は全てのサンプルを返す）
            
        Returns:
            List[Dict[str, str]]: 評価用サンプル
        """
        samples = self._get_all_samples()
        if max_samples is not None and max_samples > 0 and max_samples < len(samples):
            return samples[:max_samples]
        return samples
    
    @abstractmethod
    def _get_all_samples(self) -> List[Dict[str, str]]:
        """
        全評価サンプルを取得する
        
        Returns:
            List[Dict[str, str]]: 全評価サンプル
        """
        pass


class BaseLLM(ABC):
    """
    LLMクライアントの抽象基底クラス
    
    全てのLLMクライアントはこのクラスを継承して実装する必要がある
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        初期化メソッド
        
        Args:
            model_name: モデル名
            **kwargs: その他のパラメータ
        """
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を行う
        
        Args:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ
            
        Returns:
            str: 生成されたテキスト
        """
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        バッチ処理でテキスト生成を行う
        
        Args:
            prompts: 入力プロンプトのリスト
            **kwargs: 生成パラメータ
            
        Returns:
            List[str]: 生成されたテキストのリスト
        """
        pass


class BaseEvaluator(ABC):
    """
    評価ツールの抽象基底クラス
    
    全ての評価ツールはこのクラスを継承して実装する必要がある
    """
    
    def __init__(self, 
                 dataset: BaseDataset, 
                 llm: BaseLLM, 
                 metrics: Optional[List[BaseMetric]] = None):
        """
        初期化メソッド
        
        Args:
            dataset: 評価用データセット
            llm: 評価対象のLLM
            metrics: 評価指標のリスト（Noneの場合はデータセットの定義に従う）
        """
        self.dataset = dataset
        self.llm = llm
        self._metrics = metrics
    
    @property
    def metrics(self) -> List[BaseMetric]:
        """
        評価指標のリストを取得する
        
        Returns:
            List[BaseMetric]: 評価指標のリスト
        """
        return self._metrics
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        評価を実行する
        
        Args:
            **kwargs: 評価パラメータ
            
        Returns:
            Dict[str, Any]: 評価結果
        """
        pass
    
    @abstractmethod
    def save_results(self, results: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        評価結果を保存する
        
        Args:
            results: 評価結果
            output_path: 出力先パス
        """
        pass