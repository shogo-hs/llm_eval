"""
データセットの具体実装モジュール
"""
from typing import Dict, List, Any, Optional, Union
import json
import os
from pathlib import Path

from .base import BaseDataset


class JasterDataset(BaseDataset):
    """
    Jasterデータセットの実装
    
    Jasterベンチマークで定義されたJSON形式のデータセットを読み込み、評価に使用する
    """
    
    def __init__(self, name: str, data_path: Union[str, Path]):
        """
        初期化メソッド
        
        Args:
            name: データセット名
            data_path: データセットファイルパス
        """
        super().__init__(name, data_path)
        self._samples = None
    
    def get_samples(self) -> List[Dict[str, str]]:
        """
        評価用サンプルを取得する
        
        Returns:
            List[Dict[str, str]]: 評価用サンプル
        """
        if self._samples is None:
            self._samples = self.data.get("samples", [])
        return self._samples
    
    @property
    def output_length(self) -> Optional[int]:
        """
        期待される出力の長さを取得する
        
        Returns:
            Optional[int]: 期待される出力の長さ（定義されていない場合はNone）
        """
        return self.data.get("output_length", None)
    
    def get_prompt(self, sample_input: str, few_shot_count: int = 0) -> str:
        """
        プロンプトを生成する
        
        Args:
            sample_input: サンプルの入力
            few_shot_count: 使用するFew-shotサンプル数
            
        Returns:
            str: 生成されたプロンプト
        """
        prompt = self.instruction
        
        # Few-shot サンプルを追加
        if few_shot_count > 0 and few_shot_count <= len(self.few_shots):
            shots = self.few_shots[:few_shot_count]
            for shot in shots:
                prompt += f"\n\n{shot['input']}\n{shot['output']}"
        
        # 評価対象の入力を追加
        prompt += f"\n\n{sample_input}"
        return prompt


class DatasetFactory:
    """
    データセットファクトリー
    
    データセットタイプからデータセットインスタンスを生成する
    """
    
    _dataset_map = {
        "jaster": JasterDataset
    }
    
    @classmethod
    def create(cls, dataset_type: str, name: str, data_path: Union[str, Path]) -> BaseDataset:
        """
        データセットインスタンスを生成する
        
        Args:
            dataset_type: データセットタイプ
            name: データセット名
            data_path: データセットファイルパス
            
        Returns:
            BaseDataset: データセットインスタンス
        
        Raises:
            ValueError: 未サポートのデータセットタイプが指定された場合
        """
        if dataset_type not in cls._dataset_map:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        dataset_class = cls._dataset_map[dataset_type]
        return dataset_class(name, data_path)
    
    @classmethod
    def register(cls, dataset_type: str, dataset_class: type):
        """
        データセットクラスを登録する
        
        Args:
            dataset_type: データセットタイプ
            dataset_class: データセットクラス
        """
        cls._dataset_map[dataset_type] = dataset_class

    @classmethod
    def discover_datasets(cls, base_dir: Union[str, Path]) -> Dict[str, List[str]]:
        """
        指定ディレクトリからデータセットを検索する
        
        Args:
            base_dir: 検索基準ディレクトリ
            
        Returns:
            Dict[str, List[str]]: データセットタイプごとのデータセットファイルパスのリスト
        """
        base_path = Path(base_dir) if isinstance(base_dir, str) else base_dir
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {base_path}")
        
        result = {
            "jaster": []
        }
        
        # Jasterデータセットを検索
        for json_file in base_path.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Jasterデータセットの形式をチェック
                if "instruction" in data and "samples" in data and "metrics" in data:
                    result["jaster"].append(str(json_file))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # JSONでない場合やエンコーディングエラーの場合はスキップ
                continue
        
        return result
