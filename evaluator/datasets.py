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
    
    def __init__(self, name: str, data_path: Union[str, Path], few_shot_path: Optional[Union[str, Path]] = None):
        """
        初期化メソッド
        
        Args:
            name: データセット名
            data_path: データセットファイルパス
            few_shot_path: Few-shotサンプルのファイルパス (Noneの場合は使用しない)
        """
        super().__init__(name, data_path, few_shot_path)
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
    
    def get_few_shot_messages(self, num_few_shots: int = 0) -> List[Dict[str, str]]:
        """
        Few-shotサンプルを取得する
        
        Args:
            num_few_shots: 使用するFew-shotサンプル数
            
        Returns:
            List[Dict[str, str]]: Few-shotサンプルのリスト (role, contentのペア)
        """
        if num_few_shots <= 0 or self.few_shot_path is None or not self.few_shot_path.exists():
            return []
        
        # few_shot_pathが親ディレクトリの場合は、同じファイル名のtrainデータを参照
        target_few_shot_path = self.few_shot_path
        if target_few_shot_path.is_dir():
            target_few_shot_path = target_few_shot_path / "train" / self.data_path.name
        
        if not target_few_shot_path.exists():
            print(f"Few-shot path not found: {target_few_shot_path}")
            return []
        
        with open(target_few_shot_path, "r", encoding="utf-8") as f:
            few_shot_data = json.load(f)
        
        samples = few_shot_data.get("samples", [])
        if not samples:
            return []
        
        few_shot_messages = []
        for i in range(min(num_few_shots, len(samples))):
            few_shot_messages.append({"role": "user", "content": samples[i]["input"]})
            few_shot_messages.append({"role": "assistant", "content": samples[i]["output"]})
        
        return few_shot_messages
    
    def get_prompt(self, sample_input: str, num_few_shots: int = 0) -> str:
        """
        プロンプトを生成する
        
        Args:
            sample_input: サンプルの入力
            num_few_shots: 使用するFew-shotサンプル数
            
        Returns:
            str: 生成されたプロンプト
        """
        messages = []
        
        # Few-shot サンプルを追加
        few_shot_messages = self.get_few_shot_messages(num_few_shots)
        if few_shot_messages:
            messages.extend(few_shot_messages)
        
        # 評価対象の入力を追加
        messages.append({"role": "user", "content": sample_input})
        
        # 最初のメッセージに指示を追加
        if messages and self.instruction:
            first_content = messages[0]["content"]
            messages[0]["content"] = f"{self.instruction}\n\n{first_content}"
        
        # メッセージをプロンプト形式に変換
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"\n\nユーザー: {msg['content']}"
            else:
                prompt += f"\n\nアシスタント: {msg['content']}"
        
        return prompt.strip()


class DatasetFactory:
    """
    データセットファクトリー
    
    データセットタイプからデータセットインスタンスを生成する
    """
    
    _dataset_map = {
        "jaster": JasterDataset
    }
    
    @classmethod
    def create(cls, dataset_type: str, name: str, data_path: Union[str, Path], few_shot_path: Optional[Union[str, Path]] = None) -> BaseDataset:
        """
        データセットインスタンスを生成する
        
        Args:
            dataset_type: データセットタイプ
            name: データセット名
            data_path: データセットファイルパス
            few_shot_path: Few-shotサンプルのファイルパス (Noneの場合は使用しない)
            
        Returns:
            BaseDataset: データセットインスタンス
        
        Raises:
            ValueError: 未サポートのデータセットタイプが指定された場合
        """
        if dataset_type not in cls._dataset_map:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        dataset_class = cls._dataset_map[dataset_type]
        return dataset_class(name, data_path, few_shot_path)
    
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