"""
データセットの具体実装モジュール
"""
from typing import Dict, List, Any, Optional, Union
import json
import os
import numpy as np
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


class JBBQDataset(BaseDataset):
    """
    JBBQデータセットの実装
    
    Japanese Bias Benchmark for Question Answering
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
        self._categories = ["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Sexual_orientation"]
    
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
    
    @property
    def categories(self) -> List[str]:
        """
        JBBQカテゴリのリストを取得する
        
        Returns:
            List[str]: カテゴリのリスト
        """
        return self._categories

    def get_few_shot_messages_by_category(self, category: str, num_few_shots: int = 0) -> List[Dict[str, str]]:
        """
        カテゴリ別のFew-shotサンプルを取得する
        
        Args:
            category: カテゴリ名
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
        
        # 訓練データの読み込み
        with open(target_few_shot_path, "r", encoding="utf-8") as f:
            few_shot_data = json.load(f)
        
        # カテゴリ別にサンプルをフィルタリング
        samples = few_shot_data.get("samples", [])
        category_samples = [s for s in samples if s.get("category", "") == category]
        
        if not category_samples:
            return []
        
        # 特定のインデックスのサンプルを選択
        selected_indices = []
        if num_few_shots == 2:
            selected_indices = [0, 9]  # llm-leaderboardと同様のindexを使用
        elif num_few_shots == 4:
            selected_indices = [0, 3, 9, 10]  # llm-leaderboardと同様のindexを使用
        else:
            selected_indices = list(range(min(num_few_shots, len(category_samples))))
        
        # 選択したインデックスのサンプルを取得
        few_shot_messages = []
        for idx in selected_indices:
            if idx < len(category_samples):
                few_shot_messages.append({"role": "user", "content": category_samples[idx]["input"]})
                few_shot_messages.append({"role": "assistant", "content": category_samples[idx]["output"]})
        
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
        # サンプル入力からカテゴリを抽出
        sample_dict = {}
        for s in self.get_samples():
            if s["input"] == sample_input:
                sample_dict = s
                break
        
        category = sample_dict.get("category", "")
        
        messages = []
        
        # カテゴリに基づいてFew-shot サンプルを追加
        if category:
            few_shot_messages = self.get_few_shot_messages_by_category(category, num_few_shots)
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
    
    def _load_data(self):
        """
        データセットを読み込む
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        # JBBQデータセットの場合、メトリクスが設定されていなければデフォルト値を設定
        self._instruction = self._data.get("instruction", "")
        if not self._data.get("metrics"):
            self._data["metrics"] = ["exact_match"]
        self._metrics = self._data.get("metrics", [])


class DatasetFactory:
    """
    データセットファクトリー
    
    データセットタイプからデータセットインスタンスを生成する
    """
    
    _dataset_map = {
        "jaster": JasterDataset,
        "jbbq": JBBQDataset
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
            "jaster": [],
            "jbbq": []
        }
        
        # データセットを検索
        for json_file in base_path.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 共通の形式チェック
                if "instruction" in data and "samples" in data and "metrics" in data:
                    # ファイル名やデータ内容からデータセットタイプを判定
                    file_name = json_file.name.lower()
                    if "jbbq" in file_name or any(sample.get("category", "") in ["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Sexual_orientation"] for sample in data.get("samples", [])):
                        result["jbbq"].append(str(json_file))
                    else:
                        result["jaster"].append(str(json_file))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # JSONでない場合やエンコーディングエラーの場合はスキップ
                continue
        
        return result