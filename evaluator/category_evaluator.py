"""
カテゴリ評価マネージャー

大分類カテゴリの評価実行と結果集約を行うモジュール
"""
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np

from .base import BaseLLM
from .category_mapping import CategoryMapping
from .datasets import DatasetFactory
from .evaluator import EvaluatorFactory
from .llm import LLMFactory


class CategoryEvaluator:
    """
    カテゴリ評価マネージャー
    
    大分類カテゴリの評価実行と結果集約を行うクラス
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        dataset_base_dir: Optional[Path] = None,
        few_shot_base_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        batch_size: int = 5,
        api_url: str = "http://192.168.3.43:8000/v1/chat/completions",
        temperature: float = 0.7,
        max_tokens: int = 500,
        reuse_existing_results: bool = True,
        auto_adjust_samples: bool = True
    ):
        """
        初期化メソッド
        
        Args:
            model_name: モデル名
            output_dir: 出力ディレクトリ
            dataset_base_dir: データセットベースディレクトリ (Noneの場合はデフォルト)
            few_shot_base_dir: Few-shotサンプルベースディレクトリ (Noneの場合はデフォルト)
            max_samples: サンプリング数（Noneの場合は全サンプルを使用）
            batch_size: バッチサイズ
            api_url: ローカルLLM API URL
            temperature: 生成の温度
            max_tokens: 最大生成トークン数
            reuse_existing_results: 既存の評価結果を再利用するかどうか
            auto_adjust_samples: サンプル数を自動調整するかどうか
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.dataset_base_dir = dataset_base_dir
        self.few_shot_base_dir = few_shot_base_dir
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reuse_existing_results = reuse_existing_results
        self.auto_adjust_samples = auto_adjust_samples
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLMクライアントの作成
        self.llm = LLMFactory.create(
            "local",
            model_name,
            api_url=api_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def evaluate_category(self, category: str) -> Dict[str, Any]:
        """
        指定されたカテゴリを評価する
        
        Args:
            category: カテゴリ名（GLP_xxxx、ALT_xxxx、GLP_AVG、ALT_AVG、TOTAL_AVG）
            
        Returns:
            Dict[str, Any]: 評価結果
            
        Raises:
            ValueError: データセットが見つからない場合や評価中に重大なエラーが発生した場合
        """
        print(f"カテゴリ評価開始: {category}")
        start_time = time.time()
        
        # カテゴリに属するデータセット情報を取得
        datasets_info = CategoryMapping.get_datasets_for_category(category)
        
        # 各データセットを評価
        results = {}
        tasks = []
        dataset_paths = []
        
        # データセットパスの存在チェック（前処理）
        missing_datasets = []
        for dataset_info in datasets_info:
            dataset_name = dataset_info["dataset"]
            shot_counts = CategoryMapping.get_shot_counts_for_dataset(dataset_name)
            
            # データセットの存在を確認
            dataset_path = self._resolve_dataset_path(dataset_name)
            if not dataset_path.exists():
                missing_datasets.append((dataset_name, dataset_path))
            else:
                dataset_paths.append(dataset_path)
                
                for shot_count in shot_counts:
                    task = self._evaluate_dataset(
                        dataset_name, 
                        dataset_info["evaluator"], 
                        shot_count,
                        dataset_info["metrics"]
                    )
                    tasks.append(task)
        
        # 不足しているデータセットがある場合、エラーメッセージを生成して処理を中止
        if missing_datasets:
            error_msg = f"評価を中止します: 以下のデータセットファイルが見つかりません:\n"
            for dataset_name, dataset_path in missing_datasets:
                error_msg += f"- {dataset_name}: {dataset_path}\n"
            
            if self.dataset_base_dir:
                error_msg += f"\nカスタムデータセットディレクトリが指定されています: {self.dataset_base_dir}\n"
            
            error_msg += f"\n対応方法:\n"
            error_msg += f"1. --dataset-dirオプションで正しいデータセットディレクトリを指定する\n"
            error_msg += f"2. {Path('data/datasets').resolve()}にデータセットファイルを配置する\n"
            error_msg += f"\n注意: データセットファイルが見つからない場合、評価は実行されず結果も保存されません。"
            raise ValueError(error_msg)
        
        # 全てのデータセットがチェックを通過したら並列で評価を実行
        try:
            dataset_results = await asyncio.gather(*tasks)
            
            # エラー結果の確認
            error_results = [r for r in dataset_results if "error" in r and r["error"]]
            if error_results:
                # エラーメッセージを集約
                error_messages = [f"- {r['dataset']} ({r['few_shot_count']}-shot): {r['error']}" for r in error_results]
                error_msg = "以下のデータセット評価中にエラーが発生したため、カテゴリ評価を中止します:\n"
                error_msg += "\n".join(error_messages)
                raise ValueError(error_msg)
            
            # 結果を集約
            dataset_scores = {}
            for result in dataset_results:
                dataset_name = result["dataset"]
                shot_count = result["few_shot_count"]
                
                # データセット名にショット数を含める
                dataset_key = f"{dataset_name}_{shot_count}shot"
                dataset_scores[dataset_key] = result
            
            # スコアを集計して大分類カテゴリスコアを計算
            category_score = self._calculate_category_score(category, dataset_scores)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 結果をまとめる
            results = {
                "category": category,
                "model": self.model_name,
                "score": category_score,
                "dataset_scores": dataset_scores,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # 結果を保存
            self._save_category_results(results, category)
            
            print(f"カテゴリ評価完了: {category}, スコア: {category_score:.4f}")
            return results
            
        except Exception as e:
            # 評価中のエラーを伝搬
            raise ValueError(f"カテゴリ「{category}」の評価中にエラーが発生しました: {str(e)}")
    
    async def _evaluate_dataset(
        self,
        dataset_name: str,
        evaluator_type: str,
        few_shot_count: int,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        データセットを評価する
        
        Args:
            dataset_name: データセット名
            evaluator_type: 評価ツールタイプ
            few_shot_count: Few-shotサンプル数
            metrics: 評価指標のリスト
            
        Returns:
            Dict[str, Any]: 評価結果
            
        Raises:
            ValueError: データセットファイルが見つからない場合や評価中に重大なエラーが発生した場合
        """
        # 既存の結果ファイルを確認
        if self.reuse_existing_results:
            existing_result = self._find_existing_result(dataset_name, few_shot_count)
            if existing_result:
                print(f"既存の評価結果を使用: {dataset_name} ({few_shot_count}-shot)")
                return existing_result
        
        print(f"データセット評価開始: {dataset_name} ({few_shot_count}-shot)")
        
        # データセットとFew-shotサンプルのパスを解決
        dataset_path = self._resolve_dataset_path(dataset_name)
        
        # データセットファイルの存在確認
        if not dataset_path.exists():
            error_msg = f"Dataset file not found: {dataset_path}"
            print(f"データセット評価エラー: {dataset_name} ({few_shot_count}-shot): {error_msg}")
            raise ValueError(error_msg)
        
        few_shot_path = self._resolve_few_shot_path(dataset_name) if few_shot_count > 0 else None
        
        try:
            # データセットの作成
            dataset = DatasetFactory.create(
                evaluator_type,
                dataset_name,
                dataset_path,
                few_shot_path
            )
            
            # サンプル数を調整
            local_max_samples = self.max_samples
            if self.auto_adjust_samples and self.max_samples is None:
                # ドキュメントに従ったサンプル数に調整
                local_max_samples = CategoryMapping.get_sample_count_for_dataset(
                    dataset_name, 
                    "test"
                )
                if local_max_samples:
                    print(f"サンプル数を自動調整: {dataset_name} -> {local_max_samples}サンプル")
            
            # 評価ツールの作成
            evaluator = EvaluatorFactory.create(
                evaluator_type,
                dataset,
                self.llm,
                few_shot_count=few_shot_count,
                few_shot_path=few_shot_path,
                batch_size=self.batch_size,
                max_samples=local_max_samples
            )
            
            # 評価の実行
            result = await evaluator.evaluate()
            
            # 結果の保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shot_info = f"{few_shot_count}shot"
            output_path = self.output_dir / f"{dataset_name}_{self.model_name}_{shot_info}_{timestamp}.json"
            evaluator.save_results(result, output_path)
            
            print(f"データセット評価完了: {dataset_name} ({few_shot_count}-shot)")
            return result
            
        except Exception as e:
            # 重大なエラーが発生した場合、処理を中止するために例外を再スロー
            error_msg = f"{dataset_name} ({few_shot_count}-shot): {str(e)}"
            print(f"データセット評価エラー: {error_msg}")
            raise ValueError(error_msg)
    
    def _calculate_category_score(
        self,
        category: str,
        dataset_scores: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        大分類カテゴリスコアを計算する
        
        Args:
            category: カテゴリ名
            dataset_scores: データセットごとの評価結果
            
        Returns:
            float: 大分類カテゴリスコア
        """
        # GLP_AVG, ALT_AVG, TOTAL_AVGの場合
        if category in ["GLP_AVG", "ALT_AVG", "TOTAL_AVG"]:
            sub_categories = CategoryMapping.AGGREGATE_CATEGORIES[category]
            sub_category_scores = []
            
            for sub_category in sub_categories:
                # サブカテゴリに属するデータセットを取得
                sub_datasets_info = CategoryMapping.get_datasets_for_category(sub_category)
                sub_dataset_scores = []
                
                for dataset_info in sub_datasets_info:
                    dataset_name = dataset_info["dataset"]
                    metrics = dataset_info["metrics"]
                    
                    # 0-shotと2-shotの結果を取得して平均
                    shot_scores = []
                    shot_counts = CategoryMapping.get_shot_counts_for_dataset(dataset_name)
                    
                    for shot_count in shot_counts:
                        dataset_key = f"{dataset_name}_{shot_count}shot"
                        if dataset_key in dataset_scores and "error" not in dataset_scores[dataset_key]:
                            # 各メトリクスのスコアを取得して平均化
                            metric_scores = []
                            for metric in metrics:
                                if metric in dataset_scores[dataset_key]["metrics"]:
                                    score = dataset_scores[dataset_key]["metrics"][metric]
                                    # スコアを正規化
                                    normalized_score = CategoryMapping.normalize_score(metric, score)
                                    metric_scores.append(normalized_score)
                            
                            if metric_scores:
                                shot_scores.append(np.mean(metric_scores))
                    
                    if shot_scores:
                        sub_dataset_scores.append(np.mean(shot_scores))
                
                if sub_dataset_scores:
                    sub_category_scores.append(np.mean(sub_dataset_scores))
            
            if sub_category_scores:
                return np.mean(sub_category_scores)
            else:
                return 0.0
        
        # 通常の大分類カテゴリの場合
        else:
            dataset_info_list = CategoryMapping.get_datasets_for_category(category)
            dataset_avg_scores = []
            
            for dataset_info in dataset_info_list:
                dataset_name = dataset_info["dataset"]
                metrics = dataset_info["metrics"]
                
                # 0-shotと2-shotの結果を取得して平均
                shot_scores = []
                shot_counts = CategoryMapping.get_shot_counts_for_dataset(dataset_name)
                
                for shot_count in shot_counts:
                    dataset_key = f"{dataset_name}_{shot_count}shot"
                    if dataset_key in dataset_scores and "error" not in dataset_scores[dataset_key]:
                        # 各メトリクスのスコアを取得して平均化
                        metric_scores = []
                        for metric in metrics:
                            if metric in dataset_scores[dataset_key]["metrics"]:
                                score = dataset_scores[dataset_key]["metrics"][metric]
                                # スコアを正規化
                                normalized_score = CategoryMapping.normalize_score(metric, score)
                                metric_scores.append(normalized_score)
                        
                        if metric_scores:
                            shot_scores.append(np.mean(metric_scores))
                
                if shot_scores:
                    dataset_avg_scores.append(np.mean(shot_scores))
            
            if dataset_avg_scores:
                return np.mean(dataset_avg_scores)
            else:
                return 0.0
    
    def _find_existing_result(
        self,
        dataset_name: str,
        few_shot_count: int
    ) -> Optional[Dict[str, Any]]:
        """
        既存の評価結果を検索する
        
        Args:
            dataset_name: データセット名
            few_shot_count: Few-shotサンプル数
            
        Returns:
            Optional[Dict[str, Any]]: 評価結果（見つからない場合はNone）
        """
        shot_info = f"{few_shot_count}shot"
        result_pattern = f"{dataset_name}_{self.model_name}_{shot_info}_*.json"
        
        # 最新の結果ファイルを検索
        result_files = list(self.output_dir.glob(result_pattern))
        if not result_files:
            return None
        
        # ファイル更新日時でソート（最新のものを使用）
        result_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_result_file = result_files[0]
        
        try:
            with open(latest_result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            return result
        except Exception as e:
            print(f"既存の結果ファイル読み込みエラー: {latest_result_file}: {str(e)}")
            return None
    
    def _resolve_dataset_path(self, dataset_name: str) -> Path:
        """
        データセットパスを解決する
        
        Args:
            dataset_name: データセット名
            
        Returns:
            Path: データセットパス
        """
        # データセットベースディレクトリが指定されている場合はそれを使用
        if self.dataset_base_dir:
            # 拡張子の追加（.jsonがない場合）
            if not dataset_name.endswith(".json"):
                dataset_file = f"{dataset_name}.json"
            else:
                dataset_file = dataset_name
            
            return self.dataset_base_dir / dataset_file
        
        # デフォルトのパスを使用
        default_dataset_dir = Path("data/datasets")
        
        # MT-bench系のデータセットの場合は特別な処理
        if dataset_name.startswith("mt-bench-"):
            return default_dataset_dir / "mt-bench" / f"{dataset_name.replace('mt-bench-', '')}.json"
        
        # 通常のデータセット
        if not dataset_name.endswith(".json"):
            dataset_file = f"{dataset_name}.json"
        else:
            dataset_file = dataset_name
        
        return default_dataset_dir / dataset_file
    
    def _resolve_few_shot_path(self, dataset_name: str) -> Optional[Path]:
        """
        Few-shotサンプルパスを解決する
        
        Args:
            dataset_name: データセット名
            
        Returns:
            Optional[Path]: Few-shotサンプルパス（見つからない場合はNone）
        """
        # Few-shotベースディレクトリが指定されている場合はそれを使用
        if self.few_shot_base_dir:
            # 拡張子の追加（.jsonがない場合）
            if not dataset_name.endswith(".json"):
                few_shot_file = f"{dataset_name}_examples.json"
            else:
                few_shot_file = dataset_name.replace(".json", "_examples.json")
            
            few_shot_path = self.few_shot_base_dir / few_shot_file
            if few_shot_path.exists():
                return few_shot_path
        
        # デフォルトのパスを使用
        default_few_shot_dir = Path("data/few_shots")
        
        # MT-bench系のデータセットの場合は特別な処理
        if dataset_name.startswith("mt-bench-"):
            few_shot_path = default_few_shot_dir / "mt-bench" / f"{dataset_name.replace('mt-bench-', '')}_examples.json"
            if few_shot_path.exists():
                return few_shot_path
        
        # 通常のデータセット
        if not dataset_name.endswith(".json"):
            few_shot_file = f"{dataset_name}_examples.json"
        else:
            few_shot_file = dataset_name.replace(".json", "_examples.json")
        
        few_shot_path = default_few_shot_dir / few_shot_file
        if few_shot_path.exists():
            return few_shot_path
        
        return None
    
    def _save_category_results(self, results: Dict[str, Any], category: str) -> None:
        """
        カテゴリ評価結果を保存する
        
        Args:
            results: 評価結果
            category: カテゴリ名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{category}_{self.model_name}_{timestamp}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"カテゴリ評価結果を保存しました: {output_path}")
