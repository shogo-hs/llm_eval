#!/usr/bin/env python3
"""
LLM評価プラットフォームのメインモジュール
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

from evaluator import BaseMetric, BaseDataset, BaseLLM, BaseEvaluator
from evaluator.datasets import DatasetFactory
from evaluator.llm import LLMFactory
from evaluator.evaluator import EvaluatorFactory
from evaluator.metrics_factory import MetricFactory


async def evaluate_dataset(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    few_shot_count: int = 0,
    few_shot_path: Optional[Path] = None,
    batch_size: int = 5,
    api_url: str = "http://192.168.3.43:8000/v1/chat/completions",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Dict[str, Any]:
    """
    データセットを評価する
    
    Args:
        dataset_path: データセットパス
        model_name: モデル名
        output_dir: 出力ディレクトリ
        few_shot_count: Few-shotサンプル数
        few_shot_path: Few-shotサンプルのファイルパス (Noneの場合は使用しない)
        batch_size: バッチサイズ
        api_url: ローカルLLM API URL
        temperature: 生成の温度
        max_tokens: 最大生成トークン数
    
    Returns:
        Dict[str, Any]: 評価結果
    """
    try:
        # データセットの作成
        dataset_name = dataset_path.stem
        dataset = DatasetFactory.create("jaster", dataset_name, dataset_path, few_shot_path)
        
        # LLMクライアントの作成
        llm = LLMFactory.create(
            "local",
            model_name,
            api_url=api_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 評価ツールの作成
        evaluator = EvaluatorFactory.create(
            "jaster",
            dataset,
            llm,
            few_shot_count=few_shot_count,
            few_shot_path=few_shot_path,
            batch_size=batch_size
        )
        
        print(f"評価開始: {dataset_name}")
        
        # 評価の実行
        results = await evaluator.evaluate()
        
        # 出力ディレクトリの作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果の保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{dataset_name}_{model_name}_{timestamp}.json"
        evaluator.save_results(results, output_path)
        
        print(f"評価結果を保存しました: {output_path}")
        
        return results
    except Exception as e:
        print(f"評価中にエラーが発生しました: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "dataset": dataset_path.stem,
            "model": model_name
        }


async def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description="LLM評価プラットフォーム")
    
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="評価するデータセットパスまたはディレクトリ"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="elyza",
        help="評価対象のモデル名"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="評価結果の出力ディレクトリ"
    )
    
    parser.add_argument(
        "--few-shot", "-f",
        type=int,
        default=0,
        help="Few-shotサンプル数"
    )
    
    parser.add_argument(
        "--few-shot-path", "-p",
        default=None,
        help="Few-shotサンプルのファイルパスまたはディレクトリ (未指定の場合は使用しない)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="バッチサイズ"
    )
    
    parser.add_argument(
        "--api-url",
        default="http://192.168.3.43:8000/v1/chat/completions",
        help="ローカルLLM API URL"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="生成の温度"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="最大生成トークン数"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード（詳細なログ出力）"
    )
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    if args.debug:
        print(f"デバッグモード: 有効")
        print(f"引数: {args}")
    
    # パスの解決
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    few_shot_path = Path(args.few_shot_path) if args.few_shot_path else None
    
    if not dataset_path.exists():
        print(f"エラー: 指定されたデータセットパスが存在しません: {dataset_path}")
        return
    
    print(f"データセットパス: {dataset_path}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"Few-shotパス: {few_shot_path}")
    print(f"API URL: {args.api_url}")
    
    # データセットの評価
    if dataset_path.is_file():
        # 単一のデータセットを評価
        await evaluate_dataset(
            dataset_path,
            args.model,
            output_dir,
            args.few_shot,
            few_shot_path,
            args.batch_size,
            args.api_url,
            args.temperature,
            args.max_tokens
        )
    elif dataset_path.is_dir():
        # ディレクトリ内の全データセットを評価
        dataset_files = list(dataset_path.glob("**/*.json"))
        if not dataset_files:
            print(f"エラー: 指定されたディレクトリにデータセットファイルが見つかりません: {dataset_path}")
            return
        
        print(f"見つかったデータセットファイル数: {len(dataset_files)}")
        
        results = {}
        for dataset_file in dataset_files:
            print(f"\n評価データセット: {dataset_file}")
            result = await evaluate_dataset(
                dataset_file,
                args.model,
                output_dir,
                args.few_shot,
                few_shot_path,
                args.batch_size,
                args.api_url,
                args.temperature,
                args.max_tokens
            )
            
            if "error" not in result:
                results[dataset_file.stem] = result
        
        if results:
            # 全結果の要約を保存
            summary = {
                "model": args.model,
                "timestamp": datetime.now().isoformat(),
                "num_datasets": len(results),
                "few_shot_count": args.few_shot,
                "few_shot_path": str(few_shot_path) if few_shot_path else None,
                "metrics": {
                    dataset_name: {
                        metric_name: metric_value
                        for metric_name, metric_value in result["metrics"].items()
                    }
                    for dataset_name, result in results.items()
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = output_dir / f"summary_{args.model}_{timestamp}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n要約を保存しました: {summary_path}")
    else:
        print(f"エラー: 無効なデータセットパス: {dataset_path}")


if __name__ == "__main__":
    asyncio.run(main())