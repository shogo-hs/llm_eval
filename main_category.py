#!/usr/bin/env python3
"""
LLM評価プラットフォームの大分類カテゴリ評価モジュール
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

from evaluator.category_evaluator import CategoryEvaluator
from evaluator.category_mapping import CategoryMapping
# モジュールから直接変数をインポート
from evaluator.category_mapping import GLP_CATEGORIES, ALT_CATEGORIES, AGGREGATE_CATEGORIES

async def evaluate_categories(
    categories: List[str],
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
    auto_adjust_samples: bool = True,
) -> Dict[str, Any]:
    """
    指定されたカテゴリを評価する
    
    Args:
        categories: カテゴリのリスト
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
    
    Returns:
        Dict[str, Any]: 評価結果
    """
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # カテゴリ評価マネージャーの作成
    evaluator = CategoryEvaluator(
        model_name,
        output_dir,
        dataset_base_dir,
        few_shot_base_dir,
        max_samples,
        batch_size,
        api_url,
        temperature,
        max_tokens,
        reuse_existing_results,
        auto_adjust_samples
    )
    
    # 各カテゴリを評価
    category_results = {}
    has_errors = False
    error_message = ""
    
    try:
        for category in categories:
            print(f"\n=== カテゴリ評価: {category} ===")
            try:
                result = await evaluator.evaluate_category(category)
                category_results[category] = result
            except ValueError as e:
                has_errors = True
                error_message = str(e)
                print(f"\n評価エラー: {error_message}")
                break  # 一つでもエラーがあれば評価を中止
        
        # エラーがなければ総合結果を作成して保存
        if not has_errors and category_results:
            # 総合結果を作成
            summary = {
                "model": model_name,
                "categories": categories,
                "category_scores": {category: results["score"] for category, results in category_results.items()},
                "timestamp": datetime.now().isoformat()
            }
            
            # 総合結果を保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = output_dir / f"summary_{model_name}_{timestamp}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n総合評価結果を保存しました: {summary_path}")
            
            # スコアを表示
            print("\n=== 評価スコア ===")
            for category, score in summary["category_scores"].items():
                print(f"{category}: {score:.4f}")
            
            return summary
        else:
            # エラーがあった場合
            raise ValueError(error_message or "評価中にエラーが発生しました")
    
    except Exception as e:
        print(f"評価中にエラーが発生しました: {str(e)}")
        if traceback:
            print(traceback.format_exc())
        
        # エラー結果は返すが、ファイルには保存しない
        return {
            "error": str(e),
            "model": model_name
        }

async def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description="LLM評価プラットフォーム（大分類カテゴリ評価）")
    
    # カテゴリ一覧表示オプション
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="利用可能なカテゴリの一覧を表示"
    )
    
    # カテゴリ指定オプション
    parser.add_argument(
        "--category", "-c",
        required=False,  # 必須でなくする
        help="評価するカテゴリ（GLP_xxxx, ALT_xxxx, GLP_AVG, ALT_AVG, TOTAL_AVG）（カンマ区切りで複数指定可能）"
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
        "--dataset-dir", "-d",
        default=None,
        help="データセットベースディレクトリ（未指定の場合はデフォルト）"
    )
    
    parser.add_argument(
        "--few-shot-dir", "-f",
        default=None,
        help="Few-shotサンプルベースディレクトリ（未指定の場合はデフォルト）"
    )
    
    parser.add_argument(
        "--max-samples", "-s",
        type=int,
        default=None,
        help="最大サンプル数（デフォルト: 全サンプル）"
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
        "--no-reuse",
        action="store_true",
        help="既存の評価結果を再利用しない（常に再評価する）"
    )
    
    parser.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="サンプル数の自動調整を無効化する（ドキュメントに従ったサンプル数調整を行わない）"
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
    
    # カテゴリ一覧表示
    if args.list_categories:
        print("=== 利用可能なカテゴリ ===")
        
        print("\n--- GLP（汎用的言語処理能力）カテゴリ ---")
        for category in GLP_CATEGORIES:  # CategoryMapping.GLP_CATEGORIESから変更
            datasets = CategoryMapping.get_datasets_for_category(category)
            dataset_names = [d["dataset"] for d in datasets]
            print(f"{category}: {len(dataset_names)}個のデータセット")
        
        print("\n--- ALT（アラインメント）カテゴリ ---")
        for category in ALT_CATEGORIES:  # CategoryMapping.ALT_CATEGORIESから変更
            datasets = CategoryMapping.get_datasets_for_category(category)
            dataset_names = [d["dataset"] for d in datasets]
            print(f"{category}: {len(dataset_names)}個のデータセット")
        
        print("\n--- 総合カテゴリ ---")
        for category, sub_categories in AGGREGATE_CATEGORIES.items():  # CategoryMapping.AGGREGATE_CATEGORIESから変更
            print(f"{category}: {len(sub_categories)}個のサブカテゴリ")
        
        return
    
    # カテゴリが指定されていない場合はエラー（--list-categories以外の場合）
    if not args.category:
        parser.error("カテゴリを指定してください。--category/-c オプションが必要です。")
        return
    
    # パスの解決
    output_dir = Path(args.output)
    dataset_base_dir = Path(args.dataset_dir) if args.dataset_dir else None
    few_shot_base_dir = Path(args.few_shot_dir) if args.few_shot_dir else None
    
    # カテゴリの解析（カンマ区切り）
    categories = [c.strip() for c in args.category.split(",")]
    
    # カテゴリの検証
    all_categories = CategoryMapping.get_all_categories()
    invalid_categories = [c for c in categories if c not in all_categories]
    if invalid_categories:
        print(f"エラー: 無効なカテゴリが指定されました: {', '.join(invalid_categories)}")
        print(f"利用可能なカテゴリを確認するには --list-categories オプションを使用してください")
        return
    
    print(f"評価カテゴリ: {', '.join(categories)}")
    print(f"モデル名: {args.model}")
    print(f"出力ディレクトリ: {output_dir}")
    
    if dataset_base_dir:
        print(f"データセットベースディレクトリ: {dataset_base_dir}")
    
    if few_shot_base_dir:
        print(f"Few-shotサンプルベースディレクトリ: {few_shot_base_dir}")
    
    if args.max_samples:
        print(f"最大サンプル数: {args.max_samples}")
    
    print(f"既存結果の再利用: {'無効' if args.no_reuse else '有効'}")
    print(f"サンプル数の自動調整: {'無効' if args.no_auto_adjust else '有効'}")
    
    # 評価の実行
    await evaluate_categories(
        categories,
        args.model,
        output_dir,
        dataset_base_dir,
        few_shot_base_dir,
        args.max_samples,
        args.batch_size,
        args.api_url,
        args.temperature,
        args.max_tokens,
        not args.no_reuse,
        not args.no_auto_adjust
    )


if __name__ == "__main__":
    asyncio.run(main())
