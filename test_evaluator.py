#!/usr/bin/env python3
"""
評価モジュールのテストスクリプト
"""
import asyncio
import json
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evaluator import BaseMetric, BaseDataset, BaseLLM, BaseEvaluator
from evaluator.datasets import JasterDataset, DatasetFactory
from evaluator.llm import LocalLLM, LLMFactory
from evaluator.evaluator import JasterEvaluator, EvaluatorFactory
from evaluator.metrics import ExactMatch
from evaluator.metrics_factory import MetricFactory


class TestJasterDataset(unittest.TestCase):
    """
    Jasterデータセットのテスト
    """
    
    def setUp(self):
        """
        テスト準備
        """
        # テスト用データの作成
        self.test_data = {
            "instruction": "テスト指示",
            "output_length": 3,
            "metrics": ["exact_match"],
            "few_shots": [
                {"input": "テスト入力1", "output": "テスト出力1"},
                {"input": "テスト入力2", "output": "テスト出力2"}
            ],
            "samples": [
                {"input": "サンプル入力1", "output": "サンプル出力1"},
                {"input": "サンプル入力2", "output": "サンプル出力2"}
            ]
        }
        
        # テスト用データセットファイルの作成
        self.test_dir = Path("./test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.test_file = self.test_dir / "test_jaster.json"
        with open(self.test_file, "w", encoding="utf-8") as f:
            json.dump(self.test_data, f, ensure_ascii=False, indent=2)
        
        # テスト対象のインスタンス生成
        self.dataset = JasterDataset("test_jaster", self.test_file)
    
    def tearDown(self):
        """
        テスト終了処理
        """
        # テスト用ファイルの削除
        if self.test_file.exists():
            self.test_file.unlink()
        
        # テスト用ディレクトリの削除
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_load_data(self):
        """
        データ読み込みのテスト
        """
        # データの読み込み
        self.assertEqual(self.dataset.instruction, self.test_data["instruction"])
        self.assertEqual(self.dataset.output_length, self.test_data["output_length"])
        self.assertEqual(self.dataset.metrics, self.test_data["metrics"])
        self.assertEqual(self.dataset.few_shots, self.test_data["few_shots"])
        self.assertEqual(self.dataset.get_samples(), self.test_data["samples"])
    
    def test_get_prompt(self):
        """
        プロンプト生成のテスト
        """
        # Few-shotなしのプロンプト
        prompt = self.dataset.get_prompt("テスト入力")
        self.assertEqual(prompt, "テスト指示\n\nテスト入力")
        
        # Few-shotありのプロンプト
        prompt = self.dataset.get_prompt("テスト入力", few_shot_count=1)
        expected = "テスト指示\n\nテスト入力1\nテスト出力1\n\nテスト入力"
        self.assertEqual(prompt, expected)


class TestMetrics(unittest.TestCase):
    """
    評価指標のテスト
    """
    
    def test_exact_match(self):
        """
        完全一致評価指標のテスト
        """
        metric = ExactMatch()
        
        # 一致するケース
        self.assertEqual(metric.calculate("テスト", "テスト"), 1.0)
        
        # 一致しないケース
        self.assertEqual(metric.calculate("テスト", "テスト2"), 0.0)
        
        # 空白を除去するケース
        self.assertEqual(metric.calculate(" テスト ", "テスト"), 1.0)
    
    def test_metric_factory(self):
        """
        評価指標ファクトリーのテスト
        """
        # インスタンス生成のテスト
        metric = MetricFactory.create("exact_match")
        self.assertIsInstance(metric, ExactMatch)
        
        # リスト生成のテスト
        metrics = MetricFactory.create_from_list(["exact_match", "char_f1"])
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].name, "exact_match")
        self.assertEqual(metrics[1].name, "char_f1")


class TestLocalLLM(unittest.IsolatedAsyncioTestCase):
    """
    ローカルLLMクライアントのテスト
    """
    
    @patch("aiohttp.ClientSession.post")
    async def test_generate(self, mock_post):
        """
        テキスト生成のテスト
        """
        # モックの設定
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "content": "生成されたテキスト"
                    }
                }
            ]
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # テスト対象のインスタンス生成
        llm = LocalLLM("test_model")
        
        # テキスト生成
        result = await llm.generate("テストプロンプト")
        
        # 結果の検証
        self.assertEqual(result, "生成されたテキスト")


class TestJasterEvaluator(unittest.IsolatedAsyncioTestCase):
    """
    Jaster評価ツールのテスト
    """
    
    def setUp(self):
        """
        テスト準備
        """
        # モックの作成
        self.mock_dataset = MagicMock()
        self.mock_dataset.name = "test_dataset"
        self.mock_dataset.metrics = ["exact_match"]
        self.mock_dataset.get_samples.return_value = [
            {"input": "サンプル入力1", "output": "サンプル出力1"},
            {"input": "サンプル入力2", "output": "サンプル出力2"}
        ]
        # get_prompt メソッドのモックを正しく設定
        self.mock_dataset.get_prompt = MagicMock(side_effect=lambda input, few_shot_count=0: f"プロンプト: {input}")
        
        self.mock_llm = MagicMock()
        self.mock_llm.model_name = "test_model"
        self.mock_llm.generate_batch = AsyncMock(return_value=["サンプル出力1", "テスト出力"])
        
        # 評価指標のモック
        mock_metric = MagicMock()
        mock_metric.name = "exact_match"
        mock_metric.calculate.return_value = 1.0
        
        # テスト対象のインスタンス生成
        self.evaluator = JasterEvaluator(self.mock_dataset, self.mock_llm, metrics=[mock_metric])
    
    async def test_evaluate(self):
        """
        評価実行のテスト
        """
        # 評価の実行
        results = await self.evaluator.evaluate()
        
        # 結果の検証
        self.assertEqual(results["dataset"], "test_dataset")
        self.assertEqual(results["model"], "test_model")
        self.assertEqual(results["num_samples"], 2)
        self.assertIn("exact_match", results["metrics"])
        self.assertEqual(len(results["samples"]), 2)


if __name__ == "__main__":
    unittest.main()
