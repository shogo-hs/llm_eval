"""
LLMクライアントの具体実装モジュール
"""
import aiohttp
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import traceback

from .base import BaseLLM


class LocalLLM(BaseLLM):
    """
    ローカルLLM APIクライアント
    """
    
    def __init__(self, 
                 model_name: str, 
                 api_url: str = "http://localhost:8000/v1/chat/completions", 
                 temperature: float = 0.7, 
                 max_tokens: int = 500, 
                 timeout: float = 60.0,
                 retry_count: int = 3,
                 retry_delay: float = 1.0,
                 **kwargs):
        """
        初期化メソッド
        
        Args:
            model_name: モデル名
            api_url: API URL
            temperature: 生成の温度
            max_tokens: 最大生成トークン数
            timeout: タイムアウト秒数
            retry_count: リトライ回数
            retry_delay: リトライ前の待機時間（秒）
            **kwargs: その他のパラメータ
        """
        super().__init__(model_name)
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.kwargs = kwargs
        
        # デバッグメッセージ
        print(f"LocalLLM初期化: モデル={model_name}, URL={api_url}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を行う
        
        Args:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ（temperatureなど）
            
        Returns:
            str: 生成されたテキスト
        
        Raises:
            Exception: API呼び出しに失敗した場合
        """
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        # 追加パラメータの設定
        for k, v in kwargs.items():
            if k not in params:
                params[k] = v
        
        # APIリクエスト
        try:
            response_text = await self._api_call(params)
            return response_text
        except Exception as e:
            print(f"テキスト生成エラー: {e}")
            print(f"プロンプト: {prompt[:100]}...")
            # エラー時はモックのレスポンスを返す（テスト実行用）
            if kwargs.get("mock_on_error", False):
                print(f"モックレスポンスを返します")
                return "APIエラーによるモックレスポンス"
            raise e
    
    async def generate_batch(self, prompts: List[str], batch_size: int = 5, **kwargs) -> List[str]:
        """
        バッチ処理でテキスト生成を行う
        
        Args:
            prompts: 入力プロンプトのリスト
            batch_size: 同時実行するリクエストの最大数
            **kwargs: 生成パラメータ
            
        Returns:
            List[str]: 生成されたテキストのリスト
        """
        print(f"バッチ処理開始: {len(prompts)}件, バッチサイズ={batch_size}")
        
        # セマフォを使用して同時リクエスト数を制限
        semaphore = asyncio.Semaphore(batch_size)
        
        async def process_prompt(idx: int, prompt: str) -> Tuple[int, str]:
            async with semaphore:
                try:
                    print(f"プロンプト処理 {idx}/{len(prompts)}")
                    result = await self.generate(prompt, **kwargs)
                    return idx, result
                except Exception as e:
                    print(f"プロンプト{idx}の処理中にエラーが発生: {e}")
                    return idx, ""
        
        # 各プロンプトを処理するタスクを作成
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # すべてのタスクを実行
        results_with_index = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を元の順序で並べる
        results = [""] * len(prompts)
        for result in results_with_index:
            if isinstance(result, Exception):
                print(f"タスク実行中に例外が発生: {result}")
                continue
            
            idx, text = result
            results[idx] = text
        
        return results
    
    async def _api_call(self, params: Dict[str, Any]) -> str:
        """
        ローカルLLM APIを呼び出す
        
        Args:
            params: APIパラメータ
            
        Returns:
            str: 生成されたテキスト
            
        Raises:
            Exception: リトライ回数を超えてもAPI呼び出しに失敗した場合
        """
        headers = {"Content-Type": "application/json"}
        
        for i in range(self.retry_count + 1):
            try:
                print(f"APIリクエスト試行 {i+1}/{self.retry_count+1}: {self.api_url}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=headers,
                        data=json.dumps(params),
                        timeout=self.timeout
                    ) as response:
                        if response.status != 200:
                            response_text = await response.text()
                            raise Exception(
                                f"API呼び出しがステータス{response.status}で失敗: {response_text}"
                            )
                        
                        response_data = await response.json()
                        
                        # レスポンスから生成テキストを抽出
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            message = response_data["choices"][0].get("message", {})
                            generated_text = message.get("content", "")
                            return generated_text
                        else:
                            raise Exception(f"予期しないAPIレスポンス: {response_data}")
            
            except asyncio.TimeoutError:
                print(f"タイムアウトエラー (試行 {i+1}/{self.retry_count+1})")
                if i < self.retry_count:
                    # リトライ前に待機
                    retry_wait = self.retry_delay * (i + 1)
                    print(f"{retry_wait}秒後にリトライします")
                    await asyncio.sleep(retry_wait)
                else:
                    raise Exception(f"{self.retry_count}回のリトライ後もAPI呼び出しがタイムアウト")
            
            except Exception as e:
                print(f"API呼び出しエラー (試行 {i+1}/{self.retry_count+1}): {str(e)}")
                if i < self.retry_count:
                    # リトライ前に待機
                    retry_wait = self.retry_delay * (i + 1)
                    print(f"{retry_wait}秒後にリトライします")
                    await asyncio.sleep(retry_wait)
                else:
                    print(f"最大リトライ回数に達しました。エラー: {str(e)}")
                    print(traceback.format_exc())
                    raise e
        
        # 通常はここには到達しないはず
        raise Exception("API呼び出し中に予期せぬエラーが発生")


class MockLLM(BaseLLM):
    """
    モックLLMクライアント（テスト用）
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        初期化メソッド
        
        Args:
            model_name: モデル名
            **kwargs: その他のパラメータ
        """
        super().__init__(model_name)
        self.kwargs = kwargs
        self.mock_responses = kwargs.get("mock_responses", {})
        self.default_response = kwargs.get("default_response", "モックレスポンス")
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を行う（モック）
        
        Args:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ
            
        Returns:
            str: 生成されたテキスト
        """
        # プロンプトに基づいてレスポンスを返す
        for key, response in self.mock_responses.items():
            if key in prompt:
                return response
        
        return self.default_response
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        バッチ処理でテキスト生成を行う（モック）
        
        Args:
            prompts: 入力プロンプトのリスト
            **kwargs: 生成パラメータ
            
        Returns:
            List[str]: 生成されたテキストのリスト
        """
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, **kwargs)
            results.append(result)
        
        return results


class LLMFactory:
    """
    LLMクライアントファクトリー
    
    LLMタイプからLLMクライアントインスタンスを生成する
    """
    
    _llm_map = {
        "local": LocalLLM,
        "mock": MockLLM
    }
    
    @classmethod
    def create(cls, llm_type: str, model_name: str, **kwargs) -> BaseLLM:
        """
        LLMクライアントインスタンスを生成する
        
        Args:
            llm_type: LLMタイプ
            model_name: モデル名
            **kwargs: LLMクライアントの初期化パラメータ
            
        Returns:
            BaseLLM: LLMクライアントインスタンス
        
        Raises:
            ValueError: 未サポートのLLMタイプが指定された場合
        """
        if llm_type not in cls._llm_map:
            raise ValueError(f"未サポートのLLMタイプ: {llm_type}")
        
        llm_class = cls._llm_map[llm_type]
        return llm_class(model_name, **kwargs)
    
    @classmethod
    def register(cls, llm_type: str, llm_class: type):
        """
        LLMクライアントクラスを登録する
        
        Args:
            llm_type: LLMタイプ
            llm_class: LLMクライアントクラス
        """
        cls._llm_map[llm_type] = llm_class
