# LLM評価プラットフォーム

このプロジェクトは、LLM（Language Model）の評価を行うためのプラットフォームです。Jasterベンチマークなどの所定のJSON形式のデータセットを用いて、LLMの性能を評価することができます。

## 特徴

- 抽象化された評価指標・データセット・LLMクライアントのインターフェース
- 拡張可能なファクトリーデザインパターンによる実装
- Jasterベンチマークデータセットのサポート
- 非同期APIによる効率的なLLM呼び出し
- バッチ処理によるスケーラビリティ
- Few-shotプロンプティングのサポート
- 詳細な評価結果の出力（JSON, CSV）

## インストール

Python 3.8以上が必要です。

```bash
# 仮想環境を作成（既存の.venv環境を使用）
# 必要なパッケージをインストール
pip install -r requirements.txt
```

## 使用方法

### コマンドライン引数

```bash
python main.py --dataset /path/to/dataset.json --model model_name --output /path/to/output
```

### オプション

- `--dataset`, `-d`: 評価するデータセットファイルまたはディレクトリパス（必須）
- `--model`, `-m`: 評価対象のモデル名（デフォルト: \"elyza\"）
- `--output`, `-o`: 評価結果の出力ディレクトリ（デフォルト: \"./results\"）
- `--few-shot`, `-f`: Few-shotサンプル数（デフォルト: 0）
- `--batch-size`, `-b`: バッチサイズ（デフォルト: 5）
- `--api-url`: ローカルLLM API URL（デフォルト: \"http://192.168.3.43:8000/v1/chat/completions\"）
- `--temperature`, `-t`: 生成の温度（デフォルト: 0.7）
- `--max-tokens`: 最大生成トークン数（デフォルト: 500）

### 例

```bash
# 単一のデータセットを評価
python main.py --dataset /workspace/jaster/1.2.1/evaluation/test/jamp.json --model elyza --output /workspace/llm_eval/results

# ディレクトリ内の全データセットを評価
python main.py --dataset /workspace/jaster/1.2.1/evaluation/test --model elyza --output /workspace/llm_eval/results

# Few-shotサンプルを使用
python main.py --dataset /workspace/jaster/1.2.1/evaluation/test/jamp.json --model elyza --few-shot 3
```

## プロジェクト構造

```
/workspace/llm_eval/
├── main.py              # メインスクリプト
├── test_evaluator.py    # テストスクリプト
├── results/             # 評価結果の出力ディレクトリ
└── evaluator/           # 評価モジュール
    ├── __init__.py
    ├── base.py          # 抽象基底クラス定義
    ├── metrics.py       # 評価指標の実装
    ├── datasets.py      # データセットの実装
    ├── llm.py           # LLMクライアントの実装
    └── evaluator.py     # 評価ツールの実装
```

## 拡張方法

### 新しい評価指標の追加

`evaluator/metrics.py` に新しい評価指標クラスを追加し、`MetricFactory` に登録します。

```python
class NewMetric(BaseMetric):
    def __init__(self):
        super().__init__(name=\"new_metric\")
    
    def calculate(self, predicted: str, reference: str) -> float:
        # 評価スコアの計算処理
        return score

# ファクトリーへの登録
MetricFactory.register(\"new_metric\", NewMetric)
```

### 新しいデータセットタイプの追加

`evaluator/datasets.py` に新しいデータセットクラスを追加し、`DatasetFactory` に登録します。

```python
class NewDataset(BaseDataset):
    def __init__(self, name: str, data_path: Union[str, Path]):
        super().__init__(name, data_path)
    
    def get_samples(self) -> List[Dict[str, str]]:
        # サンプルの取得処理
        return samples

# ファクトリーへの登録
DatasetFactory.register(\"new_dataset\", NewDataset)
```

### 新しいLLMクライアントの追加

`evaluator/llm.py` に新しいLLMクライアントクラスを追加し、`LLMFactory` に登録します。

```python
class NewLLM(BaseLLM):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        # 初期化処理
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # テキスト生成処理
        return generated_text
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        # バッチテキスト生成処理
        return generated_texts

# ファクトリーへの登録
LLMFactory.register(\"new_llm\", NewLLM)
```

## ライセンス

MITライセンス