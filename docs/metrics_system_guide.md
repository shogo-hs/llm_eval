# 評価指標管理システム 使用ガイド

## 概要

このドキュメントでは、LLM評価プラットフォームの評価指標管理システムの使用方法について説明します。新しい管理システムは、評価指標をデータベースとディレクトリベースのプラグインシステムで管理し、動的な読み込みと登録をサポートします。

主な機能：

1. 評価指標の動的読み込み
2. データベースによる永続化管理
3. カスタム評価指標の追加
4. メタデータ管理（バージョン、作成者など）
5. APIエンドポイントによるフロントエンド連携

## システム構成

システムは次のコンポーネントで構成されています：

- **MetricsDatabase**: 評価指標をSQLiteデータベースで永続化管理
- **MetricsLoader**: ディレクトリからの評価指標の動的読み込み
- **MetricFactory**: 評価指標のインスタンス化と登録
- **API**: フロントエンドとの連携用エンドポイント

## 基本的な使い方

### 1. 評価指標のインポートと使用

```python
# 新しい方法（推奨）
from evaluator.metrics_factory import MetricFactory

# 評価指標のインスタンス化
metric = MetricFactory.create("exact_match")

# 評価実行
score = metric.calculate("Hello", "Hello")
print(f"Score: {score}")  # 1.0
```

旧バージョンとの互換性のために、以前の方法も引き続きサポートされています：

```python
# 旧バージョンとの互換性
from evaluator.metrics_factory import MetricFactory

# 評価指標のインスタンス化
metric = MetricFactory.create("exact_match")

# 評価実行
score = metric.calculate("Hello", "Hello")
print(f"Score: {score}")  # 1.0
```

### 2. 利用可能な評価指標の一覧取得

```python
from evaluator.metrics_factory import MetricFactory

# 利用可能な評価指標の一覧を取得
metrics = MetricFactory.list_metrics()
print(metrics)  # ['exact_match', 'exact_match_figure', 'char_f1', ...]
```

### 3. 複数の評価指標でまとめて評価

```python
from evaluator.metrics_factory import MetricFactory

# 複数の評価指標をまとめて作成
metrics = MetricFactory.create_from_list(["exact_match", "char_f1", "set_f1"])

# 評価結果
results = {}
for metric in metrics:
    score = metric.calculate("Hello World", "Hello there")
    results[metric.name] = score

print(results)  # {'exact_match': 0.0, 'char_f1': 0.9, 'set_f1': 0.5}
```

## カスタム評価指標の作成

### 1. Pythonコードでの作成

カスタム評価指標は、`BaseMetric`クラスを継承して作成します：

```python
from evaluator.base import BaseMetric

class MyCustomMetric(BaseMetric):
    """
    カスタム評価指標の説明
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="my_custom_metric")  # 名前を指定
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        # 評価ロジックを実装
        # 例：文字列の一致度を計算
        similarity = 0.0
        if predicted == reference:
            similarity = 1.0
        else:
            # カスタムロジック
            pass
        
        return similarity
```

作成したカスタム評価指標を登録する方法：

```python
from evaluator.metrics_factory import MetricFactory

# 登録
MetricFactory.register("my_custom_metric", MyCustomMetric)

# 使用
metric = MetricFactory.create("my_custom_metric")
score = metric.calculate("Hello", "Hello")
print(f"Score: {score}")  # 1.0
```

### 2. プログラムによるカスタム評価指標の動的追加

カスタム評価指標をコード文字列から動的に追加することもできます：

```python
from evaluator.metrics_factory import MetricFactory

# カスタム評価指標のPythonコード
code = """
from evaluator.base import BaseMetric

class WordCountSimilarity(BaseMetric):
    \"""
    単語数の類似度に基づく評価指標
    \"""
    
    def __init__(self):
        \"""
        初期化メソッド
        \"""
        super().__init__(name="word_count_similarity")
    
    def calculate(self, predicted: str, reference: str) -> float:
        \"""
        単語数の類似度で評価する
        \"""
        # 単語数をカウント
        pred_count = len(predicted.strip().split())
        ref_count = len(reference.strip().split())
        
        # 単語数の差を計算し、類似度に変換
        if pred_count == 0 and ref_count == 0:
            return 1.0
        
        if pred_count == 0 or ref_count == 0:
            return 0.0
        
        # 大きい方で割って、その差を1から引く
        similarity = 1.0 - abs(pred_count - ref_count) / max(pred_count, ref_count)
        return similarity
"""

# 登録
success = MetricFactory.add_custom_metric(
    name="word_count_similarity",
    code=code,
    description="単語数の類似度に基づく評価指標",
    created_by="user1",
    save_to_file=True  # Trueの場合、ファイルとしても保存
)

if success:
    # 使用
    metric = MetricFactory.create("word_count_similarity")
    score = metric.calculate("Hello World", "Hello World Test")
    print(f"Score: {score}")  # 0.67
```

### 3. ファイルベースでのカスタム評価指標の追加

カスタム評価指標は、`evaluator/metrics/custom/`ディレクトリに.pyファイルとして配置することもできます：

1. `evaluator/metrics/custom/my_metric.py`ファイルを作成
2. `BaseMetric`を継承したクラスを実装
3. システム起動時に自動的に読み込まれます

## APIの使用方法

評価指標管理システムはAPIエンドポイントを提供しており、フロントエンドからカスタム評価指標を追加できます。

### 1. APIサーバーの起動

```python
from evaluator.api import start_api_server

# APIサーバーを起動
start_api_server(host="0.0.0.0", port=8000)
```

または、以下のコマンドでも起動できます：

```bash
python -m evaluator.api
```

### 2. 主なエンドポイント

- `GET /metrics`: 利用可能な評価指標の一覧を返します
- `GET /metrics/detail`: 評価指標の詳細情報を返します
- `POST /metrics`: 新しいカスタム評価指標を追加します
- `DELETE /metrics/{metric_name}`: 評価指標を削除します（論理削除）

### 3. カスタム評価指標の追加例

```bash
curl -X POST "http://localhost:8000/metrics" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "word_count_similarity",
       "code": "from evaluator.base import BaseMetric\n\nclass WordCountSimilarity(BaseMetric):\n  ...",
       "description": "単語数の類似度に基づく評価指標",
       "created_by": "api_user",
       "save_to_file": true
     }'
```

## ベストプラクティス

1. **命名規則**: 評価指標名はスネークケース（例：`word_count_similarity`）で、クラス名はパスカルケース（例：`WordCountSimilarity`）を使用してください。

2. **スコアの範囲**: 評価スコアは0.0から1.0の範囲に正規化することをお勧めします。

3. **ドキュメンテーション**: カスタム評価指標にはdocstringを使って詳細な説明を記述してください。

4. **エッジケース処理**: 空文字列や不正な入力に対する処理を必ず実装してください。

5. **評価指標の再利用**: 複雑な評価指標は、既存の評価指標を組み合わせて作成することを検討してください。

## トラブルシューティング

1. **評価指標が見つからない**

```
ValueError: Unsupported metric: unknown_metric
```

この場合、評価指標名が間違っているか、該当する評価指標が登録されていません。`MetricFactory.list_metrics()`で利用可能な評価指標を確認してください。

2. **カスタム評価指標の登録に失敗**

登録に失敗する一般的な理由：
- BaseMetricを継承していない
- クラス名とファイル名が一致していない
- 構文エラーがある
- 必要なインポートが不足している

3. **データベースの問題**

データベースに問題がある場合は、`evaluator/data/metrics.db`を削除して再作成することができます。

## 参考情報

- BaseMetricクラス: `evaluator/base.py`
- 組み込み評価指標: `evaluator/metrics/*.py`
- カスタム評価指標の例: `evaluator/metrics/custom/rouge_l.py`
