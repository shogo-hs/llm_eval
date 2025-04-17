# カスタム評価指標作成ガイド

このガイドでは、LLM評価プラットフォーム用のカスタム評価指標の作成方法について詳しく説明します。

## 基本的な評価指標構造

カスタム評価指標は、`BaseMetric`クラスを継承して作成します。最低限、以下のメソッドを実装する必要があります：

1. `__init__`: 初期化メソッド（名前と追加パラメータを設定）
2. `calculate`: 評価スコアを計算するメソッド

## 1. 基本的なカスタム評価指標

最も単純なカスタム評価指標の例を示します：

```python
from evaluator.base import BaseMetric

class SimpleMetric(BaseMetric):
    """
    シンプルな評価指標の例
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="simple_metric")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        # 単純な例：文字数の比率
        pred_len = len(predicted)
        ref_len = len(reference)
        
        if ref_len == 0:
            return 1.0 if pred_len == 0 else 0.0
        
        # スコアを0.0-1.0の範囲に正規化
        ratio = min(pred_len / ref_len, ref_len / pred_len)
        return ratio
```

## 2. パラメータ付き評価指標

評価指標に追加のパラメータを持たせることもできます：

```python
from evaluator.base import BaseMetric

class WeightedMatchMetric(BaseMetric):
    """
    重み付き一致度評価指標
    """
    
    def __init__(self, weight: float = 0.5, case_sensitive: bool = False):
        """
        初期化メソッド
        
        Args:
            weight: 一致時の重み (0.0-1.0)
            case_sensitive: 大文字小文字を区別するかどうか
        """
        super().__init__(name="weighted_match")
        self.weight = weight
        self.case_sensitive = case_sensitive
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        if not self.case_sensitive:
            predicted = predicted.lower()
            reference = reference.lower()
        
        # 完全一致の場合はweight、それ以外は0.0
        return self.weight if predicted.strip() == reference.strip() else 0.0
```

## 3. 外部ライブラリを使用する評価指標

外部ライブラリを使用して、より高度な評価指標を作成することもできます：

```python
from evaluator.base import BaseMetric
import numpy as np
from fuzzywuzzy import fuzz

class FuzzyMatchMetric(BaseMetric):
    """
    あいまい一致度に基づく評価指標
    """
    
    def __init__(self, match_threshold: float = 0.8):
        """
        初期化メソッド
        
        Args:
            match_threshold: 一致と見なす閾値 (0.0-1.0)
        """
        super().__init__(name="fuzzy_match")
        self.match_threshold = match_threshold
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        # fuzzywuzzyを使用してあいまい一致度を計算
        ratio = fuzz.ratio(predicted.strip(), reference.strip()) / 100.0
        
        # 閾値以上なら1.0、そうでなければ正規化した値
        return 1.0 if ratio >= self.match_threshold else ratio
```

## 4. 複数の評価指標を組み合わせる

複数の評価指標を組み合わせて、より高度な評価指標を作成することもできます：

```python
from evaluator.base import BaseMetric
from evaluator.metrics_factory import MetricFactory

class CompositeMetric(BaseMetric):
    """
    複数の評価指標を組み合わせた評価指標
    """
    
    def __init__(self, weights: dict = None):
        """
        初期化メソッド
        
        Args:
            weights: 評価指標名と重みのマップ (合計1.0になるようにすること)
        """
        super().__init__(name="composite_metric")
        
        # デフォルト重み
        self.weights = weights or {
            "exact_match": 0.3,
            "char_f1": 0.4,
            "set_f1": 0.3
        }
        
        # 評価指標のインスタンスを作成
        self.metrics = {}
        for metric_name in self.weights:
            self.metrics[metric_name] = MetricFactory.create(metric_name)
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力
            reference: 正解出力
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        # 各評価指標のスコアを計算し、重み付き平均を求める
        weighted_score = 0.0
        for metric_name, metric in self.metrics.items():
            score = metric.calculate(predicted, reference)
            weighted_score += score * self.weights[metric_name]
        
        return weighted_score
```

## 5. ドメイン固有の評価指標

特定のドメインに特化した評価指標を作成する例を示します：

```python
from evaluator.base import BaseMetric
import re

class JSONValidityMetric(BaseMetric):
    """
    JSON出力の有効性評価指標
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        super().__init__(name="json_validity")
    
    def calculate(self, predicted: str, reference: str) -> float:
        """
        評価スコアを計算する
        
        Args:
            predicted: モデルの予測出力（JSON形式を期待）
            reference: 正解出力（JSON形式を期待）
            
        Returns:
            float: 評価スコア（0.0-1.0）
        """
        try:
            import json
            
            # JSON構文の確認
            predicted_json = json.loads(predicted)
            reference_json = json.loads(reference)
            
            # キーの一致度を計算
            pred_keys = set(predicted_json.keys())
            ref_keys = set(reference_json.keys())
            
            if not ref_keys:
                return 1.0 if not pred_keys else 0.0
            
            # キーの一致度をスコアとする
            jaccard = len(pred_keys.intersection(ref_keys)) / len(pred_keys.union(ref_keys))
            return jaccard
            
        except json.JSONDecodeError:
            # JSON構文エラーの場合は0.0
            return 0.0
        except Exception:
            # その他のエラーの場合も0.0
            return 0.0
```

## 6. 評価指標のテスト

カスタム評価指標を作成したら、テストすることをお勧めします：

```python
def test_custom_metric():
    """
    カスタム評価指標のテスト
    """
    # 評価指標のインスタンス化
    metric = JSONValidityMetric()
    
    # テストケース
    test_cases = [
        {"pred": '{"a": 1, "b": 2}', "ref": '{"a": 1, "b": 2, "c": 3}', "expected": 0.67},
        {"pred": '{"a": 1}', "ref": '{"b": 2}', "expected": 0.0},
        {"pred": '{"a": 1, "b": 2}', "ref": '{"a": 1, "b": 2}', "expected": 1.0},
        {"pred": 'invalid json', "ref": '{"a": 1}', "expected": 0.0},
    ]
    
    # テスト実行
    for i, case in enumerate(test_cases):
        score = metric.calculate(case["pred"], case["ref"])
        expected = case["expected"]
        passed = abs(score - expected) < 0.01  # 許容誤差
        
        print(f"テスト{i+1}: {'成功' if passed else '失敗'}, スコア: {score:.2f}, 期待値: {expected:.2f}")

# テスト実行
test_custom_metric()
```

## 7. 評価指標の公開と共有

作成したカスタム評価指標は、以下の方法で公開・共有できます：

1. **ファイルとして保存**: `evaluator/metrics/custom/` ディレクトリに配置
2. **データベースに登録**: `MetricFactory.add_custom_metric` を使用
3. **APIを通じて公開**: `/metrics` エンドポイントでPOSTリクエスト

## ベストプラクティス

1. **命名規則**:
   - 評価指標名: スネークケース（例: `json_validity`）
   - クラス名: パスカルケース（例: `JSONValidityMetric`）

2. **正規化**:
   - スコアは0.0から1.0の範囲に正規化する
   - 1.0が最高（完全一致）、0.0が最低（不一致）

3. **エラー処理**:
   - 空文字列、無効な入力、例外などのエッジケースを適切に処理する
   - エラー時は適切なデフォルト値（通常は0.0）を返す

4. **ドキュメンテーション**:
   - docstringで目的、アルゴリズム、パラメータ、戻り値などを詳細に説明する
   - 典型的な使用例やエッジケースも記述する

5. **テスト**:
   - 複数のテストケースでスコアを検証する
   - エッジケースや無効な入力も必ずテストする

## よくある質問

### Q: 評価指標の名前はどのように決めるべきですか？
**A**: 評価指標の名前は、その機能や目的を短く表現できるものが良いでしょう。一般的な規則として、スネークケース（例: `word_count_similarity`）を使用し、他の評価指標と重複しないようにしてください。

### Q: 外部ライブラリに依存する評価指標を作成してもよいですか？
**A**: はい、外部ライブラリを使用することは可能です。ただし、そのライブラリが`requirements.txt`に含まれていること、またはインストール手順を明記することをお勧めします。

### Q: 評価指標のパラメータはどのように設定すべきですか？
**A**: パラメータは`__init__`メソッドで定義し、デフォルト値を設定することをお勧めします。パラメータの型とドキュメンテーションも明記してください。

### Q: 評価指標のスコアが0.0-1.0の範囲を超えた場合はどうなりますか？
**A**: スコアは常に0.0-1.0の範囲に正規化することが推奨されます。範囲外の値は、システムの他の部分で問題を引き起こす可能性があります。必要であれば、`min(max(score, 0.0), 1.0)`のように範囲を制限することを検討してください。

### Q: 既存の評価指標を拡張することはできますか？
**A**: はい、既存の評価指標クラスを継承することで、機能を拡張したカスタム評価指標を作成できます。例えば、`class EnhancedExactMatch(ExactMatch):`のように定義します。

## まとめ

カスタム評価指標を作成することで、LLM評価プラットフォームをより柔軟に活用できます。このガイドを参考に、プロジェクトやドメインに最適な評価指標を作成してください。質問やフィードバックがあれば、お気軽にお寄せください。
