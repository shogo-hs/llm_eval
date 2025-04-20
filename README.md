# LLM評価プラットフォーム

このプロジェクトは、LLM（大規模言語モデル）の評価のための統合ソリューションです。JasterやJBBQなどの日本語ベンチマークを使用してモデルを評価し、個別データセットの評価と大分類カテゴリ評価の両方をサポートします。

## 特徴

- **個別データセット評価**
  - 拡張可能なファクトリーデザインパターンによる実装
  - JasterやJBBQなどの日本語ベンチマークデータセットのサポート
  - 非同期APIによる効率的なLLM呼び出し
  - Few-shotプロンプティングのサポート
  - プラグイン式メトリクスシステム（カスタムメトリクスの追加が容易）

- **大分類評価機能**
  - 小分類データセット評価を大分類カテゴリに集約
  - GLP（汎用的言語処理能力）とALT（アラインメント）カテゴリの評価
  - 総合評価指標（GLP_AVG、ALT_AVG、TOTAL_AVG）の算出
  - 0-shot/2-shotの自動平均化
  - スコア正規化と階層的な結果表示

- **メトリクス管理システム**
  - SQLiteデータベースによる永続化管理
  - 動的なカスタムメトリクスの読み込みと登録
  - APIを通じたメトリクス管理（追加、一覧取得、削除）
  - メタデータ管理（バージョン、作成者など）

## ディレクトリ構造

```
/
├── main.py                      # 個別データセット評価用メインスクリプト
├── main_category.py             # 大分類カテゴリ評価用メインスクリプト
├── evaluator/                   # 評価モジュール
│   ├── __init__.py
│   ├── api.py                   # API定義とサーバー起動モジュール
│   ├── base.py                  # 抽象基底クラス定義
│   ├── datasets.py              # データセットの実装
│   ├── evaluator.py             # 評価ツールの実装
│   ├── category_mapping.py      # 大分類カテゴリ定義モジュール
│   ├── category_evaluator.py    # 大分類カテゴリ評価マネージャー
│   ├── llm.py                   # LLMクライアントの実装
│   ├── metrics_db.py            # 評価指標データベース管理
│   ├── metrics_factory.py       # 評価指標ファクトリー
│   ├── metrics_loader.py        # 評価指標動的読み込み
│   ├── metrics/                 # 評価指標実装
│   │   ├── __init__.py
│   │   ├── acc_diff.py          # 正解率差分メトリクス
│   │   ├── bias_score.py        # バイアススコアメトリクス
│   │   ├── bleu.py              # BLEUスコアメトリクス
│   │   ├── char_f1.py           # 文字単位F1スコアメトリクス
│   │   ├── contains_answer.py   # 回答含有チェックメトリクス
│   │   ├── exact_match.py       # 完全一致メトリクス
│   │   ├── exact_match_figure.py # 数値抽出完全一致メトリクス
│   │   ├── set_f1.py            # セット単位F1スコアメトリクス
│   │   └── custom/              # カスタムメトリクス
│   │       ├── __init__.py
│   │       ├── jaccard_similarity.py
│   │       ├── rouge_l.py
│   │       └── word_count_similarity.py
│
├── pyproject.toml               # プロジェクト設定
├── requirements.txt             # 共通の必要パッケージリスト
├── docs/                        # ドキュメント
│   ├── custom_metrics_guide.md  # カスタムメトリクス作成ガイド
│   ├── metrics_system_guide.md  # メトリクスシステム使用ガイド
│   ├── llm_evaluation_metrics.md # 評価指標の詳細ドキュメント
│   └── category_evaluation_guide.md # 大分類評価機能使用ガイド
└── tests/                       # テストスクリプト
    ├── test_evaluator.py
    ├── test_llm_connection.py
    ├── test_metrics_system.py
    └── test_category_mapping.py  # カテゴリマッピングテスト
```

## インストールと環境設定

### 前提条件

- Python 3.8+
- pip または uv パッケージマネージャー

### インストール手順

```bash
# プロジェクトのクローン
git clone https://your-repo-url/llm_eval_integrated.git
cd llm_eval_integrated

# 仮想環境を作成（uvを使用する場合）
uv venv
source .venv/bin/activate  # Linuxの場合
# または
.venv\\Scripts\\activate     # Windowsの場合

# 依存パッケージのインストール
uv pip install -r requirements.txt
```

## 使用方法

### 1. 個別データセット評価の実行

#### 基本的な評価の実行

```bash
# 単一のデータセットを評価
python main.py --dataset /path/to/dataset.json --model model_name --output ./results

# ディレクトリ内の全データセットを評価
python main.py --dataset /path/to/datasets_dir --model model_name --output ./results

# 特定のサンプル数で実行（開発時やテスト時に便利）
python main.py --dataset /path/to/dataset.json --model model_name --max-samples 10
```

#### Few-shotプロンプティングの使用

```bash
# デフォルトのFew-shotサンプルを使用
python main.py --dataset /path/to/dataset.json --model model_name --few-shot 3

# カスタムFew-shotサンプルを使用
python main.py --dataset /path/to/dataset.json --model model_name --few-shot 3 --few-shot-path /path/to/examples.json
```

### 2. 大分類カテゴリ評価の実行

#### カテゴリ一覧の確認

```bash
# 利用可能なカテゴリの一覧を表示
python main_category.py --list-categories
```

#### 特定のカテゴリのみ評価

```bash
# GLP_翻訳カテゴリの評価
python main_category.py --category GLP_翻訳 --model model_name --output ./results

# ALT_制御性カテゴリの評価
python main_category.py --category ALT_制御性 --model model_name --output ./results
```

#### 複数カテゴリの評価

```bash
# カンマ区切りで複数カテゴリを指定
python main_category.py --category GLP_翻訳,GLP_推論,GLP_数学的推論 --model model_name --output ./results
```

#### 総合評価

```bash
# GLP全体の評価
python main_category.py --category GLP_AVG --model model_name --output ./results

# ALT全体の評価
python main_category.py --category ALT_AVG --model model_name --output ./results

# 総合評価（GLP + ALT）
python main_category.py --category TOTAL_AVG --model model_name --output ./results
```

#### 評価サンプル数の制限

```bash
# 各データセットを最大10サンプルで評価（開発時に便利）
python main_category.py --category GLP_翻訳 --model model_name --max-samples 10
```

詳細については [大分類評価機能 使用ガイド](docs/category_evaluation_guide.md) を参照してください。

### 3. カスタムメトリクスの作成と登録

#### ファイルベースの方法

1. `evaluator/metrics/custom/` ディレクトリに新しいPythonファイルを作成します（例: `my_metric.py`）
2. 以下のテンプレートを使用して実装します：

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
        super().__init__(name="my_custom_metric")
    
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
        # 例：単語の一致度を計算
        pred_words = set(predicted.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 1.0 if not pred_words else 0.0
        
        # Jaccard類似度を計算
        intersection = len(pred_words.intersection(ref_words))
        union = len(pred_words.union(ref_words))
        
        return intersection / union
```

詳細については [カスタムメトリクス作成ガイド](docs/custom_metrics_guide.md) を参照してください。

### 4. APIサーバーの起動と使用

```bash
# APIサーバーを起動
python -m evaluator.api --host 0.0.0.0 --port 8000

# 利用可能なメトリクスの一覧を取得
curl http://localhost:8000/metrics

# メトリクスの詳細情報を取得
curl http://localhost:8000/metrics/detail
```

## 評価カテゴリ

このプラットフォームでは、評価結果を以下の2つの主要カテゴリに分類しています：

### 汎用的言語処理能力（GLP）

表現、翻訳、情報検索、推論、数学的推論、抽出、知識・質問応答、英語能力、意味解析、構文解析の10カテゴリを評価します。

| カテゴリ | 説明 | 主なデータセット |
|---------|------|----------------|
| 表現 | 文章生成や表現力の評価 | roleplay, writing, humanities |
| 翻訳 | 日英・英日翻訳の精度 | alt-e-to-j, alt-j-to-e, wikicorpus-e-to-j, wikicorpus-j-to-e |
| 情報検索 | 情報の検索・抽出能力 | jsquad |
| 推論 | 論理的思考・推論能力 | reasoning |
| 数学的推論 | 数学問題解決能力 | mawps, mgsm, math |
| 抽出 | テキストからの情報抽出 | wiki_ner, wiki_coreference, chabsa, extraction |
| 知識・質問応答 | 一般知識と質問応答能力 | jcommonsenseqa, jemhopqa, jmmlu, niilc, aio, stem |
| 英語 | 英語処理能力 | mmlu_en |
| 意味解析 | テキストの意味理解 | jnli, janli, jsem, jsick, jamp |
| 構文解析 | 文法・構文の理解 | jcola-in-domain, jcola-out-of-domain, jblimp, wiki_reading, wiki_pas, wiki_dependency |

### アラインメント（ALT）

制御性、倫理・道徳、毒性、バイアス、堅牢性、真実性の6カテゴリを評価します。

| カテゴリ | 説明 | 主なデータセット |
|---------|------|----------------|
| 制御性 | 指示に従う能力 | jaster_control, lctg |
| 倫理・道徳 | 倫理的判断の適切さ | commonsensemoralja |
| 毒性 | 有害な出力の回避 | toxicity |
| バイアス | ステレオタイプやバイアスの有無 | jbbq |
| 堅牢性 | 異なる入力形式への対応 | jmmlu_robust |
| 真実性 | 事実に基づく回答の生成 | jtruthfulqa |

## 今後の開発計画

- **ダッシュボード機能**: 評価結果を可視化するためのダッシュボード機能を実装予定
- **より多くのモデルのサポート**: より多くのLLMモデル/APIの統合
- **分散評価**: 大規模評価のための分散処理機能

## ライセンス

MITライセンス

## 参考文献

- Jaster: Japanese large language model (LLM) scoring toolkit for evaluating reasoning
- JBBQ: Japanese Bias Benchmark for Question Answering
