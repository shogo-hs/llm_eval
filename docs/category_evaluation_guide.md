# 大分類評価機能 使用ガイド

## 概要

このドキュメントでは、LLM評価プラットフォームの大分類評価機能の使用方法について説明します。大分類評価機能は、小分類（個別データセット）の評価結果を集約し、より高レベルな観点からLLMの性能を評価することができます。

主な機能：

1. 大分類カテゴリ（GLP_xxxx、ALT_xxxx）による評価
2. 総合評価（GLP_AVG、ALT_AVG、TOTAL_AVG）の算出
3. ショット数（0-shot/2-shot）や評価サンプル数の制御
4. 既存評価結果の再利用
5. 階層的な評価結果の表示と保存

## 評価カテゴリ

システムは以下の2つの大きな評価軸と、それぞれに属する大分類カテゴリをサポートしています：

### 汎用的言語処理能力（GLP）カテゴリ

| カテゴリ | 説明 | データセット数 |
|---------|------|--------------|
| GLP_表現 | 文章生成や表現力の評価 | 3 |
| GLP_翻訳 | 日英・英日翻訳の精度 | 4 |
| GLP_情報検索 | 情報の検索・抽出能力 | 1 |
| GLP_推論 | 論理的思考・推論能力 | 1 |
| GLP_数学的推論 | 数学問題解決能力 | 3 |
| GLP_抽出 | テキストからの情報抽出 | 4 |
| GLP_知識・質問応答 | 一般知識と質問応答能力 | 6 |
| GLP_英語 | 英語処理能力 | 1 |
| GLP_意味解析 | テキストの意味理解 | 5 |
| GLP_構文解析 | 文法・構文の理解 | 6 |

### アラインメント（ALT）カテゴリ

| カテゴリ | 説明 | データセット数 |
|---------|------|--------------|
| ALT_制御性 | 指示に従う能力 | 2 |
| ALT_倫理・道徳 | 倫理的判断の適切さ | 1 |
| ALT_毒性 | 有害な出力の回避 | 1 |
| ALT_バイアス | ステレオタイプやバイアスの有無 | 1 |
| ALT_堅牢性 | 異なる入力形式への対応 | 1 |
| ALT_真実性 | 事実に基づく回答の生成 | 1 |

### 総合カテゴリ

総合カテゴリは、複数の大分類カテゴリの評価結果を集約した評価指標です：

- **GLP_AVG**: 全GLP大分類カテゴリの平均値
- **ALT_AVG**: 全ALT大分類カテゴリの平均値
- **TOTAL_AVG**: GLPとALTの平均値（(GLP_AVG + ALT_AVG) / 2）

## 基本的な使い方

### 1. カテゴリ一覧の確認

```bash
# 利用可能なカテゴリの一覧を表示
python main_category.py --list-categories
```

### 2. 特定のカテゴリのみ評価

```bash
# GLP_翻訳カテゴリの評価
python main_category.py --category GLP_翻訳 --model model_name --output ./results

# ALT_制御性カテゴリの評価
python main_category.py --category ALT_制御性 --model model_name --output ./results
```

### 3. 複数カテゴリの評価

```bash
# カンマ区切りで複数カテゴリを指定
python main_category.py --category GLP_翻訳,GLP_推論,GLP_数学的推論 --model model_name --output ./results
```

### 4. 総合評価

```bash
# GLP全体の評価
python main_category.py --category GLP_AVG --model model_name --output ./results

# ALT全体の評価
python main_category.py --category ALT_AVG --model model_name --output ./results

# 総合評価（GLP + ALT）
python main_category.py --category TOTAL_AVG --model model_name --output ./results
```

## 詳細設定

### サンプル数の制限

開発時やテスト時は、サンプル数を制限して評価時間を短縮できます：

```bash
# 各データセットを最大10サンプルで評価
python main_category.py --category GLP_翻訳 --model model_name --max-samples 10
```

### データセットとFew-shotパスの指定

カスタムデータセットやFew-shotサンプルを使用する場合：

```bash
# データセットとFew-shotディレクトリを指定
python main_category.py --category GLP_翻訳 \
  --dataset-dir /path/to/datasets \
  --few-shot-dir /path/to/few_shots \
  --model model_name
```

### 既存評価結果の再利用

デフォルトでは既存の評価結果を再利用しますが、常に再評価したい場合：

```bash
# 既存結果を再利用せず、常に再評価
python main_category.py --category GLP_翻訳 --model model_name --no-reuse
```

### API設定のカスタマイズ

```bash
# APIのURLとパラメータをカスタマイズ
python main_category.py --category GLP_翻訳 \
  --api-url "http://localhost:8000/v1/chat/completions" \
  --temperature 0.5 \
  --max-tokens 1000 \
  --model model_name
```

## 評価結果の解釈

評価結果は以下のような階層構造で保存されます：

1. **カテゴリレベル**: カテゴリごとに集計されたスコア
2. **データセットレベル**: 各データセットのショット数ごとの評価結果
3. **サンプルレベル**: 個々のサンプルに対する予測と評価スコア

### スコア正規化

- MT-benchのスコアは10点満点のため、0.0-1.0の範囲に正規化されます（10で割る）
- 他のメトリクスは基本的に0.0-1.0の範囲内で出力されるため、そのまま使用されます

### 0-shot/2-shotの平均

- 多くのデータセットでは0-shotと2-shotの両方で評価され、その平均値が使用されます
- 一部のデータセットは0-shotのみ、または2-shotのみで評価されます

## コマンドライン引数一覧

| 引数 | 説明 | デフォルト値 |
|------|------|------------|
| `--category`, `-c` | 評価するカテゴリ（カンマ区切りで複数指定可能） | (必須) |
| `--model`, `-m` | 評価対象のモデル名 | "elyza" |
| `--output`, `-o` | 評価結果の出力ディレクトリ | "./results" |
| `--dataset-dir`, `-d` | データセットベースディレクトリ | None (デフォルトパス) |
| `--few-shot-dir`, `-f` | Few-shotサンプルベースディレクトリ | None (デフォルトパス) |
| `--max-samples`, `-s` | 最大サンプル数 | None (全サンプル) |
| `--batch-size`, `-b` | バッチサイズ | 5 |
| `--api-url` | ローカルLLM API URL | "http://192.168.3.43:8000/v1/chat/completions" |
| `--temperature`, `-t` | 生成の温度 | 0.7 |
| `--max-tokens` | 最大生成トークン数 | 500 |
| `--no-reuse` | 既存の評価結果を再利用しない | False |
| `--list-categories` | 利用可能なカテゴリの一覧を表示 | False |
| `--debug` | デバッグモード（詳細なログ出力） | False |

## 実装詳細

大分類評価機能は以下のコンポーネントで構成されています：

1. **カテゴリ定義モジュール**（`category_mapping.py`）：
   - 大分類カテゴリとデータセットの対応関係を定義
   - ショット数の設定やスコア正規化の管理

2. **カテゴリ評価マネージャー**（`category_evaluator.py`）：
   - 大分類カテゴリの評価実行と結果集約
   - 既存評価結果の再利用機能

3. **コマンドラインインターフェース**（`main_category.py`）：
   - カテゴリ指定や評価設定の管理
   - 結果のフォーマットと表示

## 注意事項と制限

1. **評価時間**: 大分類カテゴリには多数のデータセットが含まれるため、評価には長時間かかる場合があります。開発時は `--max-samples` オプションを使用して時間を短縮してください。

2. **データセットとFew-shotサンプル**: 評価にはデータセットとFew-shotサンプルが必要です。カスタムパスを指定しない場合は、デフォルトのパス（`data/datasets` と `data/few_shots`）が使用されます。

3. **API接続**: LLM評価にはAPI接続が必要です。デフォルトではローカルAPIを想定していますが、`--api-url` オプションで変更できます。

4. **結果の保存**: 評価結果は指定された出力ディレクトリに保存されます。既存の結果ファイルが上書きされることはありません。

## トラブルシューティング

1. **データセットが見つからない**

```
データセット評価エラー: alt-e-to-j (0-shot): No such file or directory: 'data/datasets/alt-e-to-j.json'
```

この場合、`--dataset-dir` オプションでデータセットディレクトリを指定してください。

2. **APIエラー**

```
データセット評価エラー: jsquad (2-shot): API connection failed: Connection refused
```

API URLが正しいことを確認し、必要に応じて `--api-url` オプションで指定してください。

3. **メモリ不足**

評価中にメモリ不足エラーが発生した場合は、`--max-samples` や `--batch-size` を調整して使用メモリを削減してください。

## 参考情報

- カテゴリとデータセットの詳細については `evaluator/category_mapping.py` を参照
- 評価アルゴリズムの詳細については `evaluator/category_evaluator.py` を参照
- スコア計算方法の詳細については `docs/llm_evaluation_metrics.md` を参照
