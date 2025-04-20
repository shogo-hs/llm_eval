"""
カテゴリ定義モジュール

大分類評価カテゴリとデータセットの対応関係を定義するモジュール
"""
from typing import Dict, List, Set, Tuple, Optional, Union, Any

# 大分類カテゴリと小分類データセットのマッピング定義
CATEGORY_DATASETS_MAPPING = {
    # GLP（汎用的言語処理能力）カテゴリ
    "GLP_表現": [
        {"dataset": "mt-bench-roleplay", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]},
        {"dataset": "mt-bench-writing", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]}, 
        {"dataset": "mt-bench-humanities", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]}
    ],
    
    "GLP_翻訳": [
        {"dataset": "alt-e-to-j", "evaluator": "jaster", "metrics": ["bleu", "comet"]},
        {"dataset": "alt-j-to-e", "evaluator": "jaster", "metrics": ["bleu", "comet"]},
        {"dataset": "wikicorpus-e-to-j", "evaluator": "jaster", "metrics": ["bleu", "comet"]},
        {"dataset": "wikicorpus-j-to-e", "evaluator": "jaster", "metrics": ["bleu", "comet"]}
    ],
    
    "GLP_情報検索": [
        {"dataset": "jsquad", "evaluator": "jaster", "metrics": ["exact_match", "char_f1"]}
    ],
    
    "GLP_推論": [
        {"dataset": "mt-bench-reasoning", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]}
    ],
    
    "GLP_数学的推論": [
        {"dataset": "mawps", "evaluator": "jaster", "metrics": ["exact_match_figure"]},
        {"dataset": "mgsm", "evaluator": "jaster", "metrics": ["exact_match_figure"]},
        {"dataset": "mt-bench-math", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]}
    ],
    
    "GLP_抽出": [
        {"dataset": "wiki_ner", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "wiki_coreference", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "chabsa", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "mt-bench-extraction", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]}
    ],
    
    "GLP_知識・質問応答": [
        {"dataset": "jcommonsenseqa", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "jemhopqa", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "jmmlu", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "niilc", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "aio", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "mt-bench-stem", "evaluator": "mt-bench", "metrics": ["gpt4o_score"]}
    ],
    
    "GLP_英語": [
        {"dataset": "mmlu_en", "evaluator": "jaster", "metrics": ["exact_match"]}
    ],
    
    "GLP_意味解析": [
        {"dataset": "jnli", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "janli", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "jsem", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "jsick", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "jamp", "evaluator": "jaster", "metrics": ["exact_match"]}
    ],
    
    "GLP_構文解析": [
        {"dataset": "jcola-in-domain", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "jcola-out-of-domain", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "jblimp", "evaluator": "jaster", "metrics": ["exact_match"]},
        {"dataset": "wiki_reading", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "wiki_pas", "evaluator": "jaster", "metrics": ["char_f1"]},
        {"dataset": "wiki_dependency", "evaluator": "jaster", "metrics": ["char_f1"]}
    ],
    
    # ALT（アラインメント）カテゴリ
    "ALT_制御性": [
        {"dataset": "jaster_control", "evaluator": "jaster", "metrics": ["format_conformity"]},
        {"dataset": "lctg", "evaluator": "lctg", "metrics": ["quantitative_conformity"]}
    ],
    
    "ALT_倫理・道徳": [
        {"dataset": "commonsensemoralja", "evaluator": "jaster", "metrics": ["exact_match"]}
    ],
    
    "ALT_毒性": [
        {"dataset": "toxicity", "evaluator": "line_yahoo", "metrics": ["fairness", "social_norm", "prohibited_action", "violation_category"]}
    ],
    
    "ALT_バイアス": [
        {"dataset": "jbbq", "evaluator": "jbbq", "metrics": ["bias_score"]}
    ],
    
    "ALT_堅牢性": [
        {"dataset": "jmmlu_robust", "evaluator": "jaster", "metrics": ["consistency_score"]}
    ],
    
    "ALT_真実性": [
        {"dataset": "jtruthfulqa", "evaluator": "jtruthful", "metrics": ["roberta_score"]}
    ]
}

# 大分類カテゴリグループ定義
GLP_CATEGORIES = [
    "GLP_表現", "GLP_翻訳", "GLP_情報検索", "GLP_推論", "GLP_数学的推論",
    "GLP_抽出", "GLP_知識・質問応答", "GLP_英語", "GLP_意味解析", "GLP_構文解析"
]

ALT_CATEGORIES = [
    "ALT_制御性", "ALT_倫理・道徳", "ALT_毒性", "ALT_バイアス", "ALT_堅牢性", "ALT_真実性"
]

# 総合カテゴリ
AGGREGATE_CATEGORIES = {
    "GLP_AVG": GLP_CATEGORIES,
    "ALT_AVG": ALT_CATEGORIES,
    "TOTAL_AVG": GLP_CATEGORIES + ALT_CATEGORIES
}

# ショット数とデータセットの対応表
SHOT_DATASETS_MAPPING = {
    # 0-shotのみ評価するデータセット
    "zero_shot_only": [
        "mt-bench-roleplay", "mt-bench-writing", "mt-bench-humanities",
        "mt-bench-reasoning", "mt-bench-math", "mt-bench-extraction",
        "mt-bench-stem", "toxicity", "jtruthfulqa", "lctg"
    ],
    
    # 2-shotのみ評価するデータセット
    "two_shot_only": [
        "commonsensemoralja", "jbbq", "jmmlu_robust"
    ],
    
    # 0-shotと2-shotの両方で評価するデータセット（明示的に記載のないものはすべてここに含まれる）
    "both_shots": []
}

# 自動生成: both_shotsに明示的に他のリストに含まれていないデータセットを追加
for category_datasets in CATEGORY_DATASETS_MAPPING.values():
    for dataset_info in category_datasets:
        dataset_name = dataset_info["dataset"]
        if (dataset_name not in SHOT_DATASETS_MAPPING["zero_shot_only"] and 
            dataset_name not in SHOT_DATASETS_MAPPING["two_shot_only"]):
            SHOT_DATASETS_MAPPING["both_shots"].append(dataset_name)

# スコア正規化設定
SCORE_NORMALIZATION = {
    # MT-benchのスコアは10点満点のため10で割って正規化
    "gpt4o_score": {"divide_by": 10.0}
}

# 各データセットのサンプル数設定（llm_evaluation_metrics.mdに基づく）
DATASET_SAMPLE_COUNTS = {
    # MT-bench系はデフォルトで全データを使用する
    "mt-bench-roleplay": {"test": None, "dev": None},
    "mt-bench-writing": {"test": None, "dev": None},
    "mt-bench-humanities": {"test": None, "dev": None},
    "mt-bench-reasoning": {"test": None, "dev": None},
    "mt-bench-math": {"test": None, "dev": None},
    "mt-bench-extraction": {"test": None, "dev": None},
    "mt-bench-stem": {"test": None, "dev": None},
    
    # Jaster（wiki系）: 20サンプル（テスト）、5サンプル（開発）
    "wiki_ner": {"test": 20, "dev": 5},
    "wiki_coreference": {"test": 20, "dev": 5},
    "wiki_reading": {"test": 20, "dev": 5},
    "wiki_pas": {"test": 20, "dev": 5},
    "wiki_dependency": {"test": 20, "dev": 5},
    "wikicorpus-e-to-j": {"test": 20, "dev": 5},
    "wikicorpus-j-to-e": {"test": 20, "dev": 5},
    
    # Jaster（mmlu系）: 5サンプル（テスト）、1サンプル（開発）
    "jmmlu": {"test": 5, "dev": 1},
    "mmlu_en": {"test": 5, "dev": 1},
    
    # Jaster（wiki系とmmlu系以外）: 100サンプル（テスト）、10サンプル（開発）
    "alt-e-to-j": {"test": 100, "dev": 10},
    "alt-j-to-e": {"test": 100, "dev": 10},
    "jsquad": {"test": 100, "dev": 10},
    "mawps": {"test": 100, "dev": 10},
    "mgsm": {"test": 100, "dev": 10},
    "chabsa": {"test": 100, "dev": 10},
    "jcommonsenseqa": {"test": 100, "dev": 10},
    "jemhopqa": {"test": 100, "dev": 10},
    "niilc": {"test": 100, "dev": 10},
    "aio": {"test": 100, "dev": 10},
    "jnli": {"test": 100, "dev": 10},
    "janli": {"test": 100, "dev": 10},
    "jsem": {"test": 100, "dev": 10},
    "jsick": {"test": 100, "dev": 10},
    "jamp": {"test": 100, "dev": 10},
    "jcola-in-domain": {"test": 100, "dev": 10},
    "jcola-out-of-domain": {"test": 100, "dev": 10},
    "jblimp": {"test": 100, "dev": 10},
    "jaster_control": {"test": 100, "dev": 10},
    "commonsensemoralja": {"test": 100, "dev": 10},
    
    # その他特殊なデータセット
    "jbbq": {"test": 20, "dev": 4},
    "toxicity": {"test": None, "dev": None, "test_mode": 12},
    "lctg": {"test": 30, "dev": None},
    "jtruthfulqa": {"test": None, "dev": None},
    "jmmlu_robust": {"test": 5, "dev": 1}
}

class CategoryMapping:
    """カテゴリマッピング管理クラス"""
    
    @staticmethod
    def get_datasets_for_category(category: str) -> List[Dict[str, Any]]:
        """
        指定されたカテゴリに属するデータセット情報を取得する
        
        Args:
            category: カテゴリ名（GLP_xxxx、ALT_xxxx、GLP_AVG、ALT_AVG、TOTAL_AVG）
            
        Returns:
            List[Dict[str, Any]]: データセット情報のリスト
        
        Raises:
            ValueError: 未知のカテゴリが指定された場合
        """
        # 大分類カテゴリの場合
        if category in CATEGORY_DATASETS_MAPPING:
            return CATEGORY_DATASETS_MAPPING[category]
        
        # 総合カテゴリの場合
        elif category in AGGREGATE_CATEGORIES:
            # 総合カテゴリに含まれる全ての大分類カテゴリのデータセットを集約
            datasets = []
            for sub_category in AGGREGATE_CATEGORIES[category]:
                datasets.extend(CATEGORY_DATASETS_MAPPING[sub_category])
            return datasets
        
        else:
            raise ValueError(f"Unknown category: {category}")
    
    @staticmethod
    def get_all_categories() -> List[str]:
        """
        全てのカテゴリ名を取得する
        
        Returns:
            List[str]: 全てのカテゴリ名のリスト
        """
        return list(CATEGORY_DATASETS_MAPPING.keys()) + list(AGGREGATE_CATEGORIES.keys())
    
    @staticmethod
    def get_shot_counts_for_dataset(dataset: str) -> List[int]:
        """
        指定されたデータセットで評価すべきショット数を取得する
        
        Args:
            dataset: データセット名
            
        Returns:
            List[int]: ショット数のリスト（[0]、[2]、または[0, 2]）
        """
        if dataset in SHOT_DATASETS_MAPPING["zero_shot_only"]:
            return [0]
        elif dataset in SHOT_DATASETS_MAPPING["two_shot_only"]:
            return [2]
        else:
            return [0, 2]
    
    @staticmethod
    def normalize_score(metric: str, score: float) -> float:
        """
        評価スコアを正規化する
        
        Args:
            metric: メトリクス名
            score: 元のスコア
            
        Returns:
            float: 正規化されたスコア（0.0-1.0）
        """
        if metric in SCORE_NORMALIZATION:
            normalization = SCORE_NORMALIZATION[metric]
            if "divide_by" in normalization:
                return score / normalization["divide_by"]
        
        # 正規化設定がない場合はそのまま返す
        return score
    
    @staticmethod
    def get_sample_count_for_dataset(dataset: str, mode: str = "test") -> Optional[int]:
        """
        指定されたデータセットのサンプル数を取得する
        
        Args:
            dataset: データセット名
            mode: モード（"test" または "dev"、デフォルトは "test"）
            
        Returns:
            Optional[int]: サンプル数（None の場合は全サンプルを使用）
        """
        if dataset in DATASET_SAMPLE_COUNTS:
            if mode in DATASET_SAMPLE_COUNTS[dataset]:
                return DATASET_SAMPLE_COUNTS[dataset][mode]
            
        # 特別な処理: toxicity データセットのテストモード
        if dataset == "toxicity" and mode == "test" and "test_mode" in DATASET_SAMPLE_COUNTS["toxicity"]:
            return DATASET_SAMPLE_COUNTS["toxicity"]["test_mode"]
        
        # デフォルト値
        # wiki系データセット
        if any(wiki_key in dataset for wiki_key in ["wiki_", "wikicorpus"]):
            return 20 if mode == "test" else 5
            
        # mmlu系データセット
        elif any(mmlu_key in dataset for mmlu_key in ["mmlu", "jmmlu"]):
            return 5 if mode == "test" else 1
            
        # その他のJasterデータセット
        elif dataset not in ["toxicity", "lctg", "jtruthfulqa"] and "jaster" in dataset:
            return 100 if mode == "test" else 10
            
        # それ以外（MT-bench系など）
        return None
