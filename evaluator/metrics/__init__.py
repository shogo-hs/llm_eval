"""
評価指標モジュール

各種評価指標の実装を提供する
"""
# 組み込み評価指標をインポート
from .exact_match import ExactMatch
from .exact_match_figure import ExactMatchFigure
from .char_f1 import CharF1
from .set_f1 import SetF1
from .bleu import BLEUScore
from .contains_answer import ContainsAnswer

# バージョン情報
__version__ = "1.0.0"
