"""
評価指標ローダーモジュール

ディレクトリベースのプラグインシステムで評価指標を動的に読み込む
"""
import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional, Any, Set
import logging

from .base import BaseMetric
from .metrics_db import get_metrics_db, MetricsDatabase

# ロガーの設定
logger = logging.getLogger(__name__)


class MetricsLoader:
    """
    評価指標ローダークラス
    
    ディレクトリからBaseMetricを継承した評価指標クラスを動的に読み込む
    """
    
    def __init__(self, base_dir: Optional[Path] = None, db: Optional[MetricsDatabase] = None):
        """
        初期化メソッド
        
        Args:
            base_dir: 評価指標の基本ディレクトリパス (Noneの場合はデフォルトパスを使用)
            db: 評価指標データベース (Noneの場合はデフォルトインスタンスを使用)
        """
        if base_dir is None:
            # デフォルトパスはパッケージディレクトリ内の metrics ディレクトリ
            base_dir = Path(__file__).parent / "metrics"
        
        self.base_dir = base_dir
        self.metrics_db = db if db is not None else get_metrics_db()
        self.custom_dir = self.base_dir / "custom"
        
        # ディレクトリが存在することを確認
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        
        # 読み込まれた評価指標クラスのマップ
        self._metric_classes: Dict[str, Type[BaseMetric]] = {}
    
    def load_all_metrics(self) -> Dict[str, Type[BaseMetric]]:
        """
        すべての評価指標を読み込む
        
        Returns:
            Dict[str, Type[BaseMetric]]: 評価指標名とクラスのマップ
        """
        # 組み込み評価指標を読み込む
        self._load_builtin_metrics()
        
        # カスタム評価指標を読み込む
        self._load_custom_metrics()
        
        # データベースから追加のカスタム評価指標を読み込む
        self._load_metrics_from_database()
        
        return self._metric_classes
    
    def _load_builtin_metrics(self):
        """
        組み込み評価指標をディレクトリから読み込む
        """
        try:
            # metrics__init__.pyファイルがあることを確認
            init_path = self.base_dir / "__init__.py"
            if not init_path.exists():
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write('"""評価指標モジュール"""\n')
            
            # metrics ディレクトリをパッケージとしてインポート
            package_name = f"evaluator.metrics"
            
            # metrics ディレクトリ内の .py ファイルを処理
            for py_file in self.base_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # モジュール名
                    module_name = py_file.stem
                    full_module_name = f"{package_name}.{module_name}"
                    
                    # モジュールをインポート
                    module = importlib.import_module(full_module_name)
                    
                    # モジュール内のBaseMetricを継承したクラスを探す
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseMetric) and 
                            obj != BaseMetric):
                            
                            # インスタンスを一時的に作成して名前を取得
                            try:
                                metric_instance = obj()
                                metric_name = metric_instance.name
                                
                                # クラスをマップに追加
                                self._metric_classes[metric_name] = obj
                                
                                # データベースに登録
                                self.metrics_db.register_builtin_metric(
                                    obj, module_name=module_name
                                )
                                
                                logger.info(f"組み込み評価指標を読み込みました: {metric_name}")
                            except Exception as e:
                                logger.error(f"評価指標のインスタンス化エラー {name}: {e}")
                
                except Exception as e:
                    logger.error(f"モジュール読み込みエラー {py_file.name}: {e}")
        
        except Exception as e:
            logger.error(f"組み込み評価指標の読み込み中にエラーが発生: {e}")
    
    def _load_custom_metrics(self):
        """
        カスタム評価指標をディレクトリから読み込む
        """
        try:
            # custom/__init__.pyファイルがあることを確認
            init_path = self.custom_dir / "__init__.py"
            if not init_path.exists():
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write('"""カスタム評価指標モジュール"""\n')
            
            # custom ディレクトリをパッケージとしてインポート
            package_name = f"evaluator.metrics.custom"
            
            # custom ディレクトリ内の .py ファイルを処理
            for py_file in self.custom_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # モジュール名
                    module_name = py_file.stem
                    full_module_name = f"{package_name}.{module_name}"
                    
                    # モジュールをインポート
                    spec = importlib.util.spec_from_file_location(full_module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    sys.modules[full_module_name] = module
                    
                    # モジュール内のBaseMetricを継承したクラスを探す
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseMetric) and 
                            obj != BaseMetric):
                            
                            # インスタンスを一時的に作成して名前を取得
                            try:
                                metric_instance = obj()
                                metric_name = metric_instance.name
                                
                                # クラスをマップに追加
                                self._metric_classes[metric_name] = obj
                                
                                # ソースコードを取得
                                code = inspect.getsource(obj)
                                
                                # データベースに登録
                                self.metrics_db.register_custom_metric(
                                    metric_name, 
                                    code, 
                                    description=obj.__doc__
                                )
                                
                                logger.info(f"カスタム評価指標を読み込みました: {metric_name}")
                            except Exception as e:
                                logger.error(f"評価指標のインスタンス化エラー {name}: {e}")
                
                except Exception as e:
                    logger.error(f"モジュール読み込みエラー {py_file.name}: {e}")
        
        except Exception as e:
            logger.error(f"カスタム評価指標の読み込み中にエラーが発生: {e}")
    
    def _load_metrics_from_database(self):
        """
        データベースからカスタム評価指標を読み込む
        """
        try:
            # データベースからカスタム評価指標を取得
            metrics = self.metrics_db.get_all_metrics(custom_only=True)
            
            for metric_info in metrics:
                try:
                    metric_name = metric_info["name"]
                    
                    # すでに読み込まれている場合はスキップ
                    if metric_name in self._metric_classes:
                        continue
                    
                    # 評価指標クラスを動的に生成
                    metric_class = self.metrics_db.create_metric_class(metric_name)
                    
                    if metric_class:
                        # クラスをマップに追加
                        self._metric_classes[metric_name] = metric_class
                        logger.info(f"DBからカスタム評価指標を読み込みました: {metric_name}")
                
                except Exception as e:
                    logger.error(f"DBからの評価指標読み込みエラー {metric_info['name']}: {e}")
        
        except Exception as e:
            logger.error(f"DBからのカスタム評価指標の読み込み中にエラーが発生: {e}")
    
    def add_custom_metric(self, 
                          name: str, 
                          code: str, 
                          description: str = None,
                          created_by: str = None,
                          save_to_file: bool = False) -> bool:
        """
        新しいカスタム評価指標を追加する
        
        Args:
            name: 評価指標名
            code: 評価指標のPythonコード
            description: 評価指標の説明
            created_by: 作成者
            save_to_file: ファイルにも保存するかどうか
            
        Returns:
            bool: 追加成功の場合はTrue、失敗の場合はFalse
        """
        try:
            # データベースに登録
            if not self.metrics_db.register_custom_metric(
                name, code, description, created_by
            ):
                return False
            
            # ファイルに保存
            if save_to_file:
                module_name = f"{name.lower().replace(' ', '_')}"
                file_path = self.custom_dir / f"{module_name}.py"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
            
            # 評価指標クラスを動的に生成
            metric_class = self.metrics_db.create_metric_class(name)
            
            if metric_class:
                # クラスをマップに追加
                self._metric_classes[name] = metric_class
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"カスタム評価指標の追加中にエラーが発生: {e}")
            return False
    
    def get_metric_class(self, metric_name: str) -> Optional[Type[BaseMetric]]:
        """
        評価指標名からクラスを取得する
        
        Args:
            metric_name: 評価指標名
            
        Returns:
            Optional[Type[BaseMetric]]: 評価指標クラス (存在しない場合はNone)
        """
        if metric_name in self._metric_classes:
            return self._metric_classes[metric_name]
        
        # まだ読み込まれていない場合はデータベースから取得を試みる
        metric_class = self.metrics_db.create_metric_class(metric_name)
        
        if metric_class:
            self._metric_classes[metric_name] = metric_class
            return metric_class
        
        return None
    
    def create_metric(self, metric_name: str, **kwargs) -> Optional[BaseMetric]:
        """
        評価指標名からインスタンスを生成する
        
        Args:
            metric_name: 評価指標名
            **kwargs: 評価指標の初期化パラメータ
            
        Returns:
            Optional[BaseMetric]: 評価指標インスタンス (失敗した場合はNone)
        """
        metric_class = self.get_metric_class(metric_name)
        
        if metric_class:
            try:
                return metric_class(**kwargs)
            except Exception as e:
                logger.error(f"評価指標インスタンスの生成中にエラーが発生: {e}")
        
        return None
    
    def list_metrics(self) -> List[str]:
        """
        利用可能な評価指標名のリストを取得する
        
        Returns:
            List[str]: 評価指標名のリスト
        """
        return list(self._metric_classes.keys())


# グローバルなローダーインスタンス
_loader_instance = None

def get_metrics_loader() -> MetricsLoader:
    """
    メトリクスローダーのシングルトンインスタンスを取得する
    
    Returns:
        MetricsLoader: ローダーインスタンス
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = MetricsLoader()
        # 初期化時に全ての評価指標を読み込む
        _loader_instance.load_all_metrics()
    return _loader_instance


def create_metric(metric_name: str, **kwargs) -> BaseMetric:
    """
    評価指標名からインスタンスを生成するユーティリティ関数
    
    Args:
        metric_name: 評価指標名
        **kwargs: 評価指標の初期化パラメータ
        
    Returns:
        BaseMetric: 評価指標インスタンス
        
    Raises:
        ValueError: 未サポートの評価指標名が指定された場合
    """
    loader = get_metrics_loader()
    metric = loader.create_metric(metric_name, **kwargs)
    
    if metric is None:
        raise ValueError(f"Unsupported metric: {metric_name}")
    
    return metric


def create_metrics_from_list(metric_names: List[str], **kwargs) -> List[BaseMetric]:
    """
    評価指標名のリストからインスタンスのリストを生成するユーティリティ関数
    
    Args:
        metric_names: 評価指標名のリスト
        **kwargs: 評価指標の初期化パラメータ
        
    Returns:
        List[BaseMetric]: 評価指標インスタンスのリスト
    """
    return [create_metric(name, **kwargs) for name in metric_names]


def list_available_metrics() -> List[str]:
    """
    利用可能な評価指標名のリストを取得するユーティリティ関数
    
    Returns:
        List[str]: 評価指標名のリスト
    """
    loader = get_metrics_loader()
    return loader.list_metrics()
