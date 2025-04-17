"""
評価指標データベース管理モジュール

評価指標をデータベースで管理するための機能を提供する
"""
import os
import json
import importlib.util
import sys
import tempfile
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional, Union, Type, Tuple
import inspect
import logging
from datetime import datetime

from .base import BaseMetric

# ロガーの設定
logger = logging.getLogger(__name__)


class MetricsDatabase:
    """
    評価指標データベース管理クラス
    
    SQLiteを使用して評価指標を永続化管理する
    """
    
    def __init__(self, db_path: Union[str, Path] = None):
        """
        初期化メソッド
        
        Args:
            db_path: データベースファイルパス (Noneの場合はデフォルトパスを使用)
        """
        if db_path is None:
            # デフォルトパスはパッケージディレクトリ内の metrics.db
            package_dir = Path(__file__).parent
            db_path = package_dir / "data" / "metrics.db"
        
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        
        # データベースディレクトリが存在しない場合は作成
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # データベース接続
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # テーブル初期化
        self._initialize_tables()
    
    def _initialize_tables(self):
        """
        データベーステーブルを初期化する
        """
        cursor = self.conn.cursor()
        
        # metrics テーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            code TEXT NOT NULL,
            description TEXT,
            is_builtin BOOLEAN NOT NULL DEFAULT 0,
            module_name TEXT,
            class_name TEXT,
            created_by TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            version TEXT DEFAULT '1.0.0',
            is_active BOOLEAN NOT NULL DEFAULT 1
        )
        ''')
        
        # metric_parameters テーブル (オプションパラメータの管理)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metric_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_id INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            param_type TEXT NOT NULL,
            default_value TEXT,
            description TEXT,
            required BOOLEAN NOT NULL DEFAULT 0,
            FOREIGN KEY (metric_id) REFERENCES metrics (id) ON DELETE CASCADE
        )
        ''')
        
        self.conn.commit()
    
    def register_builtin_metric(self, 
                                metric_class: Type[BaseMetric], 
                                module_name: str = None,
                                description: str = None) -> bool:
        """
        組み込み評価指標をデータベースに登録する
        
        Args:
            metric_class: 評価指標クラス
            module_name: モジュール名 (Noneの場合はクラスから自動取得)
            description: 評価指標の説明
            
        Returns:
            bool: 登録成功の場合はTrue、失敗の場合はFalse
        """
        try:
            # クラスをインスタンス化してメタデータを取得
            metric_instance = metric_class()
            metric_name = metric_instance.name
            
            # モジュール名が指定されていない場合はクラスから取得
            if module_name is None:
                module_name = metric_class.__module__.split('.')[-1]
            
            class_name = metric_class.__name__
            
            # ソースコードを取得
            try:
                code = inspect.getsource(metric_class)
            except Exception:
                # ソースコード取得に失敗した場合は空文字列
                code = ""
            
            # 説明が指定されていない場合はクラスのdocstringを使用
            if description is None and metric_class.__doc__:
                description = inspect.cleandoc(metric_class.__doc__)
            
            cursor = self.conn.cursor()
            
            # 既に登録されている場合は更新、そうでない場合は挿入
            cursor.execute(
                """
                INSERT INTO metrics 
                (name, code, description, is_builtin, module_name, class_name, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                code = ?, description = ?, is_builtin = 1, module_name = ?, class_name = ?,
                updated_at = CURRENT_TIMESTAMP
                """,
                (metric_name, code, description, module_name, class_name,
                 code, description, module_name, class_name)
            )
            
            self.conn.commit()
            logger.info(f"組み込み評価指標を登録しました: {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"組み込み評価指標の登録中にエラーが発生: {e}")
            return False
    
    def register_custom_metric(self, 
                               name: str, 
                               code: str, 
                               description: str = None,
                               created_by: str = None,
                               version: str = "1.0.0") -> bool:
        """
        カスタム評価指標をデータベースに登録する
        
        Args:
            name: 評価指標名
            code: 評価指標のPythonコード
            description: 評価指標の説明
            created_by: 作成者
            version: バージョン
            
        Returns:
            bool: 登録成功の場合はTrue、失敗の場合はFalse
        """
        try:
            # コードの検証 (BaseMetricを継承したクラスが含まれているか)
            if not self._validate_metric_code(code):
                raise ValueError("評価指標コードには BaseMetric を継承したクラスが必要です")
            
            cursor = self.conn.cursor()
            
            # 既に登録されている場合は更新、そうでない場合は挿入
            cursor.execute(
                """
                INSERT INTO metrics 
                (name, code, description, is_builtin, created_by, version, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                code = ?, description = ?, created_by = ?, version = ?,
                updated_at = CURRENT_TIMESTAMP
                """,
                (name, code, description, created_by, version,
                 code, description, created_by, version)
            )
            
            self.conn.commit()
            logger.info(f"カスタム評価指標を登録しました: {name}")
            return True
            
        except Exception as e:
            logger.error(f"カスタム評価指標の登録中にエラーが発生: {e}")
            return False
    
    def get_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        評価指標の情報を取得する
        
        Args:
            metric_name: 評価指標名
            
        Returns:
            Optional[Dict[str, Any]]: 評価指標情報 (存在しない場合はNone)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM metrics WHERE name = ? AND is_active = 1",
            (metric_name,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return dict(row)
    
    def get_all_metrics(self, 
                        include_inactive: bool = False, 
                        builtin_only: bool = False,
                        custom_only: bool = False) -> List[Dict[str, Any]]:
        """
        全ての評価指標情報を取得する
        
        Args:
            include_inactive: 非アクティブな評価指標も含めるかどうか
            builtin_only: 組み込み評価指標のみを取得するかどうか
            custom_only: カスタム評価指標のみを取得するかどうか
            
        Returns:
            List[Dict[str, Any]]: 評価指標情報のリスト
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM metrics"
        conditions = []
        
        if not include_inactive:
            conditions.append("is_active = 1")
        
        if builtin_only:
            conditions.append("is_builtin = 1")
        elif custom_only:
            conditions.append("is_builtin = 0")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def delete_metric(self, metric_name: str) -> bool:
        """
        評価指標を削除する (論理削除)
        
        Args:
            metric_name: 評価指標名
            
        Returns:
            bool: 削除成功の場合はTrue、失敗の場合はFalse
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE metrics SET is_active = 0 WHERE name = ?",
                (metric_name,)
            )
            self.conn.commit()
            logger.info(f"評価指標を削除しました: {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"評価指標の削除中にエラーが発生: {e}")
            return False
    
    def _validate_metric_code(self, code: str) -> bool:
        """
        評価指標コードを検証する
        
        BaseMetricを継承したクラスが含まれているかどうかをチェック
        
        Args:
            code: 評価指標のPythonコード
            
        Returns:
            bool: 検証成功の場合はTrue、失敗の場合はFalse
        """
        try:
            # 一時ファイルを作成してコードを保存
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                # 必要なインポートを追加
                full_code = """
from evaluator.base import BaseMetric
import re
import math
import numpy as np
from typing import Optional, List, Dict, Any, Union

""" + code
                temp.write(full_code.encode('utf-8'))
                temp_path = temp.name
            
            # モジュールを動的にインポート
            module_name = f"temp_metric_{datetime.now().timestamp()}"
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # BaseMetricを継承したクラスを探す
            has_metric_class = False
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseMetric) and 
                    obj != BaseMetric):
                    has_metric_class = True
                    break
            
            # 一時ファイルを削除
            os.unlink(temp_path)
            
            return has_metric_class
            
        except Exception as e:
            logger.error(f"評価指標コードの検証中にエラーが発生: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return False
    
    def create_metric_class(self, metric_name: str) -> Optional[Type[BaseMetric]]:
        """
        評価指標名からクラスを動的に生成する
        
        Args:
            metric_name: 評価指標名
            
        Returns:
            Optional[Type[BaseMetric]]: 評価指標クラス (失敗した場合はNone)
        """
        metric_info = self.get_metric(metric_name)
        if not metric_info:
            logger.error(f"評価指標が見つかりません: {metric_name}")
            return None
        
        try:
            if metric_info["is_builtin"]:
                # 組み込み評価指標はモジュールからインポート
                module_name = f"evaluator.metrics"
                if metric_info["module_name"]:
                    module_name = f"evaluator.{metric_info['module_name']}"
                
                module = importlib.import_module(module_name)
                return getattr(module, metric_info["class_name"])
            else:
                # カスタム評価指標は動的に生成
                # 一時ファイルを作成してコードを保存
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                    # 必要なインポートを追加
                    full_code = """
from evaluator.base import BaseMetric
import re
import math
import numpy as np
from typing import Optional, List, Dict, Any, Union

""" + metric_info["code"]
                    temp.write(full_code.encode('utf-8'))
                    temp_path = temp.name
                
                # モジュールを動的にインポート
                module_name = f"custom_metric_{metric_name}"
                spec = importlib.util.spec_from_file_location(module_name, temp_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # BaseMetricを継承したクラスを探す
                metric_class = None
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseMetric) and 
                        obj != BaseMetric):
                        metric_class = obj
                        break
                
                # 一時ファイルを削除
                os.unlink(temp_path)
                
                return metric_class
                
        except Exception as e:
            logger.error(f"評価指標クラスの生成中にエラーが発生: {e}")
            return None
    
    def close(self):
        """
        データベース接続を閉じる
        """
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """
        デストラクタ
        """
        self.close()


# グローバルなデータベースインスタンス
_db_instance = None

def get_metrics_db() -> MetricsDatabase:
    """
    メトリクスデータベースのシングルトンインスタンスを取得する
    
    Returns:
        MetricsDatabase: データベースインスタンス
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = MetricsDatabase()
    return _db_instance
