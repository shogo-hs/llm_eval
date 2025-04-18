"""
評価指標ファクトリーモジュール

指標名から評価指標インスタンスを生成する機能を提供する
"""
from typing import Dict, List, Type, Optional, Any
import logging

from .base import BaseMetric
from .metrics_loader import get_metrics_loader, create_metric, create_metrics_from_list, list_available_metrics

# ロガーの設定
logger = logging.getLogger(__name__)


class MetricFactory:
    """
    評価指標ファクトリー
    
    指標名から評価指標インスタンスを生成する
    """
    
    @classmethod
    def create(cls, metric_name: str, **kwargs) -> BaseMetric:
        """
        評価指標インスタンスを生成する
        
        Args:
            metric_name: 評価指標名
            **kwargs: 評価指標の初期化パラメータ
            
        Returns:
            BaseMetric: 評価指標インスタンス
        
        Raises:
            ValueError: 未サポートの評価指標名が指定された場合
        """
        return create_metric(metric_name, **kwargs)
    
    @classmethod
    def create_from_list(cls, metric_names: List[str], dataset_type: str = None, **kwargs) -> List[BaseMetric]:
        """
        評価指標インスタンスのリストを生成する
        
        Args:
            metric_names: 評価指標名のリスト
            dataset_type: データセットタイプ（オプション）
            **kwargs: 評価指標の初期化パラメータ
            
        Returns:
            List[BaseMetric]: 評価指標インスタンスのリスト
        """
        metrics = create_metrics_from_list(metric_names, **kwargs)
        
        # JBBQデータセットの場合、バイアス関連のメトリクスを追加
        if dataset_type == "jbbq" or any(m.name == "acc_diff" for m in metrics):
            # 既存のメトリクス名を取得
            existing_metrics = {m.name for m in metrics}
            
            # 必要なメトリクスを追加
            if "acc_diff" not in existing_metrics:
                metrics.append(create_metric("acc_diff"))
            if "bias_score_dis" not in existing_metrics:
                metrics.append(create_metric("bias_score_dis"))
            if "bias_score_amb" not in existing_metrics:
                metrics.append(create_metric("bias_score_amb"))
            if "bias_score_avg" not in existing_metrics:
                metrics.append(create_metric("bias_score_avg"))
        
        return metrics
    
    @classmethod
    def register(cls, metric_name: str, metric_class: type):
        """
        評価指標クラスを登録する
        
        Args:
            metric_name: 評価指標名
            metric_class: 評価指標クラス
        """
        # ローダーを使用して登録
        loader = get_metrics_loader()
        
        # 一時的にインスタンス化して説明を取得
        try:
            instance = metric_class()
            description = metric_class.__doc__
            
            # DB登録のためにソースコードを取得
            import inspect
            code = inspect.getsource(metric_class)
            
            # DBに登録
            loader.metrics_db.register_custom_metric(
                name=metric_name,
                code=code,
                description=description
            )
            
            # メモリ上のマップにも追加
            loader._metric_classes[metric_name] = metric_class
            
            logger.info(f"評価指標を登録しました: {metric_name}")
        except Exception as e:
            logger.error(f"評価指標の登録中にエラーが発生: {e}")
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """
        利用可能な評価指標名のリストを取得する
        
        Returns:
            List[str]: 評価指標名のリスト
        """
        return list_available_metrics()
    
    @classmethod
    def add_custom_metric(cls, 
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
        loader = get_metrics_loader()
        return loader.add_custom_metric(
            name=name,
            code=code,
            description=description,
            created_by=created_by,
            save_to_file=save_to_file
        )
