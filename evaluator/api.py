# データモデル
class MetricInfo(BaseModel):
    """評価指標の情報"""
    name: str = Field(..., description="評価指標の名前")
    description: Optional[str] = Field(None, description="評価指標の説明")
    is_builtin: bool = Field(..., description="組み込み評価指標かどうか")
    created_by: Optional[str] = Field(None, description="作成者")
    created_at: Optional[str] = Field(None, description="作成日時")
    updated_at: Optional[str] = Field(None, description="更新日時")
    version: Optional[str] = Field(None, description="バージョン")
    is_active: bool = Field(..., description="アクティブかどうか")


class CustomMetricRequest(BaseModel):
    """カスタム評価指標の登録リクエスト"""
    name: str = Field(..., description="評価指標の名前")
    code: str = Field(..., description="評価指標のPythonコード")
    description: Optional[str] = Field(None, description="評価指標の説明")
    created_by: Optional[str] = Field(None, description="作成者")
    save_to_file: bool = Field(False, description="ファイルにも保存するかどうか")


class CustomMetricResponse(BaseModel):
    """カスタム評価指標の登録レスポンス"""
    success: bool = Field(..., description="成功したかどうか")
    message: str = Field(..., description="メッセージ")
    metric_name: Optional[str] = Field(None, description="登録された評価指標名")


class MetricsListResponse(BaseModel):
    """評価指標リストのレスポンス"""
    metrics: List[str] = Field(..., description="評価指標名のリスト")
    count: int = Field(..., description="評価指標の数")


class MetricsDetailResponse(BaseModel):
    """評価指標詳細のレスポンス"""
    metrics: List[MetricInfo] = Field(..., description="評価指標情報のリスト")
    count: int = Field(..., description="評価指標の数")


class ValidateCodeRequest(BaseModel):
    """コード検証リクエスト"""
    code: str = Field(..., description="検証するPythonコード")


class ValidateCodeResponse(BaseModel):
    """コード検証レスポンス"""
    valid: bool = Field(..., description="有効なコードかどうか")
    metric_name: Optional[str] = Field(None, description="検出された評価指標名")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class MetricCodeResponse(BaseModel):
    """評価指標コードレスポンス"""
    name: str = Field(..., description="評価指標名")
    code: str = Field(..., description="評価指標のコード")


# FastAPIアプリケーション
app = FastAPI(
    title="評価指標管理API",
    description="LLM評価指標を管理するためのAPI",
    version="1.0.0"
)

# CORSミドルウェアの追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンのみを許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def read_root():
    """ルートエンドポイント"""
    return {"message": "評価指標管理APIへようこそ"}


@app.get("/metrics", response_model=MetricsListResponse, tags=["Metrics"])
async def list_metrics():
    """
    利用可能な評価指標の一覧を取得する
    """
    metrics = MetricFactory.list_metrics()
    return {"metrics": metrics, "count": len(metrics)}


@app.get("/metrics/detail", response_model=MetricsDetailResponse, tags=["Metrics"])
async def get_metrics_detail(
    include_inactive: bool = False,
    builtin_only: bool = False,
    custom_only: bool = False
):
    """
    評価指標の詳細情報を取得する
    """
    db = get_metrics_db()
    metrics_data = db.get_all_metrics(
        include_inactive=include_inactive,
        builtin_only=builtin_only,
        custom_only=custom_only
    )
    
    metrics_info = []
    for metric in metrics_data:
        metrics_info.append(
            MetricInfo(
                name=metric["name"],
                description=metric["description"],
                is_builtin=bool(metric["is_builtin"]),
                created_by=metric["created_by"],
                created_at=metric["created_at"],
                updated_at=metric["updated_at"],
                version=metric["version"],
                is_active=bool(metric["is_active"])
            )
        )
    
    return {"metrics": metrics_info, "count": len(metrics_info)}


@app.get("/metrics/code/{metric_name}", response_model=MetricCodeResponse, tags=["Metrics"])
async def get_metric_code(metric_name: str):
    """
    評価指標のコードを取得する
    """
    try:
        db = get_metrics_db()
        metric_info = db.get_metric(metric_name)
        
        if not metric_info:
            raise HTTPException(
                status_code=404,
                detail=f"評価指標 {metric_name} が見つかりません"
            )
        
        if metric_info["is_builtin"]:
            # 組み込み評価指標の場合はローダーからクラスを取得してソースコードを抽出
            loader = get_metrics_loader()
            metric_class = loader.get_metric_class(metric_name)
            
            if metric_class:
                code = inspect.getsource(metric_class)
            else:
                code = metric_info["code"] or "# コードが見つかりません"
        else:
            # カスタム評価指標の場合はDBから直接取得
            code = metric_info["code"] or "# コードが見つかりません"
        
        return {
            "name": metric_name,
            "code": code
        }
    except Exception as e:
        logger.error(f"評価指標コードの取得中にエラーが発生: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"サーバーエラー: {str(e)}"
        )


@app.post("/metrics/validate", response_model=ValidateCodeResponse, tags=["Metrics"])
async def validate_code(request: ValidateCodeRequest):
    """
    評価指標コードを検証する
    """
    try:
        db = get_metrics_db()
        valid = db._validate_metric_code(request.code)
        
        if not valid:
            return {
                "valid": False,
                "metric_name": None,
                "error": "BaseMetricを継承したクラスが見つかりません"
            }
        
        # 一時的に評価指標クラスを生成してメトリック名を取得
        loader = get_metrics_loader()
        temp_module_name = f"temp_validate_{hash(request.code)}"
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            # 必要なインポートを追加
            full_code = """
from evaluator.base import BaseMetric
import re
import math
import numpy as np
from typing import Optional, List, Dict, Any, Union

""" + request.code
            temp.write(full_code.encode('utf-8'))
            temp_path = temp.name
        
        try:
            # モジュールを動的にインポート
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_path)
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
            
            # メトリック名を取得
            if metric_class:
                metric_instance = metric_class()
                metric_name = metric_instance.name
                
                return {
                    "valid": True,
                    "metric_name": metric_name,
                    "error": None
                }
            else:
                return {
                    "valid": False,
                    "metric_name": None,
                    "error": "BaseMetricを継承したクラスが見つかりません"
                }
        except Exception as e:
            return {
                "valid": False,
                "metric_name": None,
                "error": str(e)
            }
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"コード検証中にエラーが発生: {e}")
        return {
            "valid": False,
            "metric_name": None,
            "error": str(e)
        }


@app.post("/metrics", response_model=CustomMetricResponse, tags=["Metrics"])
async def add_custom_metric(request: CustomMetricRequest):
    """
    新しいカスタム評価指標を追加する
    """
    try:
        success = MetricFactory.add_custom_metric(
            name=request.name,
            code=request.code,
            description=request.description,
            created_by=request.created_by,
            save_to_file=request.save_to_file
        )
        
        if success:
            return {
                "success": True,
                "message": f"評価指標 {request.name} が正常に登録されました",
                "metric_name": request.name
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"評価指標 {request.name} の登録に失敗しました"
            )
    
    except Exception as e:
        logger.error(f"評価指標の追加中にエラーが発生: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"サーバーエラー: {str(e)}"
        )


@app.delete("/metrics/{metric_name}", response_model=CustomMetricResponse, tags=["Metrics"])
async def delete_metric(metric_name: str):
    """
    評価指標を削除する (論理削除)
    """
    try:
        db = get_metrics_db()
        success = db.delete_metric(metric_name)
        
        if success:
            return {
                "success": True,
                "message": f"評価指標 {metric_name} が正常に削除されました",
                "metric_name": metric_name
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"評価指標 {metric_name} の削除に失敗しました"
            )
    
    except Exception as e:
        logger.error(f"評価指標の削除中にエラーが発生: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"サーバーエラー: {str(e)}"
        )


@app.options("/metrics", tags=["CORS"])
async def options_metrics():
    """
    CORSプリフライトリクエスト対応
    """
    return Response()


@app.options("/metrics/{metric_name}", tags=["CORS"])
async def options_metric_detail(metric_name: str):
    """
    CORSプリフライトリクエスト対応
    """
    return Response()


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    APIサーバーを起動する
    
    Args:
        host: ホスト名
        port: ポート番号
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # コマンドラインから実行された場合はサーバーを起動
    start_api_server()