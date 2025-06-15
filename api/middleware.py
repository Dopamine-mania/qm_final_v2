"""
《心境流转》API中间件
API Middleware for Mood Transitions System

提供API服务的中间件功能
- CORS跨域处理
- 请求响应日志
- 错误处理和异常捕获
- 性能监控
- 安全防护
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import json
from typing import Callable
import traceback
from datetime import datetime
import uuid

# 配置日志
logger = logging.getLogger(__name__)

class RequestLoggingMiddleware:
    """请求日志中间件"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 记录请求开始
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            f"请求开始 - ID: {request_id}, "
            f"方法: {request.method}, "
            f"路径: {request.url.path}, "
            f"客户端: {client_ip}, "
            f"User-Agent: {user_agent}"
        )
        
        # 添加请求ID到请求对象
        request.state.request_id = request_id
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 记录请求完成
            process_time = time.time() - start_time
            logger.info(
                f"请求完成 - ID: {request_id}, "
                f"状态: {response.status_code}, "
                f"耗时: {process_time:.3f}s"
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # 记录请求错误
            process_time = time.time() - start_time
            logger.error(
                f"请求错误 - ID: {request_id}, "
                f"错误: {str(e)}, "
                f"耗时: {process_time:.3f}s"
            )
            
            # 返回错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "error": "内部服务器错误",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )

class ErrorHandlingMiddleware:
    """错误处理中间件"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException as e:
            # HTTP异常
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.warning(f"HTTP异常 - ID: {request_id}, 状态: {e.status_code}, 详情: {e.detail}")
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "status_code": e.status_code,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except ValueError as e:
            # 值错误
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.error(f"值错误 - ID: {request_id}, 错误: {str(e)}")
            
            return JSONResponse(
                status_code=400,
                content={
                    "error": "请求参数错误",
                    "details": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            # 未知异常
            request_id = getattr(request.state, 'request_id', 'unknown')
            error_trace = traceback.format_exc()
            
            logger.error(
                f"未知异常 - ID: {request_id}, "
                f"错误: {str(e)}, "
                f"堆栈: {error_trace}"
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "内部服务器错误",
                    "message": "服务暂时不可用，请稍后重试",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

class SecurityMiddleware:
    """安全中间件"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.max_request_size = 100 * 1024 * 1024  # 100MB
        self.rate_limit_requests = 100  # 每分钟最大请求数
        self.rate_limit_window = 60  # 时间窗口(秒)
        self.request_counts = {}  # 简单的速率限制实现
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # 检查请求大小
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "请求体过大",
                    "max_size_mb": self.max_request_size // (1024 * 1024)
                }
            )
        
        # 简单的速率限制
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # 清理过期记录
        self.request_counts = {
            ip: [(timestamp, count) for timestamp, count in requests 
                 if current_time - timestamp < self.rate_limit_window]
            for ip, requests in self.request_counts.items()
        }
        
        # 检查当前IP的请求数
        if client_ip in self.request_counts:
            recent_requests = len(self.request_counts[client_ip])
            if recent_requests >= self.rate_limit_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "请求过于频繁",
                        "retry_after": self.rate_limit_window
                    }
                )
        
        # 记录当前请求
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        self.request_counts[client_ip].append((current_time, 1))
        
        # 添加安全响应头
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

class PerformanceMonitoringMiddleware:
    """性能监控中间件"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.request_stats = {
            "total_requests": 0,
            "total_time": 0.0,
            "error_count": 0,
            "endpoint_stats": {}
        }
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            response = await call_next(request)
            
            # 记录成功请求
            process_time = time.time() - start_time
            self._update_stats(endpoint, process_time, response.status_code)
            
            # 添加性能头信息
            response.headers["X-Response-Time"] = f"{process_time:.3f}s"
            
            return response
            
        except Exception as e:
            # 记录错误请求
            process_time = time.time() - start_time
            self._update_stats(endpoint, process_time, 500, error=True)
            raise e
    
    def _update_stats(self, endpoint: str, process_time: float, status_code: int, error: bool = False):
        """更新统计信息"""
        self.request_stats["total_requests"] += 1
        self.request_stats["total_time"] += process_time
        
        if error or status_code >= 400:
            self.request_stats["error_count"] += 1
        
        if endpoint not in self.request_stats["endpoint_stats"]:
            self.request_stats["endpoint_stats"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "error_count": 0
            }
        
        stats = self.request_stats["endpoint_stats"][endpoint]
        stats["count"] += 1
        stats["total_time"] += process_time
        stats["min_time"] = min(stats["min_time"], process_time)
        stats["max_time"] = max(stats["max_time"], process_time)
        
        if error or status_code >= 400:
            stats["error_count"] += 1
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        total_requests = self.request_stats["total_requests"]
        avg_response_time = (
            self.request_stats["total_time"] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            "total_requests": total_requests,
            "average_response_time": avg_response_time,
            "error_rate": (
                self.request_stats["error_count"] / total_requests 
                if total_requests > 0 else 0
            ),
            "endpoint_stats": self.request_stats["endpoint_stats"]
        }

# 全局性能监控实例
performance_monitor = None

def setup_middleware(app: FastAPI):
    """设置所有中间件"""
    global performance_monitor
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 可信主机中间件（生产环境使用）
    # app.add_middleware(
    #     TrustedHostMiddleware,
    #     allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]
    # )
    
    # 性能监控中间件
    performance_monitor = PerformanceMonitoringMiddleware(app)
    app.middleware("http")(performance_monitor)
    
    # 安全中间件
    app.middleware("http")(SecurityMiddleware(app))
    
    # 错误处理中间件
    app.middleware("http")(ErrorHandlingMiddleware(app))
    
    # 请求日志中间件
    app.middleware("http")(RequestLoggingMiddleware(app))
    
    logger.info("✅ API中间件设置完成")

def get_performance_stats() -> dict:
    """获取性能统计信息"""
    global performance_monitor
    
    if performance_monitor:
        return performance_monitor.get_stats()
    else:
        return {
            "error": "性能监控未初始化",
            "total_requests": 0,
            "average_response_time": 0,
            "error_rate": 0
        }

# 自定义异常处理器
async def validation_exception_handler(request: Request, exc: Exception):
    """验证异常处理器"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "请求数据验证失败",
            "details": str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
    )