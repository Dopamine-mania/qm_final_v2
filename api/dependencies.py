"""
《心境流转》API依赖注入
API Dependencies for Mood Transitions System

提供API服务的依赖注入功能
- 服务实例管理
- 配置参数注入
- 认证和授权
- 资源限制
"""

from fastapi import Depends, HTTPException, status, Request
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

# 导入项目模块
from src.therapy.orchestrator import TherapyOrchestrator
from src.optimization import get_memory_optimizer, get_hardware_adapter
from api.schemas import UserProfile

logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================

class Settings:
    """应用设置"""
    SECRET_KEY = "mood_transitions_secret_key_2024"  # 生产环境应使用环境变量
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # API限制
    MAX_CONCURRENT_SESSIONS = 10
    MAX_SESSION_DURATION = 3600  # 1小时
    MAX_FILE_SIZE_MB = 100
    
    # 性能限制
    MAX_BATCH_SIZE = 8
    MAX_SEQUENCE_LENGTH = 2048
    
settings = Settings()

# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==================== 服务依赖 ====================

# 全局服务实例
_therapy_service = None
_optimization_service = None
_active_sessions = {}
_session_stats = {
    "total_sessions": 0,
    "active_sessions": 0,
    "failed_sessions": 0
}

async def get_therapy_service() -> TherapyOrchestrator:
    """获取治疗服务实例"""
    global _therapy_service
    
    if _therapy_service is None:
        try:
            _therapy_service = TherapyOrchestrator()
            logger.info("✅ 治疗服务实例创建成功")
        except Exception as e:
            logger.error(f"❌ 治疗服务创建失败: {e}")
            raise HTTPException(
                status_code=503,
                detail="治疗服务初始化失败"
            )
    
    return _therapy_service

async def get_optimization_service() -> Dict[str, Any]:
    """获取优化服务实例"""
    global _optimization_service
    
    if _optimization_service is None:
        try:
            # 获取内存优化器
            memory_optimizer = get_memory_optimizer()
            
            # 获取硬件适配器
            hardware_adapter = get_hardware_adapter()
            
            _optimization_service = {
                "memory_optimizer": memory_optimizer,
                "hardware_adapter": hardware_adapter,
                "status": "ready"
            }
            
            logger.info("✅ 优化服务实例创建成功")
        except Exception as e:
            logger.error(f"❌ 优化服务创建失败: {e}")
            # 优化服务失败不阻塞主服务
            _optimization_service = {
                "memory_optimizer": None,
                "hardware_adapter": None,
                "status": "limited"
            }
    
    return _optimization_service

# ==================== 认证依赖 ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)

async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """获取当前用户（可选认证）"""
    # 从请求头获取令牌
    authorization = request.headers.get("Authorization")
    
    if not authorization:
        # 如果没有提供认证信息，返回匿名用户
        return {
            "user_id": f"anonymous_{int(datetime.now().timestamp())}",
            "user_type": "anonymous",
            "permissions": ["basic_access"]
        }
    
    try:
        # 提取令牌
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证方案"
            )
        
        # 验证令牌
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌"
            )
        
        # 返回用户信息
        return {
            "user_id": user_id,
            "user_type": "authenticated",
            "permissions": payload.get("permissions", ["basic_access"]),
            "token_exp": payload.get("exp")
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌已过期"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无法验证令牌"
        )

# ==================== 资源限制依赖 ====================

async def check_session_limit(request: Request):
    """检查会话数量限制"""
    global _active_sessions, _session_stats
    
    # 清理过期会话
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session_info in _active_sessions.items()
        if (current_time - session_info["start_time"]).total_seconds() > settings.MAX_SESSION_DURATION
    ]
    
    for session_id in expired_sessions:
        del _active_sessions[session_id]
        _session_stats["active_sessions"] -= 1
    
    # 检查并发会话限制
    if len(_active_sessions) >= settings.MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            status_code=429,
            detail=f"并发会话数已达上限 ({settings.MAX_CONCURRENT_SESSIONS})"
        )
    
    return True

async def validate_file_size(request: Request):
    """验证文件大小"""
    content_length = request.headers.get("content-length")
    
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制 ({settings.MAX_FILE_SIZE_MB}MB)"
            )
    
    return True

async def validate_request_parameters(request: Request):
    """验证请求参数"""
    # 检查批处理大小
    if request.method == "POST":
        # 这里可以添加更复杂的参数验证逻辑
        pass
    
    return True

# ==================== 性能监控依赖 ====================

async def get_system_metrics() -> Dict[str, Any]:
    """获取系统性能指标"""
    try:
        optimization_service = await get_optimization_service()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(_active_sessions),
            "session_stats": _session_stats.copy()
        }
        
        # 添加硬件信息
        if optimization_service["hardware_adapter"]:
            hardware_summary = optimization_service["hardware_adapter"].get_hardware_summary()
            metrics["hardware"] = hardware_summary
        
        # 添加内存使用情况
        if optimization_service["memory_optimizer"]:
            memory_stats = optimization_service["memory_optimizer"].get_optimization_stats()
            metrics["memory"] = memory_stats
        
        return metrics
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "系统指标获取失败",
            "active_sessions": len(_active_sessions)
        }

# ==================== 会话管理依赖 ====================

def register_session(session_id: str, user_id: str, session_type: str = "therapy"):
    """注册新会话"""
    global _active_sessions, _session_stats
    
    _active_sessions[session_id] = {
        "user_id": user_id,
        "session_type": session_type,
        "start_time": datetime.now(),
        "last_activity": datetime.now(),
        "status": "active"
    }
    
    _session_stats["total_sessions"] += 1
    _session_stats["active_sessions"] += 1
    
    logger.info(f"会话已注册: {session_id} (用户: {user_id})")

def unregister_session(session_id: str, status: str = "completed"):
    """注销会话"""
    global _active_sessions, _session_stats
    
    if session_id in _active_sessions:
        session_info = _active_sessions[session_id]
        duration = (datetime.now() - session_info["start_time"]).total_seconds()
        
        del _active_sessions[session_id]
        _session_stats["active_sessions"] -= 1
        
        if status == "failed":
            _session_stats["failed_sessions"] += 1
        
        logger.info(f"会话已注销: {session_id} (状态: {status}, 时长: {duration:.1f}s)")

def update_session_activity(session_id: str):
    """更新会话活动时间"""
    global _active_sessions
    
    if session_id in _active_sessions:
        _active_sessions[session_id]["last_activity"] = datetime.now()

# ==================== 权限检查依赖 ====================

def require_permission(permission: str):
    """权限检查装饰器工厂"""
    async def permission_checker(current_user: Dict = Depends(get_current_user)):
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少必要权限: {permission}"
            )
        return current_user
    
    return permission_checker

# 预定义权限检查
require_admin = require_permission("admin_access")
require_api_access = require_permission("api_access")

# ==================== 配置依赖 ====================

async def get_api_config() -> Dict[str, Any]:
    """获取API配置"""
    return {
        "max_concurrent_sessions": settings.MAX_CONCURRENT_SESSIONS,
        "max_session_duration": settings.MAX_SESSION_DURATION,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "max_batch_size": settings.MAX_BATCH_SIZE,
        "max_sequence_length": settings.MAX_SEQUENCE_LENGTH
    }

# ==================== 健康检查依赖 ====================

async def health_check_dependencies() -> Dict[str, bool]:
    """健康检查相关服务"""
    health_status = {}
    
    try:
        therapy_service = await get_therapy_service()
        health_status["therapy_service"] = therapy_service is not None
    except:
        health_status["therapy_service"] = False
    
    try:
        optimization_service = await get_optimization_service()
        health_status["optimization_service"] = optimization_service["status"] == "ready"
    except:
        health_status["optimization_service"] = False
    
    return health_status