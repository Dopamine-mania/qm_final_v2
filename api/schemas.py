"""
《心境流转》API数据模型
API Schemas for Mood Transitions System

定义所有API接口的请求和响应数据模型
- Pydantic数据验证
- 自动API文档生成
- 类型安全保证
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum

# ==================== 基础数据类型 ====================

class EmotionState(BaseModel):
    """情绪状态"""
    valence: float = Field(..., ge=-1.0, le=1.0, description="情绪效价 (-1到1)")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="情绪唤醒度 (-1到1)")
    
class TherapyFocus(str, Enum):
    """治疗重点"""
    SLEEP = "sleep"
    ANXIETY = "anxiety" 
    DEPRESSION = "depression"
    STRESS = "stress"
    GENERAL = "general"

class SessionStatus(str, Enum):
    """会话状态"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ISOStage(str, Enum):
    """ISO治疗阶段"""
    SYNCHRONIZATION = "synchronization"
    GUIDANCE = "guidance" 
    CONSOLIDATION = "consolidation"

# ==================== 情绪识别相关 ====================

class EmotionRecognitionRequest(BaseModel):
    """情绪识别请求"""
    text_input: Optional[str] = Field(None, description="文本输入")
    audio_input: Optional[str] = Field(None, description="音频文件路径或Base64编码")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="用户上下文信息")
    
    @validator('text_input', 'audio_input')
    def at_least_one_input(cls, v, values):
        if not v and not values.get('text_input') and not values.get('audio_input'):
            raise ValueError('至少需要提供文本或音频输入之一')
        return v

class EmotionRecognitionResponse(BaseModel):
    """情绪识别响应"""
    emotion_state: EmotionState
    confidence: float = Field(..., ge=0.0, le=1.0, description="识别置信度")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")
    session_id: Optional[str] = Field(None, description="会话ID")
    recommendations: List[str] = Field(default_factory=list, description="建议列表")

# ==================== 治疗规划相关 ====================

class UserProfile(BaseModel):
    """用户档案"""
    user_id: str = Field(..., description="用户ID")
    age: Optional[int] = Field(None, ge=0, le=120, description="年龄")
    gender: Optional[str] = Field(None, description="性别")
    sleep_issues: List[str] = Field(default_factory=list, description="睡眠问题列表")
    therapy_history: Optional[Dict[str, Any]] = Field(default_factory=dict, description="治疗历史")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="个人偏好")

class SessionConfig(BaseModel):
    """会话配置"""
    max_duration: int = Field(600, ge=60, le=3600, description="最大时长(秒)")
    therapy_focus: TherapyFocus = Field(TherapyFocus.SLEEP, description="治疗重点")
    preferred_modalities: List[str] = Field(default_factory=lambda: ["audio", "visual"], description="偏好模态")
    intensity_level: float = Field(0.5, ge=0.0, le=1.0, description="强度级别")

class TherapyPlanRequest(BaseModel):
    """治疗方案请求"""
    current_emotion: EmotionState
    target_emotion: EmotionState  
    user_profile: UserProfile
    session_config: SessionConfig

class ISOStageConfig(BaseModel):
    """ISO阶段配置"""
    duration: int = Field(..., ge=30, description="阶段时长(秒)")
    target_emotion: EmotionState
    strategy: str = Field(..., description="治疗策略")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="阶段参数")

class TherapyPlanResponse(BaseModel):
    """治疗方案响应"""
    plan_id: str = Field(..., description="方案ID")
    iso_stages: Dict[ISOStage, ISOStageConfig] = Field(..., description="ISO三阶段配置")
    total_duration: int = Field(..., description="总时长(秒)")
    recommended_modalities: List[str] = Field(..., description="推荐模态")
    personalization_factors: Dict[str, Any] = Field(default_factory=dict, description="个性化因子")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")

# ==================== 内容生成相关 ====================

class MusicPrescription(BaseModel):
    """音乐处方"""
    tempo_bpm: int = Field(..., ge=40, le=200, description="节拍(BPM)")
    key: str = Field(..., description="调性")
    mood: str = Field(..., description="情绪风格")
    instruments: List[str] = Field(..., description="乐器列表")
    therapeutic_features: List[str] = Field(default_factory=list, description="治疗特征")

class MusicGenerationRequest(BaseModel):
    """音乐生成请求"""
    prescription: MusicPrescription
    duration: int = Field(..., ge=30, le=1800, description="时长(秒)")
    quality_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="质量要求")

class QualityMetrics(BaseModel):
    """质量指标"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="总体得分")
    technical_quality: float = Field(..., ge=0.0, le=1.0, description="技术质量")
    therapeutic_alignment: float = Field(..., ge=0.0, le=1.0, description="治疗对齐度")

class MusicGenerationResponse(BaseModel):
    """音乐生成响应"""
    audio_id: str = Field(..., description="音频ID")
    duration: int = Field(..., description="实际时长(秒)")
    quality_metrics: QualityMetrics
    therapeutic_features: List[str] = Field(..., description="治疗特征")
    download_url: str = Field(..., description="下载URL")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")

class VisualConfig(BaseModel):
    """视觉配置"""
    theme: str = Field(..., description="视觉主题")
    style: str = Field(..., description="视觉风格")
    color_temperature: float = Field(..., ge=2000, le=6500, description="色温(K)")
    brightness: float = Field(..., ge=0.0, le=1.0, description="亮度")
    motion_intensity: float = Field(..., ge=0.0, le=1.0, description="运动强度")

class VideoGenerationRequest(BaseModel):
    """视频生成请求"""
    visual_config: VisualConfig
    duration: int = Field(..., ge=30, le=1800, description="时长(秒)")
    sync_with_audio: Optional[str] = Field(None, description="同步音频ID")

class VideoGenerationResponse(BaseModel):
    """视频生成响应"""
    video_id: str = Field(..., description="视频ID")
    duration: int = Field(..., description="实际时长(秒)")
    resolution: str = Field(..., description="分辨率")
    quality_metrics: QualityMetrics
    visual_features: List[str] = Field(..., description="视觉特征")
    download_url: str = Field(..., description="下载URL")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")

# ==================== 治疗会话相关 ====================

class TherapySessionRequest(BaseModel):
    """治疗会话请求"""
    user_id: str = Field(..., description="用户ID")
    therapy_plan_id: str = Field(..., description="治疗方案ID")
    session_config: Optional[SessionConfig] = None

class TherapySessionResponse(BaseModel):
    """治疗会话响应"""
    session_id: str = Field(..., description="会话ID")
    status: SessionStatus = Field(..., description="会话状态")
    current_stage: Optional[ISOStage] = Field(None, description="当前阶段")
    estimated_duration: int = Field(..., description="预估时长(秒)")
    next_action: str = Field(..., description="下一步行动")

class RealTimeMetrics(BaseModel):
    """实时指标"""
    heart_rate: Optional[float] = Field(None, description="心率")
    stress_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="压力水平")
    engagement: Optional[float] = Field(None, ge=0.0, le=1.0, description="参与度")
    effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0, description="效果评估")

class SessionStatusResponse(BaseModel):
    """会话状态响应"""
    session_id: str = Field(..., description="会话ID")
    status: SessionStatus = Field(..., description="会话状态")
    current_stage: Optional[ISOStage] = Field(None, description="当前阶段")
    progress: float = Field(..., ge=0.0, le=1.0, description="进度百分比")
    remaining_time: int = Field(..., description="剩余时间(秒)")
    real_time_metrics: Optional[RealTimeMetrics] = None

class SessionFeedback(BaseModel):
    """会话反馈"""
    satisfaction: int = Field(..., ge=1, le=5, description="满意度(1-5)")
    effectiveness: int = Field(..., ge=1, le=5, description="效果评价(1-5)")
    comfort: int = Field(..., ge=1, le=5, description="舒适度(1-5)")
    comments: Optional[str] = Field(None, description="评论")
    suggestions: Optional[str] = Field(None, description="建议")

# ==================== 系统管理相关 ====================

class PerformanceMetrics(BaseModel):
    """性能指标"""
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU使用率(%)")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="内存使用率(%)")
    gpu_usage: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU使用率(%)")
    response_time_ms: float = Field(..., description="响应时间(毫秒)")
    throughput: float = Field(..., description="吞吐量(请求/秒)")

class SystemStatsResponse(BaseModel):
    """系统统计响应"""
    timestamp: str = Field(..., description="时间戳")
    uptime: float = Field(..., description="运行时间(秒)")
    active_sessions: int = Field(..., description="活跃会话数")
    total_requests: int = Field(..., description="总请求数")
    performance_metrics: Optional[PerformanceMetrics] = None
    hardware_info: Optional[Dict[str, Any]] = None

# ==================== 错误响应 ====================

class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误消息")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")

# ==================== 响应包装器 ====================

class APIResponse(BaseModel):
    """通用API响应包装器"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    error: Optional[ErrorResponse] = Field(None, description="错误信息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")

# ==================== 配置模型 ====================

class APIConfig(BaseModel):
    """API配置"""
    debug: bool = Field(False, description="调试模式")
    max_request_size: int = Field(100 * 1024 * 1024, description="最大请求大小(字节)")
    rate_limit: int = Field(100, description="速率限制(请求/分钟)")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS允许来源")
    
class ModelConfig(BaseModel):
    """模型配置"""
    emotion_model: str = Field("roberta-emotion", description="情绪识别模型")
    music_model: str = Field("musicgen-medium", description="音乐生成模型")
    video_model: str = Field("hunyuan-video", description="视频生成模型")
    device: str = Field("auto", description="计算设备")
    precision: str = Field("fp16", description="计算精度")

# ==================== 验证器 ====================

class RequestValidator:
    """请求验证器"""
    
    @staticmethod
    def validate_emotion_state(emotion: EmotionState) -> bool:
        """验证情绪状态"""
        return -1.0 <= emotion.valence <= 1.0 and -1.0 <= emotion.arousal <= 1.0
    
    @staticmethod  
    def validate_duration(duration: int, min_duration: int = 30, max_duration: int = 1800) -> bool:
        """验证时长参数"""
        return min_duration <= duration <= max_duration
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """验证用户ID格式"""
        return len(user_id) > 0 and len(user_id) <= 64