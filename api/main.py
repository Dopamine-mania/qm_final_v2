"""
《心境流转》API服务主入口
Main API Service for Mood Transitions System

基于FastAPI的RESTful API服务
- 多模态情绪识别接口
- 治疗内容生成接口
- 实时治疗会话管理
- 系统监控和管理接口
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging
from pathlib import Path

# 导入项目模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.emotion_recognition import EmotionRecognitionAdapter
from src.models.music_generation import MusicGenerationAdapter  
from src.models.video_generation import VideoGenerationAdapter
from src.therapy.orchestrator import TherapyOrchestrator
from src.optimization import quick_setup_for_jupyterhub
from api.schemas import *
from api.middleware import setup_middleware
from api.dependencies import get_therapy_service, get_optimization_service

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="《心境流转》API服务",
    description="Sleep-Oriented Audio-Visual Therapy System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 设置中间件
setup_middleware(app)

# 全局服务实例
therapy_service = None
optimization_service = None

@app.on_event("startup")
async def startup_event():
    """应用启动初始化"""
    global therapy_service, optimization_service
    
    logger.info("🚀 《心境流转》API服务启动中...")
    
    try:
        # 初始化性能优化
        optimization_service = quick_setup_for_jupyterhub()
        logger.info("✅ 性能优化模块初始化完成")
        
        # 初始化治疗服务
        therapy_service = TherapyOrchestrator()
        logger.info("✅ 治疗服务初始化完成")
        
        logger.info("🎉 《心境流转》API服务启动成功!")
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭清理"""
    logger.info("🛑 《心境流转》API服务关闭中...")
    
    # 清理资源
    if therapy_service:
        # 停止所有进行中的会话
        pass
    
    logger.info("✅ 服务关闭完成")

# ==================== 核心API接口 ====================

@app.get("/")
async def root():
    """根路径 - 系统信息"""
    return {
        "name": "《心境流转》睡眠导向音视觉治疗系统",
        "version": "1.0.0",
        "status": "running",
        "description": "Sleep-Oriented Audio-Visual Therapy System",
        "academic_level": "硕士学位论文",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    global therapy_service, optimization_service
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "therapy_service": therapy_service is not None,
            "optimization_service": optimization_service is not None
        }
    }
    
    # 检查关键组件
    if therapy_service:
        health_status["components"] = {
            "emotion_recognition": True,
            "music_generation": True,
            "video_generation": True,
            "therapy_planning": True
        }
    
    return health_status

# ==================== 情绪识别接口 ====================

@app.post("/api/emotion/recognize", response_model=EmotionRecognitionResponse)
async def recognize_emotion(request: EmotionRecognitionRequest):
    """多模态情绪识别"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        # 执行情绪识别
        start_time = time.time()
        
        result = therapy_service.recognize_emotion(
            text_input=request.text_input,
            audio_input=request.audio_input,
            user_context=request.user_context
        )
        
        processing_time = time.time() - start_time
        
        return EmotionRecognitionResponse(
            emotion_state=result['emotion_state'],
            confidence=result['confidence'],
            processing_time_ms=processing_time * 1000,
            session_id=result.get('session_id'),
            recommendations=result.get('recommendations', [])
        )
        
    except Exception as e:
        logger.error(f"情绪识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"情绪识别处理失败: {str(e)}")

# ==================== 治疗规划接口 ====================

@app.post("/api/therapy/plan", response_model=TherapyPlanResponse)
async def create_therapy_plan(request: TherapyPlanRequest):
    """创建个性化治疗方案"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        # 创建治疗方案
        start_time = time.time()
        
        plan = therapy_service.create_therapy_plan(
            current_emotion=request.current_emotion,
            target_emotion=request.target_emotion,
            user_profile=request.user_profile,
            session_config=request.session_config
        )
        
        processing_time = time.time() - start_time
        
        return TherapyPlanResponse(
            plan_id=plan['plan_id'],
            iso_stages=plan['iso_stages'],
            total_duration=plan['total_duration'],
            recommended_modalities=plan['recommended_modalities'],
            personalization_factors=plan['personalization_factors'],
            processing_time_ms=processing_time * 1000
        )
        
    except Exception as e:
        logger.error(f"治疗规划失败: {e}")
        raise HTTPException(status_code=500, detail=f"治疗规划失败: {str(e)}")

# ==================== 内容生成接口 ====================

@app.post("/api/content/music/generate", response_model=MusicGenerationResponse)
async def generate_music(request: MusicGenerationRequest):
    """生成治疗音乐"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        start_time = time.time()
        
        result = therapy_service.generate_therapeutic_music(
            prescription=request.prescription,
            duration=request.duration,
            quality_requirements=request.quality_requirements
        )
        
        processing_time = time.time() - start_time
        
        return MusicGenerationResponse(
            audio_id=result['audio_id'],
            duration=result['duration'],
            quality_metrics=result['quality_metrics'],
            therapeutic_features=result['therapeutic_features'],
            download_url=f"/api/content/music/{result['audio_id']}/download",
            processing_time_ms=processing_time * 1000
        )
        
    except Exception as e:
        logger.error(f"音乐生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"音乐生成失败: {str(e)}")

@app.post("/api/content/video/generate", response_model=VideoGenerationResponse)  
async def generate_video(request: VideoGenerationRequest):
    """生成治疗视频"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        start_time = time.time()
        
        result = therapy_service.generate_therapeutic_video(
            visual_config=request.visual_config,
            duration=request.duration,
            sync_with_audio=request.sync_with_audio
        )
        
        processing_time = time.time() - start_time
        
        return VideoGenerationResponse(
            video_id=result['video_id'],
            duration=result['duration'],
            resolution=result['resolution'],
            quality_metrics=result['quality_metrics'],
            visual_features=result['visual_features'],
            download_url=f"/api/content/video/{result['video_id']}/download",
            processing_time_ms=processing_time * 1000
        )
        
    except Exception as e:
        logger.error(f"视频生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")

# ==================== 治疗会话接口 ====================

@app.post("/api/session/start", response_model=TherapySessionResponse)
async def start_therapy_session(request: TherapySessionRequest):
    """开始治疗会话"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        session = therapy_service.start_therapy_session(
            user_id=request.user_id,
            therapy_plan_id=request.therapy_plan_id,
            session_config=request.session_config
        )
        
        return TherapySessionResponse(
            session_id=session['session_id'],
            status=session['status'],
            current_stage=session['current_stage'],
            estimated_duration=session['estimated_duration'],
            next_action=session['next_action']
        )
        
    except Exception as e:
        logger.error(f"治疗会话启动失败: {e}")
        raise HTTPException(status_code=500, detail=f"治疗会话启动失败: {str(e)}")

@app.get("/api/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """获取会话状态"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        status = therapy_service.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return SessionStatusResponse(
            session_id=session_id,
            status=status['status'],
            current_stage=status['current_stage'],
            progress=status['progress'],
            remaining_time=status['remaining_time'],
            real_time_metrics=status['real_time_metrics']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话状态失败: {str(e)}")

@app.post("/api/session/{session_id}/feedback")
async def submit_session_feedback(session_id: str, feedback: SessionFeedback):
    """提交会话反馈"""
    if not therapy_service:
        raise HTTPException(status_code=503, detail="治疗服务未初始化")
    
    try:
        result = therapy_service.process_session_feedback(session_id, feedback.dict())
        
        return {
            "status": "success",
            "message": "反馈已提交",
            "adaptive_adjustments": result.get('adaptive_adjustments', [])
        }
        
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

# ==================== 系统管理接口 ====================

@app.get("/api/system/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """获取系统统计信息"""
    global optimization_service
    
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time(),  # 简化实现
            "active_sessions": 0,  # 实际应从therapy_service获取
            "total_requests": 0,   # 实际应从监控系统获取
            "performance_metrics": {}
        }
        
        if optimization_service:
            hardware_adapter = optimization_service.get('hardware_adapter')
            if hardware_adapter:
                stats["hardware_info"] = hardware_adapter.get_hardware_summary()
        
        return SystemStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统统计失败: {str(e)}")

@app.get("/api/system/performance")
async def get_performance_metrics():
    """获取性能指标"""
    global optimization_service
    
    if not optimization_service:
        raise HTTPException(status_code=503, detail="优化服务未初始化")
    
    try:
        memory_optimizer = optimization_service.get('memory_optimizer')
        
        if memory_optimizer:
            stats = memory_optimizer.get_optimization_stats()
            return {
                "status": "success",
                "performance_data": stats,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "limited",
                "message": "性能监控功能有限（CPU模式）",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

# ==================== 内容下载接口 ====================

@app.get("/api/content/music/{audio_id}/download")
async def download_music(audio_id: str):
    """下载生成的音乐文件"""
    # 实际实现中应该从存储系统获取文件
    # 这里返回示例响应
    return {
        "message": "音乐下载功能（演示模式）",
        "audio_id": audio_id,
        "note": "实际部署时将返回真实音频文件"
    }

@app.get("/api/content/video/{video_id}/download")
async def download_video(video_id: str):
    """下载生成的视频文件"""
    # 实际实现中应该从存储系统获取文件
    # 这里返回示例响应
    return {
        "message": "视频下载功能（演示模式）",
        "video_id": video_id,
        "note": "实际部署时将返回真实视频文件"
    }

# ==================== WebSocket实时接口 ====================

@app.websocket("/ws/session/{session_id}")
async def websocket_session_monitor(websocket, session_id: str):
    """WebSocket会话实时监控"""
    await websocket.accept()
    
    try:
        while True:
            # 获取实时会话数据
            if therapy_service:
                status = therapy_service.get_session_status(session_id)
                if status:
                    await websocket.send_json(status)
            
            await asyncio.sleep(1)  # 每秒更新一次
            
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        await websocket.close()

# ==================== 启动配置 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="《心境流转》API服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    print("🌟 《心境流转》睡眠导向音视觉治疗系统 API服务")
    print(f"🔗 服务地址: http://{args.host}:{args.port}")
    print(f"📚 API文档: http://{args.host}:{args.port}/docs")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )