#!/usr/bin/env python3
"""
音乐质量评估模块 - 评估生成音乐的治疗效果和技术质量

理论基础：
1. Music Information Retrieval (MIR) Metrics:
   - Spectral Centroid: 音色亮度
   - Zero Crossing Rate: 噪声检测
   - MFCC: 音色特征
   - Tempo Estimation: 节奏稳定性

2. Therapeutic Music Assessment (2024):
   - "Quantitative Assessment of Music Therapy" - Journal of Music Therapy
   - 基于情绪一致性、节奏稳定性、音量动态的评估框架

3. Audio Quality Metrics:
   - SNR (Signal-to-Noise Ratio)
   - THD (Total Harmonic Distortion) 
   - Dynamic Range

作者：心境流转团队
日期：2024
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """音乐质量评估结果"""
    # 技术质量指标
    signal_to_noise_ratio: float     # 信噪比
    dynamic_range: float             # 动态范围
    spectral_centroid_mean: float    # 平均频谱质心
    zero_crossing_rate: float        # 过零率
    
    # 治疗效果指标
    tempo_stability: float           # 节奏稳定性
    volume_consistency: float        # 音量一致性
    emotional_coherence: float       # 情绪一致性
    therapeutic_suitability: float   # 治疗适用性
    
    # 综合评分
    technical_score: float           # 技术质量评分 (0-1)
    therapeutic_score: float         # 治疗效果评分 (0-1)
    overall_score: float            # 综合评分 (0-1)
    
    # 建议和警告
    recommendations: List[str]       # 改进建议
    warnings: List[str]             # 质量警告

class MusicQualityEvaluator:
    """
    音乐质量评估器
    
    评估维度：
    1. 技术质量：信噪比、动态范围、频谱分析
    2. 治疗效果：节奏稳定性、情绪一致性、睡眠适用性
    3. 用户体验：音量一致性、听觉舒适度
    """
    
    # 治疗音乐的质量标准（基于音乐治疗研究）
    THERAPEUTIC_STANDARDS = {
        'tempo_stability_threshold': 0.95,      # 节奏稳定性阈值
        'volume_consistency_threshold': 0.90,   # 音量一致性阈值
        'snr_minimum': 20,                      # 最小信噪比 (dB)
        'dynamic_range_optimal': (12, 30),     # 最优动态范围 (dB)
        'spectral_centroid_sleep_max': 2000,   # 睡眠音乐最大频谱质心 (Hz)
        'zero_crossing_rate_max': 0.1          # 最大过零率（噪声控制）
    }
    
    def __init__(self, sample_rate: int = 32000):
        """
        初始化评估器
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        
        # 检查依赖库
        self.librosa_available = self._check_librosa()
        
    def _check_librosa(self) -> bool:
        """检查librosa库是否可用"""
        try:
            import librosa
            self.librosa = librosa
            return True
        except ImportError:
            logger.warning("librosa库未安装，某些高级分析功能将不可用")
            logger.warning("建议安装：pip install librosa")
            return False
    
    def evaluate_music_quality(self, 
                             audio_data: np.ndarray, 
                             metadata: Dict,
                             therapy_context: Optional[Dict] = None) -> QualityMetrics:
        """
        全面评估音乐质量
        
        Args:
            audio_data: 音频数据
            metadata: 生成元数据
            therapy_context: 治疗上下文信息
            
        Returns:
            QualityMetrics: 详细的质量评估结果
        """
        logger.info("🔍 开始音乐质量评估...")
        
        # 1. 技术质量分析
        technical_metrics = self._analyze_technical_quality(audio_data)
        
        # 2. 治疗效果分析
        therapeutic_metrics = self._analyze_therapeutic_effectiveness(
            audio_data, metadata, therapy_context
        )
        
        # 3. 综合评分计算
        scores = self._calculate_composite_scores(technical_metrics, therapeutic_metrics)
        
        # 4. 生成建议和警告
        recommendations, warnings = self._generate_recommendations(
            technical_metrics, therapeutic_metrics, metadata
        )
        
        # 5. 构建评估结果
        quality_metrics = QualityMetrics(
            # 技术质量
            signal_to_noise_ratio=technical_metrics['snr'],
            dynamic_range=technical_metrics['dynamic_range'],
            spectral_centroid_mean=technical_metrics['spectral_centroid_mean'],
            zero_crossing_rate=technical_metrics['zero_crossing_rate'],
            
            # 治疗效果
            tempo_stability=therapeutic_metrics['tempo_stability'],
            volume_consistency=therapeutic_metrics['volume_consistency'],
            emotional_coherence=therapeutic_metrics['emotional_coherence'],
            therapeutic_suitability=therapeutic_metrics['therapeutic_suitability'],
            
            # 综合评分
            technical_score=scores['technical_score'],
            therapeutic_score=scores['therapeutic_score'],
            overall_score=scores['overall_score'],
            
            # 建议和警告
            recommendations=recommendations,
            warnings=warnings
        )
        
        self._log_evaluation_results(quality_metrics)
        
        return quality_metrics
    
    def _analyze_technical_quality(self, audio_data: np.ndarray) -> Dict:
        """分析技术质量指标"""
        
        # 1. 信噪比估算（简化实现）
        signal_power = np.mean(audio_data ** 2)
        noise_estimate = np.mean(np.abs(np.diff(audio_data))) * 0.1  # 简化噪声估计
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        
        # 2. 动态范围
        max_amplitude = np.max(np.abs(audio_data))
        min_amplitude = np.min(np.abs(audio_data[np.abs(audio_data) > 0.001]))
        dynamic_range = 20 * np.log10(max_amplitude / (min_amplitude + 1e-10))
        
        # 3. 过零率
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zero_crossing_rate = zero_crossings / len(audio_data)
        
        # 4. 频谱质心（如果librosa可用）
        if self.librosa_available:
            try:
                spectral_centroid = self.librosa.feature.spectral_centroid(
                    y=audio_data, sr=self.sample_rate
                )[0]
                spectral_centroid_mean = np.mean(spectral_centroid)
            except Exception as e:
                logger.warning(f"频谱质心计算失败: {e}")
                spectral_centroid_mean = 1000  # 默认值
        else:
            # 简化的频谱质心估算
            fft = np.abs(np.fft.fft(audio_data))
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            spectral_centroid_mean = np.sum(freqs[:len(freqs)//2] * fft[:len(fft)//2]) / np.sum(fft[:len(fft)//2])
        
        return {
            'snr': snr,
            'dynamic_range': dynamic_range,
            'spectral_centroid_mean': abs(spectral_centroid_mean),
            'zero_crossing_rate': zero_crossing_rate
        }
    
    def _analyze_therapeutic_effectiveness(self, 
                                         audio_data: np.ndarray, 
                                         metadata: Dict,
                                         therapy_context: Optional[Dict]) -> Dict:
        """分析治疗效果指标"""
        
        # 1. 节奏稳定性分析
        tempo_stability = self._analyze_tempo_stability(audio_data)
        
        # 2. 音量一致性
        volume_consistency = self._analyze_volume_consistency(audio_data)
        
        # 3. 情绪一致性（基于元数据）
        emotional_coherence = self._analyze_emotional_coherence(metadata, therapy_context)
        
        # 4. 治疗适用性评估
        therapeutic_suitability = self._assess_therapeutic_suitability(
            audio_data, metadata, therapy_context
        )
        
        return {
            'tempo_stability': tempo_stability,
            'volume_consistency': volume_consistency,
            'emotional_coherence': emotional_coherence,
            'therapeutic_suitability': therapeutic_suitability
        }
    
    def _analyze_tempo_stability(self, audio_data: np.ndarray) -> float:
        """分析节奏稳定性"""
        if not self.librosa_available:
            # 简化实现：基于能量变化估算
            window_size = self.sample_rate // 4  # 0.25秒窗口
            energy_windows = []
            
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                energy = np.mean(window ** 2)
                energy_windows.append(energy)
            
            if len(energy_windows) < 2:
                return 0.5
            
            # 计算能量变化的稳定性
            energy_variance = np.var(energy_windows)
            energy_mean = np.mean(energy_windows)
            stability = 1.0 / (1.0 + energy_variance / (energy_mean + 1e-10))
            
            return min(stability, 1.0)
        
        try:
            # 使用librosa进行节拍检测
            tempo, beats = self.librosa.beat.beat_track(
                y=audio_data, sr=self.sample_rate
            )
            
            if len(beats) < 4:
                return 0.5  # 节拍太少，无法评估
            
            # 计算节拍间隔的稳定性
            beat_intervals = np.diff(beats) / self.sample_rate
            if len(beat_intervals) == 0:
                return 0.5
            
            interval_variance = np.var(beat_intervals)
            interval_mean = np.mean(beat_intervals)
            
            # 稳定性评分：方差越小越稳定
            stability = 1.0 / (1.0 + interval_variance / (interval_mean + 1e-10))
            return min(stability, 1.0)
            
        except Exception as e:
            logger.warning(f"节奏稳定性分析失败: {e}")
            return 0.5
    
    def _analyze_volume_consistency(self, audio_data: np.ndarray) -> float:
        """分析音量一致性"""
        # 分析RMS能量的一致性
        window_size = self.sample_rate  # 1秒窗口
        rms_values = []
        
        for i in range(0, len(audio_data) - window_size + 1, window_size // 2):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if len(rms_values) < 2:
            return 1.0
        
        # 计算RMS值的变异系数
        rms_mean = np.mean(rms_values)
        rms_std = np.std(rms_values)
        
        if rms_mean == 0:
            return 0.0
        
        coefficient_of_variation = rms_std / rms_mean
        
        # 一致性评分：变异系数越小越一致
        consistency = 1.0 / (1.0 + coefficient_of_variation * 10)
        return min(consistency, 1.0)
    
    def _analyze_emotional_coherence(self, 
                                   metadata: Dict, 
                                   therapy_context: Optional[Dict]) -> float:
        """分析情绪一致性"""
        if not metadata or 'emotion_state' not in metadata:
            return 0.5
        
        emotion_state = metadata['emotion_state']
        target_emotion = emotion_state.get('primary_emotion', 'neutral')
        
        # 基于prompt分析情绪一致性
        prompt = metadata.get('prompt', '').lower()
        
        # 情绪关键词匹配度
        emotion_keywords = {
            'anger': ['aggressive', 'powerful', 'intense', 'rock', 'metal'],
            'fear': ['dark', 'tense', 'mysterious', 'ambient'],
            'sadness': ['melancholic', 'slow', 'blues', 'emotional'],
            'joy': ['uplifting', 'energetic', 'bright', 'major'],
            'neutral': ['calm', 'peaceful', 'ambient', 'relaxing']
        }
        
        target_keywords = emotion_keywords.get(target_emotion, emotion_keywords['neutral'])
        matches = sum(1 for keyword in target_keywords if keyword in prompt)
        
        coherence = matches / len(target_keywords) if target_keywords else 0.5
        return min(coherence, 1.0)
    
    def _assess_therapeutic_suitability(self, 
                                      audio_data: np.ndarray,
                                      metadata: Dict,
                                      therapy_context: Optional[Dict]) -> float:
        """评估治疗适用性"""
        suitability_factors = []
        
        # 1. BPM适用性（针对睡眠治疗）
        bpm_target = metadata.get('bpm_target', 70)
        if 40 <= bpm_target <= 100:  # 治疗音乐的理想BPM范围
            bpm_suitability = 1.0
        elif 30 <= bpm_target <= 120:  # 可接受范围
            bpm_suitability = 0.7
        else:
            bpm_suitability = 0.3
        
        suitability_factors.append(bpm_suitability)
        
        # 2. 频谱适用性（睡眠音乐应避免过高频率）
        if hasattr(self, 'spectral_centroid_mean'):
            spectral_centroid = self.spectral_centroid_mean
        else:
            # 简化的频谱质心计算
            fft = np.abs(np.fft.fft(audio_data))
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * fft[:len(fft)//2]) / np.sum(fft[:len(fft)//2])
        
        if spectral_centroid < 1000:  # 低频为主，适合睡眠
            spectral_suitability = 1.0
        elif spectral_centroid < 2000:  # 中频，较适合
            spectral_suitability = 0.8
        else:  # 高频，不太适合睡眠
            spectral_suitability = 0.4
        
        suitability_factors.append(spectral_suitability)
        
        # 3. 动态范围适用性（睡眠音乐不应有剧烈的音量变化）
        max_amplitude = np.max(np.abs(audio_data))
        amplitude_changes = np.abs(np.diff(audio_data))
        max_change = np.max(amplitude_changes) if len(amplitude_changes) > 0 else 0
        
        if max_change / (max_amplitude + 1e-10) < 0.1:  # 变化温和
            dynamic_suitability = 1.0
        elif max_change / (max_amplitude + 1e-10) < 0.3:  # 变化适中
            dynamic_suitability = 0.7
        else:  # 变化剧烈
            dynamic_suitability = 0.3
        
        suitability_factors.append(dynamic_suitability)
        
        # 4. 时长适用性
        duration = len(audio_data) / self.sample_rate
        target_duration = metadata.get('duration', 60)
        
        duration_ratio = duration / target_duration if target_duration > 0 else 1
        if 0.9 <= duration_ratio <= 1.1:  # 时长误差在10%以内
            duration_suitability = 1.0
        elif 0.8 <= duration_ratio <= 1.2:  # 时长误差在20%以内
            duration_suitability = 0.8
        else:
            duration_suitability = 0.5
        
        suitability_factors.append(duration_suitability)
        
        # 综合适用性评分
        overall_suitability = np.mean(suitability_factors)
        return overall_suitability
    
    def _calculate_composite_scores(self, 
                                  technical_metrics: Dict, 
                                  therapeutic_metrics: Dict) -> Dict:
        """计算综合评分"""
        
        # 1. 技术质量评分
        tech_factors = []
        
        # SNR评分
        snr = technical_metrics['snr']
        if snr >= 30:
            snr_score = 1.0
        elif snr >= 20:
            snr_score = 0.8
        elif snr >= 10:
            snr_score = 0.6
        else:
            snr_score = 0.3
        tech_factors.append(snr_score)
        
        # 动态范围评分
        dr = technical_metrics['dynamic_range']
        optimal_range = self.THERAPEUTIC_STANDARDS['dynamic_range_optimal']
        if optimal_range[0] <= dr <= optimal_range[1]:
            dr_score = 1.0
        elif 8 <= dr <= 40:  # 可接受范围
            dr_score = 0.7
        else:
            dr_score = 0.4
        tech_factors.append(dr_score)
        
        # 过零率评分（噪声控制）
        zcr = technical_metrics['zero_crossing_rate']
        max_zcr = self.THERAPEUTIC_STANDARDS['zero_crossing_rate_max']
        if zcr <= max_zcr:
            zcr_score = 1.0
        elif zcr <= max_zcr * 2:
            zcr_score = 0.6
        else:
            zcr_score = 0.2
        tech_factors.append(zcr_score)
        
        technical_score = np.mean(tech_factors)
        
        # 2. 治疗效果评分
        therapeutic_score = np.mean([
            therapeutic_metrics['tempo_stability'],
            therapeutic_metrics['volume_consistency'],
            therapeutic_metrics['emotional_coherence'],
            therapeutic_metrics['therapeutic_suitability']
        ])
        
        # 3. 综合评分（技术质量权重0.3，治疗效果权重0.7）
        overall_score = 0.3 * technical_score + 0.7 * therapeutic_score
        
        return {
            'technical_score': technical_score,
            'therapeutic_score': therapeutic_score,
            'overall_score': overall_score
        }
    
    def _generate_recommendations(self, 
                                technical_metrics: Dict, 
                                therapeutic_metrics: Dict,
                                metadata: Dict) -> Tuple[List[str], List[str]]:
        """生成改进建议和质量警告"""
        recommendations = []
        warnings = []
        
        # 技术质量建议
        if technical_metrics['snr'] < 20:
            warnings.append("信噪比过低，可能影响音频质量")
            recommendations.append("调整生成参数以减少噪声")
        
        if technical_metrics['dynamic_range'] < 10:
            recommendations.append("增加动态范围以提升音乐表现力")
        elif technical_metrics['dynamic_range'] > 35:
            warnings.append("动态范围过大，可能影响睡眠治疗效果")
            recommendations.append("降低音量变化幅度")
        
        if technical_metrics['zero_crossing_rate'] > 0.1:
            warnings.append("检测到较高的噪声水平")
            recommendations.append("使用低通滤波器减少高频噪声")
        
        # 治疗效果建议
        if therapeutic_metrics['tempo_stability'] < 0.8:
            warnings.append("节奏稳定性不足")
            recommendations.append("调整prompt以获得更稳定的节奏")
        
        if therapeutic_metrics['volume_consistency'] < 0.7:
            warnings.append("音量变化过大")
            recommendations.append("应用音量规范化处理")
        
        if therapeutic_metrics['emotional_coherence'] < 0.6:
            recommendations.append("优化prompt以提高情绪一致性")
        
        if therapeutic_metrics['therapeutic_suitability'] < 0.7:
            warnings.append("治疗适用性评分较低")
            recommendations.append("检查BPM设置和频谱分布")
        
        # 元数据相关建议
        bpm_target = metadata.get('bpm_target', 70)
        if bpm_target > 100:
            warnings.append(f"BPM ({bpm_target}) 过高，不适合睡眠治疗")
            recommendations.append("将BPM降低至60-80范围")
        elif bpm_target < 40:
            warnings.append(f"BPM ({bpm_target}) 过低，可能过于单调")
            recommendations.append("将BPM调整至50-70范围")
        
        return recommendations, warnings
    
    def _log_evaluation_results(self, metrics: QualityMetrics):
        """记录评估结果"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 [音乐质量评估 v1.0] 评估完成:")
        logger.info(f"{'='*60}")
        logger.info(f"🔧 技术质量评分: {metrics.technical_score:.2f}/1.0")
        logger.info(f"  - 信噪比: {metrics.signal_to_noise_ratio:.1f} dB")
        logger.info(f"  - 动态范围: {metrics.dynamic_range:.1f} dB")
        logger.info(f"  - 频谱质心: {metrics.spectral_centroid_mean:.0f} Hz")
        logger.info(f"  - 过零率: {metrics.zero_crossing_rate:.3f}")
        
        logger.info(f"\n💊 治疗效果评分: {metrics.therapeutic_score:.2f}/1.0")
        logger.info(f"  - 节奏稳定性: {metrics.tempo_stability:.2f}")
        logger.info(f"  - 音量一致性: {metrics.volume_consistency:.2f}")
        logger.info(f"  - 情绪一致性: {metrics.emotional_coherence:.2f}")
        logger.info(f"  - 治疗适用性: {metrics.therapeutic_suitability:.2f}")
        
        logger.info(f"\n🌟 综合评分: {metrics.overall_score:.2f}/1.0")
        
        if metrics.warnings:
            logger.info(f"\n⚠️ 质量警告:")
            for warning in metrics.warnings:
                logger.info(f"  • {warning}")
        
        if metrics.recommendations:
            logger.info(f"\n💡 改进建议:")
            for rec in metrics.recommendations:
                logger.info(f"  • {rec}")
        
        logger.info(f"{'='*60}")


# 工厂函数
def create_music_quality_evaluator(sample_rate: int = 32000) -> MusicQualityEvaluator:
    """
    创建音乐质量评估器实例
    
    Args:
        sample_rate: 音频采样率
        
    Returns:
        MusicQualityEvaluator实例
    """
    return MusicQualityEvaluator(sample_rate=sample_rate)