"""
《心境流转》学术评估系统
Academic Evaluation System for Mood Transitions

提供科学严谨的学术评估体系
- 治疗效果量化评估
- 学术指标计算
- 科学验证方法
- 论文支撑数据生成
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

@dataclass
class EvaluationMetrics:
    """评估指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    mse: float
    correlation: float
    significance: float

@dataclass
class TherapyEffectiveness:
    """治疗效果评估"""
    emotion_improvement: float  # 情绪改善度
    sleep_quality_improvement: float  # 睡眠质量改善
    stress_reduction: float  # 压力减少
    user_satisfaction: float  # 用户满意度
    treatment_adherence: float  # 治疗依从性
    long_term_efficacy: float  # 长期疗效

@dataclass
class AcademicValidation:
    """学术验证结果"""
    statistical_significance: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    sample_size: int
    power_analysis: float

class AcademicEvaluator:
    """学术评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_history = []
        self.baseline_metrics = None
        
        # 学术标准阈值
        self.significance_threshold = 0.05
        self.effect_size_threshold = 0.5  # Cohen's d
        self.minimum_sample_size = 30
        
    def evaluate_emotion_recognition_accuracy(self, 
                                            predicted_emotions: List[Dict], 
                                            ground_truth_emotions: List[Dict]) -> EvaluationMetrics:
        """评估情绪识别准确性"""
        
        # 转换为数值数据
        pred_valence = [e['valence'] for e in predicted_emotions]
        pred_arousal = [e['arousal'] for e in predicted_emotions]
        true_valence = [e['valence'] for e in ground_truth_emotions]
        true_arousal = [e['arousal'] for e in ground_truth_emotions]
        
        # 计算相关性
        valence_corr = stats.pearsonr(pred_valence, true_valence)[0]
        arousal_corr = stats.pearsonr(pred_arousal, true_arousal)[0]
        overall_corr = (valence_corr + arousal_corr) / 2
        
        # 计算MSE
        valence_mse = np.mean((np.array(pred_valence) - np.array(true_valence))**2)
        arousal_mse = np.mean((np.array(pred_arousal) - np.array(true_arousal))**2)
        overall_mse = (valence_mse + arousal_mse) / 2
        
        # 分类准确性（四象限分类）
        pred_categories = [self._emotion_to_category(e) for e in predicted_emotions]
        true_categories = [self._emotion_to_category(e) for e in ground_truth_emotions]
        
        accuracy = accuracy_score(true_categories, pred_categories)
        precision = precision_score(true_categories, pred_categories, average='weighted')
        recall = recall_score(true_categories, pred_categories, average='weighted')
        f1 = f1_score(true_categories, pred_categories, average='weighted')
        
        # 显著性检验
        _, p_value = stats.ttest_rel(pred_valence, true_valence)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=0.0,  # 需要ROC曲线计算
            mse=overall_mse,
            correlation=overall_corr,
            significance=p_value
        )
    
    def _emotion_to_category(self, emotion: Dict) -> str:
        """将情绪坐标转换为类别"""
        valence = emotion['valence']
        arousal = emotion['arousal']
        
        if valence > 0 and arousal > 0:
            return "high_positive"  # 兴奋愉悦
        elif valence > 0 and arousal <= 0:
            return "low_positive"   # 平静满足
        elif valence <= 0 and arousal > 0:
            return "high_negative"  # 焦虑紧张
        else:
            return "low_negative"   # 抑郁低落
    
    def evaluate_therapy_effectiveness(self, 
                                     pre_treatment_data: List[Dict],
                                     post_treatment_data: List[Dict],
                                     follow_up_data: Optional[List[Dict]] = None) -> TherapyEffectiveness:
        """评估治疗效果"""
        
        # 计算情绪改善
        emotion_improvement = self._calculate_emotion_improvement(
            pre_treatment_data, post_treatment_data
        )
        
        # 计算睡眠质量改善
        sleep_improvement = self._calculate_sleep_improvement(
            pre_treatment_data, post_treatment_data
        )
        
        # 计算压力减少
        stress_reduction = self._calculate_stress_reduction(
            pre_treatment_data, post_treatment_data
        )
        
        # 用户满意度（从反馈数据中提取）
        satisfaction_scores = [d.get('satisfaction', 3.0) for d in post_treatment_data]
        user_satisfaction = np.mean(satisfaction_scores) / 5.0  # 标准化到0-1
        
        # 治疗依从性
        completed_sessions = sum(1 for d in post_treatment_data if d.get('completed', False))
        adherence = completed_sessions / len(post_treatment_data) if post_treatment_data else 0
        
        # 长期疗效（如果有随访数据）
        long_term_efficacy = 0.0
        if follow_up_data:
            long_term_efficacy = self._calculate_long_term_efficacy(
                pre_treatment_data, follow_up_data
            )
        
        return TherapyEffectiveness(
            emotion_improvement=emotion_improvement,
            sleep_quality_improvement=sleep_improvement,
            stress_reduction=stress_reduction,
            user_satisfaction=user_satisfaction,
            treatment_adherence=adherence,
            long_term_efficacy=long_term_efficacy
        )
    
    def _calculate_emotion_improvement(self, pre_data: List[Dict], post_data: List[Dict]) -> float:
        """计算情绪改善度"""
        # 计算治疗前后的情绪距离改善
        pre_emotions = [d.get('emotion_state', {}) for d in pre_data]
        post_emotions = [d.get('emotion_state', {}) for d in post_data]
        
        # 目标情绪状态（平静愉悦：valence=0.3, arousal=-0.3）
        target_emotion = {'valence': 0.3, 'arousal': -0.3}
        
        # 计算治疗前后与目标状态的距离
        pre_distances = [
            np.sqrt((e.get('valence', 0) - target_emotion['valence'])**2 + 
                   (e.get('arousal', 0) - target_emotion['arousal'])**2)
            for e in pre_emotions
        ]
        
        post_distances = [
            np.sqrt((e.get('valence', 0) - target_emotion['valence'])**2 + 
                   (e.get('arousal', 0) - target_emotion['arousal'])**2)
            for e in post_emotions
        ]
        
        pre_avg_distance = np.mean(pre_distances) if pre_distances else 1.0
        post_avg_distance = np.mean(post_distances) if post_distances else 1.0
        
        # 改善度 = (治疗前距离 - 治疗后距离) / 治疗前距离
        improvement = (pre_avg_distance - post_avg_distance) / pre_avg_distance
        return max(0, min(1, improvement))  # 限制在0-1范围
    
    def _calculate_sleep_improvement(self, pre_data: List[Dict], post_data: List[Dict]) -> float:
        """计算睡眠质量改善"""
        pre_sleep_scores = [d.get('sleep_quality', 3.0) for d in pre_data]
        post_sleep_scores = [d.get('sleep_quality', 3.0) for d in post_data]
        
        pre_avg = np.mean(pre_sleep_scores) if pre_sleep_scores else 3.0
        post_avg = np.mean(post_sleep_scores) if post_sleep_scores else 3.0
        
        # 睡眠质量改善（假设1-5分量表）
        improvement = (post_avg - pre_avg) / 4.0  # 最大改善为4分
        return max(0, min(1, improvement))
    
    def _calculate_stress_reduction(self, pre_data: List[Dict], post_data: List[Dict]) -> float:
        """计算压力减少"""
        pre_stress_levels = [d.get('stress_level', 0.5) for d in pre_data]
        post_stress_levels = [d.get('stress_level', 0.5) for d in post_data]
        
        pre_avg = np.mean(pre_stress_levels) if pre_stress_levels else 0.5
        post_avg = np.mean(post_stress_levels) if post_stress_levels else 0.5
        
        # 压力减少度
        reduction = (pre_avg - post_avg) / pre_avg if pre_avg > 0 else 0
        return max(0, min(1, reduction))
    
    def _calculate_long_term_efficacy(self, pre_data: List[Dict], follow_up_data: List[Dict]) -> float:
        """计算长期疗效"""
        # 与短期效果计算类似，但使用随访数据
        return self._calculate_emotion_improvement(pre_data, follow_up_data)
    
    def statistical_validation(self, 
                             control_group_data: List[Dict],
                             treatment_group_data: List[Dict],
                             outcome_measure: str = 'emotion_improvement') -> AcademicValidation:
        """统计学验证"""
        
        # 提取结果变量
        control_outcomes = [d.get(outcome_measure, 0.0) for d in control_group_data]
        treatment_outcomes = [d.get(outcome_measure, 0.0) for d in treatment_group_data]
        
        # t检验
        t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)
        
        # 效应量计算 (Cohen's d)
        pooled_std = np.sqrt(((len(control_outcomes) - 1) * np.var(control_outcomes, ddof=1) + 
                             (len(treatment_outcomes) - 1) * np.var(treatment_outcomes, ddof=1)) / 
                            (len(control_outcomes) + len(treatment_outcomes) - 2))
        
        effect_size = (np.mean(treatment_outcomes) - np.mean(control_outcomes)) / pooled_std
        
        # 置信区间
        se = pooled_std * np.sqrt(1/len(control_outcomes) + 1/len(treatment_outcomes))
        df = len(control_outcomes) + len(treatment_outcomes) - 2
        t_critical = stats.t.ppf(0.975, df)  # 95% CI
        
        mean_diff = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        ci_lower = mean_diff - t_critical * se
        ci_upper = mean_diff + t_critical * se
        
        # 功效分析
        power = self._calculate_statistical_power(
            effect_size, len(control_outcomes) + len(treatment_outcomes), 0.05
        )
        
        return AcademicValidation(
            statistical_significance=p_value < self.significance_threshold,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            sample_size=len(control_outcomes) + len(treatment_outcomes),
            power_analysis=power
        )
    
    def _calculate_statistical_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """计算统计功效"""
        # 简化的功效计算（实际应使用专门的统计包）
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(sample_size/4) - z_alpha
        power = stats.norm.cdf(z_beta)
        return max(0, min(1, power))
    
    def generate_academic_report(self, 
                                evaluation_results: Dict[str, Any],
                                study_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """生成学术报告"""
        
        report = {
            'study_information': {
                'title': '《心境流转》睡眠导向音视觉治疗系统效果评估',
                'date': datetime.now().isoformat(),
                'sample_size': study_metadata.get('sample_size', 0),
                'study_duration': study_metadata.get('duration_weeks', 0),
                'methodology': '随机对照试验 (RCT)'
            },
            
            'primary_outcomes': {},
            'secondary_outcomes': {},
            'statistical_analysis': {},
            'clinical_significance': {},
            'limitations': [],
            'conclusions': [],
            'recommendations': []
        }
        
        # 填充主要结果
        if 'therapy_effectiveness' in evaluation_results:
            effectiveness = evaluation_results['therapy_effectiveness']
            report['primary_outcomes'] = {
                'emotion_improvement': {
                    'value': effectiveness.emotion_improvement,
                    'interpretation': self._interpret_effect_size(effectiveness.emotion_improvement)
                },
                'sleep_quality_improvement': {
                    'value': effectiveness.sleep_quality_improvement,
                    'interpretation': self._interpret_effect_size(effectiveness.sleep_quality_improvement)
                }
            }
        
        # 填充统计分析
        if 'statistical_validation' in evaluation_results:
            validation = evaluation_results['statistical_validation']
            report['statistical_analysis'] = {
                'significance': validation.statistical_significance,
                'p_value': validation.p_value,
                'effect_size': validation.effect_size,
                'confidence_interval': validation.confidence_interval,
                'statistical_power': validation.power_analysis
            }
        
        # 生成结论
        report['conclusions'] = self._generate_conclusions(evaluation_results)
        
        # 生成建议
        report['recommendations'] = self._generate_recommendations(evaluation_results)
        
        return report
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """解释效应量"""
        if effect_size < 0.2:
            return "微小效应"
        elif effect_size < 0.5:
            return "小效应"
        elif effect_size < 0.8:
            return "中等效应"
        else:
            return "大效应"
    
    def _generate_conclusions(self, evaluation_results: Dict) -> List[str]:
        """生成研究结论"""
        conclusions = []
        
        # 基于统计显著性
        if evaluation_results.get('statistical_validation', {}).get('statistical_significance', False):
            conclusions.append("治疗干预显示出统计学显著的治疗效果")
        
        # 基于效应量
        effect_size = evaluation_results.get('statistical_validation', {}).get('effect_size', 0)
        if effect_size > 0.5:
            conclusions.append("治疗效果具有临床意义")
        
        # 基于用户满意度
        satisfaction = evaluation_results.get('therapy_effectiveness', {}).get('user_satisfaction', 0)
        if satisfaction > 0.7:
            conclusions.append("用户对治疗系统表现出高满意度")
        
        return conclusions
    
    def _generate_recommendations(self, evaluation_results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于依从性
        adherence = evaluation_results.get('therapy_effectiveness', {}).get('treatment_adherence', 0)
        if adherence < 0.8:
            recommendations.append("需要改进治疗依从性策略")
        
        # 基于长期效果
        long_term = evaluation_results.get('therapy_effectiveness', {}).get('long_term_efficacy', 0)
        if long_term < 0.6:
            recommendations.append("需要加强长期效果维持机制")
        
        # 基于样本量
        sample_size = evaluation_results.get('statistical_validation', {}).get('sample_size', 0)
        if sample_size < 100:
            recommendations.append("建议扩大样本量以提高研究功效")
        
        return recommendations
    
    def export_results_for_publication(self, 
                                     evaluation_results: Dict,
                                     output_path: str = "academic_results.json"):
        """导出学术发表用结果"""
        
        publication_data = {
            'metadata': {
                'system_name': '《心境流转》',
                'evaluation_date': datetime.now().isoformat(),
                'academic_level': '硕士学位论文',
                'field': '人工智能 + 心理健康'
            },
            
            'methodology': {
                'approach': '多模态深度学习 + 音乐治疗理论',
                'technical_framework': 'ISO三阶段治疗原则',
                'emotion_model': 'Valence-Arousal二维情绪模型',
                'evaluation_metrics': '多维度量化评估体系'
            },
            
            'results': evaluation_results,
            
            'academic_contributions': [
                '首创睡眠导向的多模态治疗方法',
                '将音乐治疗理论与AI技术深度融合', 
                '建立了情绪识别到治疗内容生成的完整链路',
                '提供了量化的治疗效果评估体系'
            ],
            
            'innovation_points': [
                '多感官协同治疗机制',
                '个性化情绪轨迹规划',
                '实时治疗效果监测',
                '科学理论指导的AI应用'
            ]
        }
        
        # 保存结果
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(publication_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"学术评估结果已导出至: {output_file}")
        
        return publication_data

if __name__ == "__main__":
    # 测试学术评估器
    evaluator = AcademicEvaluator()
    
    # 模拟数据
    predicted_emotions = [
        {'valence': 0.2, 'arousal': -0.1},
        {'valence': 0.1, 'arousal': -0.2},
        {'valence': 0.3, 'arousal': -0.3}
    ]
    
    ground_truth_emotions = [
        {'valence': 0.25, 'arousal': -0.05},
        {'valence': 0.15, 'arousal': -0.15},
        {'valence': 0.28, 'arousal': -0.25}
    ]
    
    # 评估情绪识别准确性
    metrics = evaluator.evaluate_emotion_recognition_accuracy(
        predicted_emotions, ground_truth_emotions
    )
    
    print("📊 情绪识别评估结果:")
    print(f"   准确率: {metrics.accuracy:.3f}")
    print(f"   相关性: {metrics.correlation:.3f}")
    print(f"   MSE: {metrics.mse:.3f}")
    
    # 生成学术报告
    evaluation_results = {
        'emotion_recognition_metrics': metrics,
        'therapy_effectiveness': TherapyEffectiveness(
            emotion_improvement=0.65,
            sleep_quality_improvement=0.58,
            stress_reduction=0.72,
            user_satisfaction=0.84,
            treatment_adherence=0.76,
            long_term_efficacy=0.61
        )
    }
    
    report = evaluator.generate_academic_report(
        evaluation_results,
        {'sample_size': 50, 'duration_weeks': 4}
    )
    
    print("\n📝 学术评估报告已生成")
    print(f"研究结论数量: {len(report['conclusions'])}")
    print(f"改进建议数量: {len(report['recommendations'])}")