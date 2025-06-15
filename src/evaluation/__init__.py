"""
《心境流转》评估模块
Evaluation Module for Mood Transitions System

提供全面的学术评估和验证体系
- 治疗效果量化评估
- 学术指标计算和验证
- 科学研究数据分析
- 论文支撑材料生成
"""

from .academic_evaluator import (
    AcademicEvaluator,
    EvaluationMetrics,
    TherapyEffectiveness,
    AcademicValidation
)

__all__ = [
    'AcademicEvaluator',
    'EvaluationMetrics', 
    'TherapyEffectiveness',
    'AcademicValidation'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "Mood Transitions Research Team"
__description__ = "Academic evaluation and validation system for sleep-oriented audio-visual therapy"

def get_evaluation_info():
    """获取评估模块信息"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'capabilities': [
            '情绪识别准确性评估',
            '治疗效果量化分析',
            '统计学显著性检验',
            '效应量计算',
            '学术报告生成',
            '论文数据导出',
            '临床意义解释',
            '改进建议生成'
        ],
        'academic_standards': [
            'RCT随机对照试验设计',
            '统计功效分析',
            'Cohen效应量标准',
            '95%置信区间',
            'p<0.05显著性水平',
            '多维度结果评估'
        ]
    }

def quick_evaluation_setup():
    """快速评估设置"""
    print("📊 《心境流转》学术评估模块")
    print("="*50)
    
    # 创建评估器实例
    evaluator = AcademicEvaluator()
    
    info = get_evaluation_info()
    print(f"模块版本: {info['version']}")
    print(f"评估能力: {len(info['capabilities'])}项")
    print(f"学术标准: {len(info['academic_standards'])}项")
    
    print("\n✅ 学术评估模块初始化完成")
    print("🎓 符合硕士学位论文学术要求")
    
    return evaluator