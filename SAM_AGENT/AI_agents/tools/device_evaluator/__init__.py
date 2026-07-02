"""
Device Evaluator - 钙钛矿太阳能电池器件效率预测器

该模块提供基于SAM分子和器件参数的效率等级预测功能。

Classes:
    DeviceEvaluator: 器件效率预测器核心类

Functions:
    train_and_save_models: 模型训练函数（来自train_models模块）

Usage:
    from AI_agents.tools.device_evaluator import DeviceEvaluator
    
    evaluator = DeviceEvaluator()
    result = evaluator.predict(
        smiles="O=P(O)(O)CCCCn1c2ccccc2c2c3ccccc3ccc21",
        Cs_ratio=0.05, FA_ratio=0.9, Br_ratio=0.15, PVK_Eg=1.56,
        ITO=0, FTO=0, NiO=1, SnO2=0, ZnO=0, TiO2=0
    )
"""

from .device_evaluator import DeviceEvaluator

__all__ = ['DeviceEvaluator']
__version__ = '1.0.0'

