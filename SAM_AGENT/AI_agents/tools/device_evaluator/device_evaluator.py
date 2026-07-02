import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Union
from rdkit import Chem
from .data_processing import calculate_ecfp, normalize_predict_sample

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceEvaluator:
    """
    钙钛矿太阳能电池器件效率预测器
    
    功能：基于SAM分子SMILES和器件参数预测效率等级（低效/中等效率/高效）
    模型：100个SVM分类器集成
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 模型文件路径，默认为当前目录下的 trained_models.pkl
            scaler_path: Scaler文件路径，默认为当前目录下的 scaler_dict.pkl
        """
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置默认路径
        self.model_path = model_path or os.path.join(current_dir, 'trained_models.pkl')
        self.scaler_path = scaler_path or os.path.join(current_dir, 'scaler_dict.pkl')
        
        # 加载模型和scaler
        self.models = None
        self.scaler_dict = None
        self._load_models()
        
        logger.info("DeviceEvaluator 初始化完成")
    
    
    def _load_models(self):
        """加载预训练的模型和scaler"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"模型文件不存在: {self.model_path}\n"
                    f"请先运行 train_models.py 训练模型"
                )
            
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(
                    f"Scaler文件不存在: {self.scaler_path}\n"
                    f"请先运行 train_models.py 训练模型"
                )
            
            # 加载模型
            with open(self.model_path, 'rb') as f:
                self.models = pickle.load(f)
            logger.info(f"成功加载 {len(self.models)} 个预训练模型")
            
            # 加载scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler_dict = pickle.load(f)
            logger.info("成功加载归一化参数")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    
    def predict(self, smiles: str, Cs_ratio: float, FA_ratio: float, 
                Br_ratio: float, PVK_Eg: float, ITO: int, FTO: int, 
                NiO: int, SnO2: int, ZnO: int, TiO2: int) -> str:
        """
        预测器件效率等级
        
        Args:
            smiles: SAM分子的SMILES字符串
            Cs_ratio: 铯离子比例 (0-1)
            FA_ratio: 甲脒离子比例 (0-1)
            Br_ratio: 溴离子比例 (0-1)
            PVK_Eg: 钙钛矿带隙 (eV)
            ITO, FTO, NiO, SnO2, ZnO, TiO2: 基底类型 (0或1，6选1)
        
        Returns:
            str: 预测效率等级（低效/中等效率/高效）或错误信息
        """
        try:
            # 步骤1：验证SMILES有效性
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"错误：无效的SMILES字符串: {smiles}"
            
            # 步骤2：验证参数有效性
            validation_error = self._validate_parameters(
                Cs_ratio, FA_ratio, Br_ratio, PVK_Eg,
                ITO, FTO, NiO, SnO2, ZnO, TiO2
            )
            if validation_error:
                return validation_error
            
            # 步骤3：准备特征
            features = self._prepare_features(
                smiles, Cs_ratio, FA_ratio, Br_ratio, PVK_Eg,
                ITO, FTO, NiO, SnO2, ZnO, TiO2
            )
            
            # 步骤4：执行预测
            result = self._do_predict(features)
            
            return result
            
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}")
            return f"预测失败: {str(e)}"
    
    
    def _validate_parameters(self, Cs_ratio: float, FA_ratio: float, 
                            Br_ratio: float, PVK_Eg: float,
                            ITO: int, FTO: int, NiO: int, 
                            SnO2: int, ZnO: int, TiO2: int) -> Union[str, None]:
        """
        验证参数有效性
        
        Returns:
            str: 如果验证失败，返回错误信息；如果成功，返回None
        """
        # 验证离子比例范围
        for name, value in [("Cs_ratio", Cs_ratio), 
                           ("FA_ratio", FA_ratio), 
                           ("Br_ratio", Br_ratio)]:
            if not (0 <= value <= 1):
                return f"参数验证失败：{name} 必须在0-1之间，当前值：{value}"
        
        # 验证带隙范围（合理范围）
        if not (1.0 <= PVK_Eg <= 2.0):
            return (f"参数验证失败：PVK_Eg 应在1.0-2.0 eV范围内，当前值：{PVK_Eg}\n"
                   f"提示：典型钙钛矿带隙为1.5-1.7 eV")
        
        # 验证基底类型值
        substrates = {"ITO": ITO, "FTO": FTO, "NiO": NiO, 
                     "SnO2": SnO2, "ZnO": ZnO, "TiO2": TiO2}
        
        for name, value in substrates.items():
            if value not in [0, 1]:
                return f"参数验证失败：{name} 必须为0或1，当前值：{value}"
        
        # 验证基底类型互斥性（必须有且仅有一个为1）
        substrate_sum = sum(substrates.values())
        if substrate_sum != 1:
            selected = [name for name, val in substrates.items() if val == 1]
            if substrate_sum == 0:
                return ("参数验证失败：必须选择一种基底类型\n"
                       "可选项：ITO, FTO, NiO, SnO2, ZnO, TiO2")
            else:
                return (f"参数验证失败：只能选择一种基底类型，当前选择了 {substrate_sum} 个：{selected}\n"
                       f"请将其中一个设为1，其余设为0")
        
        return None  # 验证通过
    
    
    def _prepare_features(self, smiles: str, Cs_ratio: float, FA_ratio: float,
                         Br_ratio: float, PVK_Eg: float, ITO: int, FTO: int,
                         NiO: int, SnO2: int, ZnO: int, TiO2: int) -> pd.DataFrame:
        """
        准备模型输入特征
        
        Returns:
            pd.DataFrame: 形状为(1, 2058)的特征矩阵
        """
        # 步骤1：计算分子的ECFP指纹 (2048维)
        df_smiles = pd.DataFrame({'smiles': [smiles]})
        ecfp_features = calculate_ecfp(df_smiles)
        
        # 步骤2：准备器件参数
        device_params = pd.DataFrame([{
            'Cs_ratio': Cs_ratio,
            'FA_ratio': FA_ratio,
            'Br_ratio': Br_ratio,
            'PVK_Eg': PVK_Eg,
            'ITO': ITO,
            'FTO': FTO,
            'NiO': NiO,
            'SnO2': SnO2,
            'ZnO': ZnO,
            'TiO2': TiO2
        }])
        
        # 步骤3：归一化器件参数 (10维)
        device_params_normalized = normalize_predict_sample(
            device_params, self.scaler_dict
        )
        
        # 步骤4：合并特征 (10 + 2048 = 2058维)
        final_features = pd.concat([device_params_normalized, ecfp_features], axis=1)
        
        logger.info(f"特征准备完成，形状: {final_features.shape}")
        
        return final_features
    
    
    def _do_predict(self, features: pd.DataFrame) -> str:
        """
        使用100个模型进行集成预测，并只返回效率等级分类。
        """
        if self.models is None:
            return "错误：模型未加载"
        
        # 保留DataFrame列名，避免sklearn在预测时反复提示缺少feature names。
        X = features.astype(float)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)[0]
            predictions.append(pred)
        
        predictions_array = np.array(predictions)
        mean_pred = np.mean(predictions_array)
        
        if mean_pred < 0.5:
            final_label = 0
        elif mean_pred < 1.5:
            final_label = 1
        else:
            final_label = 2
        
        efficiency_map = {
            0: "低效",
            1: "中等效率",
            2: "高效"
        }
        
        efficiency_level = efficiency_map[final_label]
        logger.info(f"预测完成: {efficiency_level} (Label {final_label})")
        return efficiency_level


# 测试函数
if __name__ == "__main__":
    # 简单测试
    try:
        evaluator = DeviceEvaluator()
        
        # 测试样例（来自训练数据的第一条）
        test_smiles = "O=P(O)(O)CCCCn1c2ccccc2c2c3ccccc3ccc21"
        result = evaluator.predict(
            smiles=test_smiles,
            Cs_ratio=0.05,
            FA_ratio=0.9,
            Br_ratio=0.15,
            PVK_Eg=1.56,
            ITO=0, FTO=0, NiO=1, SnO2=0, ZnO=0, TiO2=0
        )
        print(result)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

