"""
钙钛矿器件效率预测模型训练脚本

功能：
1. 加载训练数据
2. 进行特征工程
3. 按分子骨架分类
4. 100次重复训练SVM模型
5. 保存模型和归一化参数

使用方法：
    python train_models.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.svm import SVC
from rdkit import Chem
from tqdm import tqdm

# 处理导入：支持作为脚本运行和作为模块导入
try:
    from .data_processing import dataset_process
except ImportError:
    from data_processing import dataset_process

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def classify_by_scaffold(processed_data):
    """
    按照分子骨架分类数据集
    
    分类规则：
    - df_1: 链状（无环）
    - df_2: 萘酰胺
    - df_3: 三苯胺
    - df_4: 咔唑
    - df_5: 联噻吩
    - df_6: 单环
    - df_7: 其他
    
    Args:
        processed_data: 处理后的数据集
    
    Returns:
        list: 包含7个DataFrame的列表
    """
    df_copy = processed_data.copy()
    
    # 初始化7个子数据集
    df_1 = pd.DataFrame(columns=df_copy.columns)
    df_2 = pd.DataFrame(columns=df_copy.columns)
    df_3 = pd.DataFrame(columns=df_copy.columns)
    df_4 = pd.DataFrame(columns=df_copy.columns)
    df_5 = pd.DataFrame(columns=df_copy.columns)
    df_6 = pd.DataFrame(columns=df_copy.columns)
    df_7 = pd.DataFrame(columns=df_copy.columns)
    
    print("开始按骨架分类...")
    
    # 分类1: 链状（无环）
    indices_to_drop = []
    for index, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings == 0:
                df_1 = pd.concat([df_1, pd.DataFrame([row])], ignore_index=True)
                indices_to_drop.append(index)
    
    df_copy = df_copy.drop(indices_to_drop)
    print(f"链状分子: {len(df_1)} 个")
    
    # 分类2: 萘酰胺
    patt_1 = Chem.MolFromSmiles('O=C1NC(=O)C2=CC=CC3=C2C1=CC=C3')
    indices_to_drop = []
    for index, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(patt_1):
            df_2 = pd.concat([df_2, pd.DataFrame([row])], ignore_index=True)
            indices_to_drop.append(index)
    
    df_copy = df_copy.drop(indices_to_drop)
    print(f"萘酰胺: {len(df_2)} 个")
    
    # 分类3: 三苯胺
    patt_2 = Chem.MolFromSmiles('C1=CC=C(C=C1)N(C1=CC=CC=C1)C1=CC=CC=C1')
    indices_to_drop = []
    for index, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(patt_2):
            df_3 = pd.concat([df_3, pd.DataFrame([row])], ignore_index=True)
            indices_to_drop.append(index)
    
    df_copy = df_copy.drop(indices_to_drop)
    print(f"三苯胺: {len(df_3)} 个")
    
    # 分类4: 咔唑
    patt_3 = Chem.MolFromSmiles('N1C2=C(C=CC=C2)C2=C1C=CC=C2')
    indices_to_drop = []
    for index, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(patt_3):
            df_4 = pd.concat([df_4, pd.DataFrame([row])], ignore_index=True)
            indices_to_drop.append(index)
    
    df_copy = df_copy.drop(indices_to_drop)
    print(f"咔唑: {len(df_4)} 个")
    
    # 分类5: 联噻吩
    patt_4 = Chem.MolFromSmiles('S1C=CC=C1C1=CC=CS1')
    indices_to_drop = []
    for index, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(patt_4):
            df_5 = pd.concat([df_5, pd.DataFrame([row])], ignore_index=True)
            indices_to_drop.append(index)
    
    df_copy = df_copy.drop(indices_to_drop)
    print(f"联噻吩: {len(df_5)} 个")
    
    # 分类6: 单环
    indices_to_drop = []
    for index, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings == 1:
                df_6 = pd.concat([df_6, pd.DataFrame([row])], ignore_index=True)
                indices_to_drop.append(index)
    
    df_copy = df_copy.drop(indices_to_drop)
    print(f"单环: {len(df_6)} 个")
    
    # 分类7: 其他
    df_7 = df_copy.copy()
    print(f"其他: {len(df_7)} 个")
    
    print(f"总计: {len(df_1) + len(df_2) + len(df_3) + len(df_4) + len(df_5) + len(df_6) + len(df_7)} 个分子\n")
    
    return [df_1, df_2, df_3, df_4, df_5, df_6, df_7]


def train_models(processed_data, n_iterations=100):
    """
    训练多个SVM模型
    
    Args:
        processed_data: 处理后的数据集
        n_iterations: 训练迭代次数，默认100
    
    Returns:
        list: 包含所有训练好的模型的列表
    """
    # 按骨架分类
    scaffold_dfs = classify_by_scaffold(processed_data)
    df_1, df_2, df_3, df_4, df_5, df_6, df_7 = scaffold_dfs
    
    # 只使用前6类进行训练（df_7太少，不参与训练集划分）
    dfs = [df_1, df_2, df_3, df_4, df_5, df_6]
    
    # 最佳超参数（来自notebook调优结果）
    best_params = {
        'C': 50,
        'kernel': 'linear',
        'gamma': 0.0001
    }
    
    models = []
    
    print(f"开始训练 {n_iterations} 个模型...")
    print(f"超参数: {best_params}\n")
    
    # 训练循环
    for i in tqdm(range(n_iterations), desc="训练进度"):
        df_test = pd.DataFrame()
        df_train = pd.DataFrame()
        
        # 为每个骨架类别划分训练/测试集
        for df_sub in dfs:
            if len(df_sub) == 0:
                continue
            
            # 根据数据量确定测试集大小
            if len(df_sub) <= 10:
                test_size = 1
            elif 10 < len(df_sub) <= 20:
                test_size = 2
            elif 20 < len(df_sub) <= 30:
                test_size = 3
            elif 30 < len(df_sub) <= 40:
                test_size = 4
            elif 40 < len(df_sub) <= 50:
                test_size = 5
            else:
                test_size = 6
            
            # 随机选择测试集
            test_indices = np.random.choice(df_sub.index, size=test_size, replace=False)
            
            # 添加到测试集
            df_test = pd.concat([df_test, df_sub.loc[test_indices]], ignore_index=False)
            
            # 剩余的添加到训练集
            train_indices = df_sub.index.difference(test_indices)
            df_train = pd.concat([df_train, df_sub.loc[train_indices]], ignore_index=False)
        
        # 分离特征和标签
        df_train_X = df_train.iloc[:, 1:-1].astype(float)
        df_train_label = df_train['label'].astype(int)
        
        # 训练模型
        model = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma']
        )
        model.fit(df_train_X, df_train_label)
        
        models.append(model)
    
    print(f"\n训练完成！共生成 {len(models)} 个模型")
    
    return models


def train_and_save_models():
    """
    主函数：训练模型并保存
    """
    print("=" * 60)
    print("钙钛矿器件效率预测模型训练")
    print("=" * 60)
    print()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据路径
    dataset_path = os.path.join(current_dir, "SAM_data_PCE_15%_0721.csv")
    
    # 输出路径
    model_output_path = os.path.join(current_dir, "trained_models.pkl")
    scaler_output_path = os.path.join(current_dir, "scaler_dict.pkl")
    
    # 步骤1: 加载数据
    print("步骤 1/4: 加载训练数据")
    print(f"数据路径: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"训练数据文件不存在: {dataset_path}")
    
    dataset = pd.read_csv(dataset_path)
    print(f"数据加载完成，共 {len(dataset)} 条记录\n")
    
    # 步骤2: 特征工程
    print("步骤 2/4: 特征工程处理")
    print("使用的特征类型: ECFP, 钙钛矿离子比例, 钙钛矿带隙, 基底类型")
    
    features_to_include = ['ecfp', 'pvk_ions', 'pvk_bandgap', 'substrate']
    processed_data, scaler_dict = dataset_process(dataset, features_to_include)
    
    print(f"特征处理完成，特征维度: {processed_data.shape}\n")
    
    # 步骤3: 训练模型
    print("步骤 3/4: 训练模型")
    models = train_models(processed_data, n_iterations=100)
    print()
    
    # 步骤4: 保存模型
    print("步骤 4/4: 保存模型和参数")
    
    # 保存模型
    with open(model_output_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"✓ 模型已保存至: {model_output_path}")
    
    # 保存scaler
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
    print(f"✓ 归一化参数已保存至: {scaler_output_path}")
    
    # 显示文件大小
    model_size = os.path.getsize(model_output_path) / (1024 * 1024)  # MB
    scaler_size = os.path.getsize(scaler_output_path) / 1024  # KB
    print(f"\n文件大小:")
    print(f"  - 模型文件: {model_size:.2f} MB")
    print(f"  - Scaler文件: {scaler_size:.2f} KB")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n现在可以使用 DeviceEvaluator 进行预测了")


if __name__ == "__main__":
    try:
        train_and_save_models()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

