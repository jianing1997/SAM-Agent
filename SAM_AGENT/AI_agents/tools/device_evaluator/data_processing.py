import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler


DEVICE_FEATURES = ['Cs_ratio', 'FA_ratio', 'Br_ratio', 'PVK_Eg', 'ITO', 'FTO', 'NiO', 'SnO2', 'ZnO', 'TiO2']
SUPPORTED_FEATURES = {'ecfp', 'pvk_ions', 'pvk_bandgap', 'substrate'}


# 计算 ECFP 指纹
def calculate_ecfp(dataset):
    ecfp_rows = []
    for _, row in dataset.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES字符串: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        ecfp_rows.append(list(map(int, fp.ToBitString())))

    ecfp_df = pd.DataFrame(ecfp_rows)
    ecfp_df.columns = [f'ECFP_{i}' for i in range(ecfp_df.shape[1])]
    return ecfp_df


# 处理钙钛矿离子特征
def process_pvk_ions(dataset, scaler_dict=None):
    if scaler_dict is None:
        scaler_dict = {}

    pvk_ions = dataset[['Cs_ratio', 'FA_ratio', 'Br_ratio']].copy()
    pvk_ions.fillna(pvk_ions.mean(), inplace=True)

    pvk_ions_norm = pd.DataFrame()
    for column in pvk_ions.columns:
        if column not in scaler_dict:
            scaler = MinMaxScaler()
            scaler.fit(pvk_ions[[column]])
            scaler_dict[column] = scaler
        else:
            scaler = scaler_dict[column]

        pvk_ions_norm[column] = scaler.transform(pvk_ions[[column]]).flatten()

    return pvk_ions_norm, scaler_dict


# 处理钙钛矿带隙特征
def process_pvk_bandgap(dataset, scaler_dict=None):
    if scaler_dict is None:
        scaler_dict = {}

    pvk_bandgap = dataset[['PVK_Eg']].copy()
    pvk_bandgap.fillna(pvk_bandgap.mean(), inplace=True)

    if 'Bandgap' not in scaler_dict:
        scaler = MinMaxScaler()
        scaler.fit(pvk_bandgap)
        scaler_dict['Bandgap'] = scaler
    else:
        scaler = scaler_dict['Bandgap']

    pvk_bandgap_norm = scaler.transform(pvk_bandgap)
    pvk_bandgap_norm = pd.DataFrame(pvk_bandgap_norm, columns=['Bandgap'])

    return pvk_bandgap_norm, scaler_dict


# 处理基底类型特征
def process_substrate(dataset, scaler_dict=None):
    if scaler_dict is None:
        scaler_dict = {}

    substrate = dataset[['ITO', 'FTO', 'NiO', 'SnO2', 'ZnO', 'TiO2']].copy()
    substrate.fillna(substrate.mean(), inplace=True)

    substrate_norm = pd.DataFrame()
    for column in substrate.columns:
        if column not in scaler_dict:
            scaler = MinMaxScaler()
            scaler.fit(substrate[[column]])
            scaler_dict[column] = scaler
        else:
            scaler = scaler_dict[column]

        substrate_norm[column] = scaler.transform(substrate[[column]]).flatten()

    return substrate_norm, scaler_dict


# 归一化待预测样本
def normalize_predict_sample(df_predict_sample, scaler_dict):
    pvk_ions_norm = process_pvk_ions(df_predict_sample, scaler_dict)[0]
    pvk_bandgap_norm = process_pvk_bandgap(df_predict_sample, scaler_dict)[0]
    substrate_norm = process_substrate(df_predict_sample, scaler_dict)[0]

    return pd.concat([pvk_ions_norm, pvk_bandgap_norm, substrate_norm], axis=1)


# 处理训练数据集
def dataset_process(df, features_to_include):
    unsupported_features = set(features_to_include) - SUPPORTED_FEATURES
    if unsupported_features:
        supported = ', '.join(sorted(SUPPORTED_FEATURES))
        unsupported = ', '.join(sorted(unsupported_features))
        raise ValueError(f"不支持的特征类型: {unsupported}。当前支持: {supported}")

    dataset = df.copy()
    feature_list = []
    scaler_dict = {}

    if 'pvk_ions' in features_to_include:
        pvk_ions_features, scaler_dict = process_pvk_ions(dataset, scaler_dict)
        feature_list.append(pvk_ions_features)

    if 'pvk_bandgap' in features_to_include:
        pvk_bandgap_features, scaler_dict = process_pvk_bandgap(dataset, scaler_dict)
        feature_list.append(pvk_bandgap_features)

    if 'substrate' in features_to_include:
        substrate_features, scaler_dict = process_substrate(dataset, scaler_dict)
        feature_list.append(substrate_features)

    if 'ecfp' in features_to_include:
        ecfp_features = calculate_ecfp(dataset=dataset)
        feature_list.append(ecfp_features)

    final_features = pd.concat(feature_list, axis=1)
    print(final_features.shape)

    if 'label' in dataset.columns:
        processed_dataset = pd.concat([dataset['smiles'], final_features, dataset['label']], axis=1)
    else:
        processed_dataset = pd.concat([dataset['smiles'], final_features], axis=1)

    print(processed_dataset.shape)
    return processed_dataset, scaler_dict
