import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
#from mordred import Calculator, descriptors
#from padelpy import padeldescriptor
import glob
from sklearn.preprocessing import MinMaxScaler
import os

# 计算 ECFP 指纹
def calculate_ecfp(dataset):
    ecfp_df = pd.DataFrame()
    for index, row in dataset.iterrows():
        smiles = row['smiles']  
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
           fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
           fingerprint = fp.ToBitString()
           fingerprint_list = list(map(int, fingerprint))
           ecfp_df = pd.concat([ecfp_df, pd.DataFrame([fingerprint_list])], ignore_index=True)

    ecfp_df = ecfp_df.fillna(0)
    ecfp_df.columns = [f'ECFP_{i}' for i in range(ecfp_df.shape[1])]
    return ecfp_df

# 计算 MACCS 指纹
def calculate_maccs(dataset):
    maccs_df = pd.DataFrame()
    for index, row in dataset.iterrows():
        smiles = row["smiles"] 
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
           fingerprint = MACCSkeys.GenMACCSKeys(mol)
           fingerprint_list = list(map(int, fingerprint))
           maccs_df = pd.concat([maccs_df, pd.DataFrame([fingerprint_list])], ignore_index=True)
    maccs_df = maccs_df.fillna(0)
    maccs_df.columns = [f'MACCS_{i}' for i in range(maccs_df.shape[1])]
    return maccs_df

# 计算 Estate 指纹
def calculate_estate_fingerprint(dataset):
    smiles_file = 'temp_smiles_estate.smi'
    with open(smiles_file, 'w') as f:
        for smiles in dataset['smiles']:
            f.write(smiles + '\n')
    
    xml_files = glob.glob("D:\\fingerprints_xml\\*.xml")
    FP_list = ['AtomPairs2DCount',
               'AtomPairs2D',
               'EState',
               'CDKextended',
               'CDK',
               'CDKgraphonly',
               'KlekotaRothCount',
               'KlekotaRoth',
               'MACCS',
               'PubChem',
               'SubstructureCount',
               'Substructure'
               ]
    fp = dict(zip(FP_list, xml_files))
    fingerprint = 'EState'
    output_csv = ''.join([fingerprint,'.csv'])
    fingerprint_descriptortypes = fp[fingerprint]
    # 运行 PaDEL-Descriptor 生成 EState 指纹
    padeldescriptor(
                mol_dir=smiles_file, 
                d_file=output_csv,
                descriptortypes= fingerprint_descriptortypes,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False, 
                fingerprints=True
            )
        # 尝试用不同的编码读取CSV文件
    try:
        estate_df = pd.read_csv(output_csv, encoding='utf-8')
    except UnicodeDecodeError:
        estate_df = pd.read_csv(output_csv, encoding='ISO-8859-1')
    os.remove(smiles_file)
    os.remove(output_csv)
    return estate_df.iloc[:,1:]

# 计算 Pubchem 指纹
def calculate_pubchem(dataset, output_csv='PubChem.csv'):
    smiles_file = 'temp_smiles_pubchem.smi'
    with open(smiles_file, 'w') as f:
        for smiles in dataset['smiles']:
            f.write(smiles + '\n')
    
    xml_files = glob.glob("D:\\fingerprints_xml\\*.xml")
    FP_list = ['AtomPairs2DCount',
               'AtomPairs2D',
               'EState',
               'CDKextended',
               'CDK',
               'CDKgraphonly',
               'KlekotaRothCount',
               'KlekotaRoth',
               'MACCS',
               'PubChem',
               'SubstructureCount',
               'Substructure'
               ]
    fp = dict(zip(FP_list, xml_files))
    
    fingerprint = 'PubChem'
    output_csv = ''.join([fingerprint,'.csv'])
    fingerprint_descriptortypes = fp[fingerprint]
    
    padeldescriptor(
                mol_dir=smiles_file, 
                d_file=output_csv,
                descriptortypes= fingerprint_descriptortypes, 
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False, 
                fingerprints=True
            )
        # 尝试用不同的编码读取CSV文件
    try:
        pubchem_df = pd.read_csv(output_csv, encoding='utf-8')
    except UnicodeDecodeError:
        pubchem_df = pd.read_csv(output_csv, encoding='ISO-8859-1')
    os.remove(smiles_file)
    os.remove(output_csv)
    return pubchem_df.iloc[:,1:]

# 计算 RDkit描述符
def calculate_rdkit_descriptor(dataset):
    descriptors = [desc[0] for desc in Descriptors._descList]
    rows_to_append = []

    for index, row in dataset.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            print(f"Error: Invalid SMILES at index {index}: {row['smiles']}")
            descriptor_values = [None] * len(descriptors)
        else:
            AllChem.Compute2DCoords(mol)
            descriptor_values = []
            for desc_name in descriptors:
                try:
                    descriptor_values.append(getattr(Descriptors, desc_name)(mol))
                except Exception as e:
                    print(f"Error calculating descriptor {desc_name} for SMILES {row['smiles']} at index {index}: {e}")
                    descriptor_values.append(None)
                    
        rows_to_append.append(dict(zip(descriptors, descriptor_values)))

    rdkit_descriptors = pd.DataFrame(rows_to_append)
    
    # 去除标准差较小的描述符列
    normalized_df = rdkit_descriptors.div(rdkit_descriptors.abs().max())
    filtered_df = normalized_df.loc[:, normalized_df.std() >= 0.1]
    
    # 计算 pearson 相关性系数并移除相关性高的列
    correlation_matrix = filtered_df.corr().abs()
    to_remove = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.90:
                to_remove.add(correlation_matrix.columns[i])

    selected_descriptors_df = filtered_df.drop(columns=to_remove)

    # 归一化处理
    scaler = MinMaxScaler()
    rdkit_descriptors_norm = pd.DataFrame(
        scaler.fit_transform(selected_descriptors_df),
        columns=selected_descriptors_df.columns
    )
    
    rdkit_descriptors_norm.fillna(rdkit_descriptors_norm.mean(), inplace=True)
    
    return rdkit_descriptors_norm

# 计算 mordred 描述符
def calculate_mordred_descriptor(dataset):
    calc = Calculator(descriptors, ignore_3D=True)
    descriptor_names = [desc.__str__() for desc in calc.descriptors]
    mordred_descriptors = []

    for index, row in dataset.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        try:
            descriptor_values = calc.pandas([mol]).iloc[0]
        except Exception as e:
            print(f"Error calculating descriptors for SMILES '{row['smiles']}': {e}")
            descriptor_values = pd.Series([float('nan')] * len(descriptor_names), index=descriptor_names)
        mordred_descriptors.append(descriptor_values)

    mordred_df = pd.DataFrame(mordred_descriptors)
    
    # 用均值填充缺失值
    numeric_cols = mordred_df.select_dtypes(include=[np.number]).columns
    mordred_df[numeric_cols].fillna(mordred_df[numeric_cols].mean(), inplace=True)
    
    # 去除标准差较小的描述符列
    normalized_df = mordred_df[numeric_cols].div(mordred_df[numeric_cols].abs().max())
    filtered_df = normalized_df.loc[:, normalized_df.std() >= 0.1]
    
    # 计算 pearson 相关性系数并移除相关性高的列
    correlation_matrix = filtered_df.corr().abs()
    to_remove = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.90:
                to_remove.add(correlation_matrix.columns[i])

    selected_descriptors_df = filtered_df.drop(columns=to_remove)

    # 归一化处理
    scaler = MinMaxScaler()
    mordred_descriptors_norm = pd.DataFrame(
        scaler.fit_transform(selected_descriptors_df),
        columns=selected_descriptors_df.columns
    )
    
    mordred_descriptors_norm.fillna(mordred_descriptors_norm.mean(), inplace=True)
    
    return mordred_descriptors_norm
    
# 处理钙钛矿离子特征
def process_pvk_ions(dataset):
    X_scaler = MinMaxScaler()
    pvk_ions = dataset.iloc[:, 25:28]  
    pvk_ions.fillna(pvk_ions.mean(), inplace=True)
    X_scaler.fit(pvk_ions)
    pvk_ions_norm = X_scaler.transform(pvk_ions)
    pvk_ions_norm = pd.DataFrame(pvk_ions_norm, columns=pvk_ions.columns)
    return pvk_ions_norm

# 处理钙钛矿带隙特征
def process_pvk_bandgap(dataset):
    X_scaler = MinMaxScaler()
    pvk_bandgap = dataset.iloc[:, 24:25] 
    pvk_bandgap = pd.DataFrame(pvk_bandgap).fillna(pvk_bandgap.mean())
    X_scaler.fit(pvk_bandgap)
    pvk_bandgap_norm = X_scaler.transform(pvk_bandgap)
    pvk_bandgap_norm = pd.DataFrame(pvk_bandgap_norm, columns=['Bandgap'])
    return pvk_bandgap_norm

# 处理基底类型特征
def process_substrate(dataset):
    X_scaler = MinMaxScaler()
    substrate = dataset.iloc[:, 18:24]  
    substrate.fillna(substrate.mean(), inplace=True)
    X_scaler.fit(substrate)
    substrate_norm = X_scaler.transform(substrate)
    substrate_norm = pd.DataFrame(substrate_norm, columns=substrate.columns)
    return substrate_norm

# 处理HOMO_LUMO特征
def process_homo_lumo(dataset_homo_lumo):
    X_scaler = MinMaxScaler()
    homo_lumo = dataset_homo_lumo.iloc[:, 1:3]  
    homo_lumo.fillna(homo_lumo.mean(), inplace=True)
    X_scaler.fit(homo_lumo)
    homo_lumo_norm = X_scaler.transform(homo_lumo)
    homo_lumo_norm = pd.DataFrame(homo_lumo_norm, columns=['HOMO', 'LUMO'])
    return homo_lumo_norm

def dataset_process(dataset, features_to_include):
    # 直接使用传入的 DataFrame
    dataset = dataset

    feature_list = []

    if 'estate' in features_to_include:
        estate_features = calculate_estate_fingerprint(dataset=dataset)
        feature_list.append(estate_features)

    if 'pubchem' in features_to_include:
        pubchem_features = calculate_pubchem(dataset=dataset)
        feature_list.append(pubchem_features)

    if 'homo_lumo' in features_to_include:
        hl_path = "./SAM_data_PCE_15%_0721_with_HOMO_LUMO.csv"
        dataset_homo_lumo = pd.read_csv(hl_path)
        homo_lumo_features = process_homo_lumo(dataset_homo_lumo=dataset_homo_lumo)
        feature_list.append(homo_lumo_features)

    if 'pvk_ions' in features_to_include:
        pvk_ions_features = process_pvk_ions(dataset=dataset)
        feature_list.append(pvk_ions_features)

    if 'pvk_bandgap' in features_to_include:
        pvk_bandgap_features = process_pvk_bandgap(dataset=dataset)
        feature_list.append(pvk_bandgap_features)
        
    if 'substrate' in features_to_include:
        substrate_features = process_substrate(dataset=dataset)
        feature_list.append(substrate_features)
       
    if 'rdkit_descriptor' in features_to_include:
        rdkit_descriptor_features = calculate_rdkit_descriptor(dataset=dataset)
        feature_list.append(rdkit_descriptor_features)    

    if 'mordred_descriptor' in features_to_include:
        rdkit_descriptor_features = calculate_mordred_descriptor(dataset=dataset)
        feature_list.append(rdkit_descriptor_features) 

    if 'ecfp' in features_to_include:
        ecfp_features = calculate_ecfp(dataset=dataset)
        feature_list.append(ecfp_features) 

    if 'maccs' in features_to_include:
        maccs_features = calculate_maccs(dataset=dataset)
        feature_list.append(maccs_features) 

    final_features = pd.concat(feature_list, axis=1)
    print(final_features.shape)
    # 检查是否存在 'label' 列
    if 'label' in dataset.columns:
        processed_dataset = pd.concat([dataset["smiles"], final_features, dataset["label"]], axis=1)
    else:
        processed_dataset = pd.concat([dataset["smiles"], final_features], axis=1)

    return processed_dataset