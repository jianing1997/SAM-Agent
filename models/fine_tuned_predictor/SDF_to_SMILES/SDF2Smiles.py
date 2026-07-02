import argparse
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
from tqdm import tqdm
import pandas as pd

def sdf_to_smiles(sdf_file
                  ,valid_smiles
                  ,invalid_smiles):
    try:
        mol = Chem.MolFromMolFile(sdf_file)
        smi = Chem.MolToSmiles(mol)
        valid_smiles.append({"SMILES":smi, "name":os.path.basename(sdf_file)})
    except:
        invalid_smiles.append({"SMILES":"", "name":os.path.basename(sdf_file)})
    return valid_smiles, invalid_smiles

def process_sdf_folder(folder_path):
    sdf_files = [f for f in os.listdir(folder_path) if f.endswith('.sdf')]
    valid_smiles = []
    invalid_smiles = []
    for sdf_file in sdf_files:
        sdf_file_path = os.path.join(folder_path, sdf_file)
        valid_smiles, invalid_smiles = sdf_to_smiles(sdf_file_path, valid_smiles, invalid_smiles)
    return valid_smiles, invalid_smiles

def save_to_csv(data, filename):
    data_pd = pd.DataFrame(data)
    data_pd.to_csv(filename,index=False)

if __name__ == "__main__":
    # nohup python ./SDF_to_SMILES/SDF2Smiles.py --folder_path ./SDF_to_SMILES/TestFinal --invalid_csv ./SDF_to_SMILES/TestFinal_invalid_smiles.csv --valid_csv ./SDF_to_SMILES/TestFinal_valid_smiles.csv >> ./SDF_to_SMILES/TestFinal.log 2>&1 &
    # nohup python ./SDF_to_SMILES/SDF2Smiles.py --folder_path ./SDF_to_SMILES/Dataset --invalid_csv ./SDF_to_SMILES/Dataset_invalid_smiles.csv --valid_csv ./SDF_to_SMILES/Dataset_valid_smiles.csv >> ./SDF_to_SMILES/Dataset.log 2>&1 &
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str,
                        help="Path to the folder containing SDF files", required=True)
    parser.add_argument('--invalid_csv', type=str,
                        help="Filename to save invalid SMILES information (CSV)", required=True)
    parser.add_argument('--valid_csv', type=str,
                        help="Filename to save valid SMILES information (CSV)", required=True)

    args = parser.parse_args()

    folder_path = args.folder_path
    invalid_csv = args.invalid_csv
    valid_csv = args.valid_csv

    valid_smiles, invalid_smiles = process_sdf_folder(folder_path)

    # 打印无法转换的分子信息
    if len(invalid_smiles)>0:
        print("\nUnable to convert the following molecules to SMILES:")
        for mol_info in invalid_smiles:
            print(f"Molecule err in file: {mol_info['name']}")

    # 将无效和有效 SMILES 分别保存到 CSV 文件中
    if invalid_smiles:
        save_to_csv(invalid_smiles, invalid_csv)
        print(f"Invalid SMILES information saved to '{invalid_csv}'")
    if valid_smiles:
        save_to_csv(valid_smiles, valid_csv)
        print(f"Valid SMILES information saved to '{valid_csv}'")
