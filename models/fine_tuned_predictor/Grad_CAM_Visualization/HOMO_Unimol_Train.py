from train import MolTrain
from predict import MolPredict
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# nohup python ./Train_props/HOMO_Unimol_Train.py >> ./Train_props/HOMO.log 2>&1 &
# 读取 mol_train.csv 文件
# mol_train = pd.read_csv("./SDF_SMILES_Prop/sdf_train_lumo.csv")
# mol_test = pd.read_csv("C:\\Users\\JIANi\\Downloads\\mol_test.csv")
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=10,
                learning_rate=0.0001,
                batch_size=128,
                early_stopping=5,
                metrics='mae',
                split='random',
                save_path='/home/dell/af/cs/cs_dataset_bs_64',
                remove_hs=True
              )

clf.fit("/home/dell/af/测试/测试数据.csv")
# clf.fit("/home/dell/af/SDF_SMILES_Prop/Dataset_train_homo.csv")

# MODEL_CONFIG = {
#     "weight":{
#         "protein": "poc_pre_220816.pt",
#         "molecule_no_h": "mol_pre_no_h_220816.pt",
#         "molecule_all_h": "mol_pre_all_h_220816.pt",
#         "crystal": "mp_all_h_230313.pt",
#         "oled": "oled_pre_no_h_230101.pt",
#     },
#     "dict":{
#         "protein": "poc.dict.txt",
#         "molecule_no_h": "mol.dict.txt",
#         "molecule_all_h": "mol.dict.txt",
#         "crystal": "mp.dict.txt",
#         "oled": "oled.dict.txt",
#     },
# }