from unimol_tools import MolTrain, MolPredict
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# nohup python ./Train_props/HOMO_Unimol_Train.py >> ./Train_props/HOMO.log 2>&1 &
# 读取 mol_train.csv 文件
# mol_train = pd.read_csv("./SDF_SMILES_Prop/sdf_train_lumo.csv")
# mol_test = pd.read_csv("C:\\Users\\JIANi\\Downloads\\mol_test.csv")
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=50,
                learning_rate=0.001,
                batch_size=32,
                early_stopping=5,
                metrics='mae',
                split='random',
                save_path='./weights_fix_0226/lumo/bs_32_lr_1e-3',
                remove_hs=True,
                dropout=0
              )
clf.fit("./SDF_SMILES_Prop/Dataset_train_lumo.csv")