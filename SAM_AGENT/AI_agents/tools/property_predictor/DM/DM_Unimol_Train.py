from unimol_tools import MolTrain, MolPredict
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# nohup python ./DM/DM_Unimol_Train.py >> ./DM/0902_bs_32_lr_1e-4.log 2>&1 &
# 读取 mol_train.csv 文件
# mol_train = pd.read_csv("./SDF_SMILES_Prop/sdf_train_lumo.csv")
# mol_test = pd.read_csv("C:\\Users\\JIANi\\Downloads\\mol_test.csv")
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=50,
                learning_rate=0.0001,
                batch_size=32,
                early_stopping=5,
                metrics='mae',
                split='random',
                save_path='./Unimol_weights/DM/0902_bs_32_lr_1e-4',
                remove_hs=True
              )
clf.fit("./DM/0902_DM4ft.csv")

# import os
# import sys
# local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# print(local_dir)
# sys.path.append(local_dir)
# from Uni_Mol.unimol_tools.unimol_tools import MolTrain, MolPredict
# import pandas as pd
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # nohup python ./DM/DM_Unimol_Train.py >> ./DM/0901_bs_32_lr_1e-4.log 2>&1 &
# # 读取 mol_train.csv 文件
# # mol_train = pd.read_csv("./SDF_SMILES_Prop/sdf_train_lumo.csv")
# # mol_test = pd.read_csv("C:\\Users\\JIANi\\Downloads\\mol_test.csv")
# clf = MolTrain(task='regression',
#                 data_type='molecule',
#                 epochs=50,
#                 learning_rate=0.0001,
#                 batch_size=32,
#                 early_stopping=5,
#                 metrics='mae',
#                 split='random',
#                 save_path='./Unimol_weights/DM/0901_bs_32_lr_1e-4',
#                 remove_hs=True
#               )
# clf.fit("./DM/0901_DM4ft.csv")

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