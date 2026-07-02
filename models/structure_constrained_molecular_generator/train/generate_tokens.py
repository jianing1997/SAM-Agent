import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import re

file_path = r"./train/df_HL_SAM_for_transfer_learning_moses.csv"
data = pd.read_csv(file_path)
data_zdy = pd.read_csv(r"./train/tl_moses.csv")
data_o = pd.read_csv(r"./train/moses2.csv")
# print(data.loc[:,"smiles"])
# print(data.loc[:,"scaffold_smiles"])
scaf_list = data.loc[:, "scaffold_smiles"].to_list()
pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)
scaf_condition = [scaf + str("<")*(100 - len(regex.findall(scaf))) for scaf in scaf_list]
scaf_condition = [regex.findall(scaf) for scaf in scaf_condition]

data_o = data_o.dropna(axis=0).reset_index(drop=True)
data_zdy = data_zdy.dropna(axis=0).reset_index(drop=True)

data_o_ss = list(data_o.loc[:, "SMILES"].values) + list(data_o.loc[:, "scaffold_smiles"].values)
data_zdy_ss = list(data_zdy.loc[:, "smiles"].values) + list(data_zdy.loc[:, "scaffold_smiles"].values)
data_ss = list(data.loc[:, "smiles"].values) + list(data.loc[:, "scaffold_smiles"].values)

o_l = ["<"]
for id_, seq in enumerate(data_o_ss):
    r_seq = regex.findall(seq.strip())
    for i in r_seq:
        if i not in o_l:
            o_l.append(i)

zdy_l = ["<"]
for id_, seq in enumerate(data_zdy_ss):
    r_seq = regex.findall(seq.strip())
    for i in r_seq:
        if i not in zdy_l:
            zdy_l.append(i)

d_l = ["<"]
for id_, seq in enumerate(data_ss):
    r_seq = regex.findall(seq.strip())
    for i in r_seq:
        if i not in d_l:
            d_l.append(i)

whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']

all_ = sorted(list(set(o_l + zdy_l + d_l + whole_string)))

print(all_)
print(len(all_))
