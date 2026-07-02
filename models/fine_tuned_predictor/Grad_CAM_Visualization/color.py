import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

data_file = r"./visualization/result/homo_cams.csv"
data = pd.read_csv(data_file)
save_file = r"./visualization/homo_colored_mol"
if not os.path.exists(save_file):
    os.makedirs(save_file)


def draw(smiles,colors,save_file):
    # norm = plt.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('YlGn')
    mol = Chem.MolFromSmiles(smiles)
    # AllChem.Compute2DCoords(mol)
    atom_colors = {}
    colors = np.fromstring(colors.strip('[]'), sep=' ')
    for i, value in enumerate(colors):
        rgba_color = cmap(value)
        rgb_color = (rgba_color[0], rgba_color[1], rgba_color[2])  # 转为 RGB 三元组
        # rgb_color = (int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
        atom_colors[i] = rgb_color
    highlight_atoms = list(atom_colors.keys())
    highlight_atom_colors = {atom_idx: color for atom_idx, color in atom_colors.items()}
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(save_file)
    
for i in range(data.shape[0]):
    save_atoms_path = os.path.join(save_file, data.loc[i, "SMILES"])
    if not os.path.exists(save_atoms_path):
        os.makedirs(save_atoms_path)
    draw(data.loc[i, "SMILES"], data.loc[i, "MODEL_0"], os.path.join(save_atoms_path, "MODEL_0.png"))
    draw(data.loc[i, "SMILES"], data.loc[i, "MODEL_1"], os.path.join(save_atoms_path, "MODEL_1.png"))
    draw(data.loc[i, "SMILES"], data.loc[i, "MODEL_2"], os.path.join(save_atoms_path, "MODEL_2.png"))
    draw(data.loc[i, "SMILES"], data.loc[i, "MODEL_3"], os.path.join(save_atoms_path, "MODEL_3.png"))
    draw(data.loc[i, "SMILES"], data.loc[i, "MODEL_4"], os.path.join(save_atoms_path, "MODEL_4.png"))
    draw(data.loc[i, "SMILES"], data.loc[i, "MODEL_ALL"], os.path.join(save_atoms_path, "MODEL_ALL.png"))
