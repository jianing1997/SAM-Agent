from pathlib import Path
from typing import List

from AI_agents.config.paths import GENERATED_DATA_CSV, SMILES_INPUT_CSV
from AI_agents.tools.property_predictor.unimol_tools.predict import MolPredict
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent


def input_form(smiles_list: List[str]) -> str:
    if not isinstance(smiles_list, list):
        raise ValueError("Input should be a list of SMILES strings.")
    df = pd.DataFrame({"SMILES": smiles_list})
    df.to_csv(SMILES_INPUT_CSV, index=False)
    return str(SMILES_INPUT_CSV)
    


class Predictor:
    def __init__(self):
        self.HOMO_dir=str(MODULE_DIR / 'homo_bs_32_lr_1e-4')
        self.LUMO_dir=str(MODULE_DIR / 'lumo_bs_32_lr_1e-4')
        self.DM_dir=str(MODULE_DIR / 'dm_bs_32_lr_1e-4')
        
    def HOMO_pred(self,smiles,generated):
        if generated:
            smiles_dir=str(GENERATED_DATA_CSV)
        else:
            smiles_dir=input_form(smiles)
        HOMO_predictor=MolPredict(load_model=self.HOMO_dir)
        HOMO_pred=HOMO_predictor.predict(smiles_dir)
        return HOMO_pred
    
    
    def LUMO_pred(self,smiles,generated):
        if generated:
            smiles_dir=str(GENERATED_DATA_CSV)
        else:
            smiles_dir=input_form(smiles)
        LUMO_predictor=MolPredict(load_model=self.LUMO_dir)
        LUMO_pred=LUMO_predictor.predict(smiles_dir)
        return LUMO_pred
    
    def DM_pred(self,smiles,generated):
        if generated:
            smiles_dir=str(GENERATED_DATA_CSV)
        else:
            smiles_dir=input_form(smiles) 
        DM_predictor=MolPredict(load_model=self.DM_dir)
        DM_pred=DM_predictor.predict(smiles_dir)
        return DM_pred
    
    def prop_pred(self, smiles, generated, HOMO=False, LUMO=False, DM=False):
        results = {}
        if HOMO:
            results['HOMO'] = self.HOMO_pred(smiles, generated)
        if LUMO:
            results['LUMO'] = self.LUMO_pred(smiles, generated)
        if DM:
            results['DM'] = self.DM_pred(smiles, generated)
        return results
                
            
    

   