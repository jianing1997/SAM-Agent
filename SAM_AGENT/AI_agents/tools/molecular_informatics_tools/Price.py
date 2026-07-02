import os

from AI_agents.tools.molecular_informatics_tools.Mol_utils import Smiles2MolPort_id,MolPort_info, canonical_smiles
import pandas as pd

class Molinfo:
    def __init__(self):
        self.api_key = os.environ.get("MOLPORT_API_KEY", "")
        self.gen_smiles_list=[]
    
    # Decide which dataset to use and convert the SMILES to the canonical SMILES    
    def data_loader(self,smiles_list=None, generated=False):
        if generated:
            decided_list = self.gen_smiles_list
        else:
            if smiles_list is None:
                raise ValueError("smiles_list must be provided if generated is False.")
            elif isinstance(smiles_list, list):
                decided_list = smiles_list
            else:
                decided_list = [smiles_list]
            
        # Convert the list to Canonical SMILES
        # Can_SMILES_list=[canonical_smiles(i) for i in decided_list]
            
        return decided_list
        
        
        
        
    def find_ids(self, mol_list=None, generated=False):
        if not self.api_key:
            raise ValueError("MOLPORT_API_KEY is required for MolPort supplier lookup.")
        canonical_smiles = self.data_loader(mol_list, generated)
        ids = Smiles2MolPort_id(smiles_list=canonical_smiles, API_key=self.api_key)
        return ids
        
        
    def collect_info(self, mol_list=None, generated=False):
        if not self.api_key:
            raise ValueError("MOLPORT_API_KEY is required for MolPort supplier lookup.")
        canonical_smiles_list = self.data_loader(mol_list, generated)
        try:
            info = MolPort_info(input_list=canonical_smiles_list, API_key=self.api_key)
            if info is None or (isinstance(info, pd.DataFrame) and info.empty):
                return ("No supplier information found for the provided molecules. "
                        "The MolPort API may be unavailable or the molecules may not be listed.")
            return info
        except Exception as e:
            return f"Supplier lookup failed: {str(e)}. The MolPort API may be temporarily unavailable."



    

    
    
    
    
    
    
    
    
