from AI_agents.config.paths import GENERATED_DATA_CSV
from AI_agents.tools.molecular_informatics_tools.Mol_utils import canonical_smiles,Smiles2IUPAC_name, IUPAC_name2Smiles
from rdkit.Chem.Draw import MolsToGridImage
from rdkit import Chem
from rdkit.Chem import rdChemReactions # Add this import
from IPython.display import display
import pandas as pd


class Mol_tool:
    def __init__(self):
        self.generated_data_csv = GENERATED_DATA_CSV

    def load_generated_smiles(self):
        if not self.generated_data_csv.exists():
            raise FileNotFoundError(
                f"Generated molecules file not found: {self.generated_data_csv}. "
                "Run the Molecular_Generator tool first."
            )
        return pd.read_csv(self.generated_data_csv).SMILES
    
    
    # Decide which dataset to use and convert the SMILES to the canonical SMILES    
    def data_loader(self,smiles_list, generated=False):
        if generated:
            decided_list= self.load_generated_smiles()
        else:
            if smiles_list is None:
                raise ValueError("smiles_list must be provided if generated is False.")
            else:
                decided_list= smiles_list
            
        # Convert the list to Canonical SMILES
        Can_SMILES_list = []
        for i in decided_list:
            try:
                # Convert to canonical SMILES
                can_smiles = canonical_smiles(i)
                Can_SMILES_list.append(can_smiles)
            except Exception as e:
                # Handle the exception (e.g., print an error message)
                print(f"Error converting SMILES '{i}' to canonical SMILES: {e}")
                # Optionally, you can append None or some placeholder to Can_SMILES_list
                Can_SMILES_list.append(None)
         
        return Can_SMILES_list
    
    def name_loader(self, name_list, generated=False):
        if generated:
            decided_list= self.load_generated_smiles()
        else:
            if name_list is None:
                raise ValueError("name_list must be provided if generated is False.")
            else:
                decided_list= name_list
            
            
        return decided_list
    
   
    def Smiles2Name(self, data_list, generated=False):
        # Note! the input data should be iterative such as the dictionary or the list.
        canonical_smiles = self.data_loader(data_list, generated)
        IUPAC_names=[Smiles2IUPAC_name(i) for i in canonical_smiles]
        return IUPAC_names

    # Need to be rectified
    def Name2Smiles(self,data_list, generated=False):
        Names=self.name_loader(data_list, generated)
        SMILES=[IUPAC_name2Smiles(i) for i in Names]
        return SMILES
        
    def Mol2Image(self,smiles_list,generated=False,molsPerRow=5, display_limit=None):
        canonical_smiles = self.data_loader(smiles_list, generated) # Changed 'data' to 'smiles_list'
        if display_limit:
            if canonical_smiles:
                # Filter out None values that might have resulted from conversion errors in data_loader
                valid_smiles = [s for s in canonical_smiles if s is not None]
                if not valid_smiles:
                    return "No valid SMILES found after canonicalization."
                mols=[Chem.MolFromSmiles(i) for i in valid_smiles]
                # Filter out None values if MolFromSmiles failed
                mols = [m for m in mols if m is not None]
                if not mols:
                    return "Could not generate RDKit molecules from the provided SMILES."
                # 使用 SVG 格式避免 Cairo 依赖
                Images=MolsToGridImage(mols[:display_limit],molsPerRow=molsPerRow, useSVG=True)
                display(Images)
                return "Succeed visualization"
            
            else:
                return "No SMILES in the data, you need to convert the molecules to SMILES form"
        else:
            if canonical_smiles:
                # Filter out None values that might have resulted from conversion errors in data_loader
                valid_smiles = [s for s in canonical_smiles if s is not None]
                if not valid_smiles:
                    return "No valid SMILES found after canonicalization."
                mols=[Chem.MolFromSmiles(i) for i in valid_smiles]
                # Filter out None values if MolFromSmiles failed
                mols = [m for m in mols if m is not None]
                if not mols:
                    return "Could not generate RDKit molecules from the provided SMILES."
                # 使用 SVG 格式避免 Cairo 依赖
                Images=MolsToGridImage(mols,molsPerRow=molsPerRow, useSVG=True)
                display(Images)
                return "Succeed visualization"
            
            else:
                return "No SMILES in the data, you need to convert the molecules to SMILES form"
    
    def reaction2image(self,reaction_routes_str):
        """
        Parses a routes string, generates images for each reaction step, and displays them.
        """
        retrosynthetic_steps_str = reaction_routes_str.split('|')
        if not retrosynthetic_steps_str or not retrosynthetic_steps_str[0]:
            return "Input routes string is empty or invalid."

        forward_steps_info = []
        for retro_step_str in reversed(retrosynthetic_steps_str):
            parts = retro_step_str.split('>')
            if len(parts) < 3:
                print(f"Warning: Could not parse step: {retro_step_str}")
                continue
            
            forward_product_smiles = parts[0].strip()
            forward_reactants_smiles_str = parts[-1].strip()
            forward_reactant_smiles_list = [s.strip() for s in forward_reactants_smiles_str.split('.')]

            forward_steps_info.append({
                'reactants': forward_reactant_smiles_list,
                'products': [forward_product_smiles]
            })

        if not forward_steps_info:
            return "No valid reaction steps could be parsed from the input."

        images_generated_count = 0
        images_displayed_count = 0
        
        print(f"Found {len(forward_steps_info)} potential reaction steps.")

        steps_to_process = forward_steps_info
        
        for i, step_info in enumerate(steps_to_process):
            reactants_smi = ".".join(step_info['reactants'])
            products_smi = ".".join(step_info['products'])
            reaction_smarts = f"{reactants_smi}>>{products_smi}"

            print(f"\nProcessing Forward Reaction Step {i+1}:")
            print(f"  Reactants: {step_info['reactants']}")
            print(f"  Product:   {step_info['products'][0]}")
            print(f"  Reaction SMARTS: {reaction_smarts}")

            try:
                rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
                # Ensure Chem.Draw is used as Chem is imported.
                # 使用 SVG 格式避免 Cairo 依赖
                img = Chem.Draw.ReactionToImage(rxn, subImgSize=(400, 200), useSVG=True)
                images_generated_count +=1
                display(img)
                images_displayed_count +=1
                print("  Successfully generated and displayed image.")
            except Exception as e:
                print(f"  Error generating image for step {i+1}: {e}")
                print(f"  Problematic SMARTS: {reaction_smarts}")
        
        if images_displayed_count > 0:
            return f"Successfully displayed {images_displayed_count} of {images_generated_count} generated reaction images."
        elif images_generated_count > 0 and images_displayed_count == 0:
            return f"Generated {images_generated_count} reaction images, but failed to display them."
        elif forward_steps_info: # Parsed steps but failed to generate any images
            return "Parsed reaction steps, but failed to generate any images."
        else: # Should be caught earlier, but as a fallback
            return "No reaction images were generated or displayed."        
        
        
            
            
            
        
        
        

        