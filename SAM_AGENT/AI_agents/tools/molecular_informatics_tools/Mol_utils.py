import requests
from rdkit import Chem, DataStructs, RDLogger
import pandas as pd
import urllib


############################################################################################################
def canonical_smiles(smiles):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"
    
def is_smiles(s):
    try:
        # Suppress RDKit error messages
        RDLogger.DisableLog('rdApp.error')
        mol = Chem.MolFromSmiles(s)
        smiles = Chem.MolToSmiles(mol) if mol else None
        return smiles is not None
    except Exception:
        return False
############################################################################################################
# This function converts the  SMILES to the IUPAC name
def Smiles2IUPAC_name(smiles):
    base_url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON'
    try:
        url=base_url.format(smiles)
        response = requests.get(url)
        response.raise_for_status()
        properties = response.json().get('PropertyTable', {}).get('Properties', [])
        if not properties:
            return "No IUPAC name found for the provided SMILES."
        
        iupac_name = properties[0].get('IUPACName', 'No name found')
        return iupac_name
    
    except requests.exceptions.RequestException as e:
        return "Error"
        
# THis function converts the IUPAC name to the SMILES         
def IUPAC_name2Smiles(name):
    # URL encode the chemical name to handle special characters
    encoded_name = urllib.parse.quote(name)
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/JSON'
    
    try:
        response = requests.get(url, timeout=15)  # Add timeout to prevent hanging
        response.raise_for_status()
        properties = response.json().get('PropertyTable', {}).get('Properties', [])
        
        if not properties:
            return "No SMILES found for the provided name."
        
        SMILES = properties[0].get('CanonicalSMILES', 'No SMILES found')
        return SMILES

    except requests.exceptions.RequestException as e:
        return f"Error: SMILES conversion failure: {str(e)}"  # Return more specific error information
    except ValueError as e:  # Handle JSON parsing errors
        return f"Error: SMILES conversion failure: {str(e)}"
    
############################################################################################################
    
    



######################################################################


#cleans the purity text
def process_purity(value):
    if value == "('',)":
        return ""
    else:
        return value.strip('\'(%\'),')

# standardize the price and purity columns with chemspace data   
def molport_standardize_columns(data):
    
    data = data.astype(str)
    
    # Apply the custom function to the 'Purity' column
    data['Purity'] = data['Purity'].apply(process_purity)

    # Create new columns Price_USD and Price_EUR with empty strings
    data['Price_USD'] = ""

    # Replace relevant values with corresponding prices
    data.loc[data['Currency'] == 'USD', 'Price_USD'] = data.loc[data['Currency'] == 'USD', 'Price']

    # Remove the Price and Currency columns
    data.drop(['Price', 'Currency'], axis=1, inplace=True)
    
    return data


######################################################################
######################################################################

def Smiles2MolPort_id(smiles_list,API_key):
    id_smiles_list = []
    for smiles in smiles_list:
        payload={
            "API Key":API_key,
	        "Structure":smiles,
            "Search Type":5,# Perfect search for the mol
            "Maximum Search Time":60000,
            "Maximum Result Count":10000,
            "Chemical Similarity Index":0.9
            }
    
        r = requests.post('https://api.molport.com/api/chemical-search/search', json=payload)
        #Retrieve Python dict file from server response
        response = r.json()
        molecules = response["Data"]["Molecules"]

        # Iterate over the molecules and their information
        for molecule in molecules:
            molecule_id = molecule["Id"]
            id_smiles_list.append((molecule_id, smiles))
                
    df = pd.DataFrame(id_smiles_list, columns=["ID", "Input SMILES"])
    return df

def MolPort_info(API_key, input_list=None):
    # Process inputs - check if each is SMILES, if not try to convert from IUPAC name
    processed_smiles = []
    for item in input_list:
        if is_smiles(item):
            processed_smiles.append(canonical_smiles(item))        
        else:
            # Try to convert from IUPAC name to SMILES
            converted_smiles = IUPAC_name2Smiles(item)
            if converted_smiles != "Error" and converted_smiles != "No SMILES found for the provided name.":
                processed_smiles.append(converted_smiles)
            else:
                print(f"Invalid input: {item} is neither a valid SMILES nor a convertible IUPAC name.")
                continue
    
    # Skip if no valid SMILES were found
    if not processed_smiles:
        return pd.DataFrame()
        
    mol_ids = Smiles2MolPort_id(processed_smiles, API_key)
    
    if mol_ids.empty:
        return "No molecules found in MolPort database for the provided SMILES."
    
    # Build result from search data (molecule/load API returns empty for this key)
    result_rows = []
    for _, row in mol_ids.iterrows():
        molport_id = row["ID"]
        input_smi = row["Input SMILES"]
        molport_url = f"https://www.molport.com/shop/molecule-link/load-molecule/{molport_id}"
        result_rows.append({
            "Input SMILES": input_smi,
            "MolPort ID": molport_id,
            "MolPort URL": molport_url,
            "Availability": "Available (check MolPort for pricing)"
        })
    
    return pd.DataFrame(result_rows)
    
    
    
