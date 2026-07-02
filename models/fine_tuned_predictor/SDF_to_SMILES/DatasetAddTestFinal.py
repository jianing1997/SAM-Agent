import pandas as pd

def addfile(dataset
            ,testfinal
            ,sdf2smiles):
    dataset = pd.read_csv(dataset)
    testfinal = pd.read_csv(testfinal)
    pd.concat([dataset, testfinal], axis=0).to_csv(sdf2smiles, index=False)

dataset = r"./SDF_to_SMILES/Dataset_invalid_smiles.csv"
testfinal = r"./SDF_to_SMILES/TestFinal_invalid_smiles.csv"
sdf2smiles = r"./SDF_to_SMILES/sdf_invalid_smiles.csv"
addfile(dataset, testfinal, sdf2smiles)

dataset = r"./SDF_to_SMILES/Dataset_valid_smiles.csv"
testfinal = r"./SDF_to_SMILES/TestFinal_valid_smiles.csv"
sdf2smiles = r"./SDF_to_SMILES/sdf_valid_smiles.csv"
addfile(dataset, testfinal, sdf2smiles)