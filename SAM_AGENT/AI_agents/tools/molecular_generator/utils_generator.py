import re
import json
from rdkit import Chem
from rdkit.Chem import RDConfig, MolToSmiles, MolFromSmiles
import os
import sys
import threading
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer



def load_stoi(stoi_name):
    # 获取当前模块所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stoi_path = os.path.join(current_dir, f'{stoi_name}.json')
    
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def canonic_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    else:
        return None

def get_mol(smiles):
    return Chem.MolFromSmiles(smiles)


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()   
    model.eval()

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)   # for liggpt
        # logits, _, _ = model(x_cond)   # for char_rnn
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out