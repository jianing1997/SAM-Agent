from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_mol
import re
import json
from rdkit.Chem import RDConfig
import json
import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# python ./generate/generate.py --model_weight ./weights_samgpt/moses2_bs_256_lr_6e-4.pt --scaffold --csv_name ./1107_1.csv --context 'O=P(O)(O)' --scaf 'c1ccc(N(c2ccccc2)c2ccc(-c3ccc4c(c3)Sc3cc(-c5ccc(N(c6ccccc6)c6ccccc6)cc5)ccc3N4)cc2)cc1' --repeat_times 1000 --batch_size 32 --predict_len 100 --temperature 1 --sample --top_k 10 --mol_csv ./1107.csv --data_stoi_name ./train/datasets/117_tokens_stoi.json
# python ./generate/generate.py --model_weight ./weights_samgpt/moses2_bs_256_lr_6e-4.pt --scaffold --csv_name ./sam_gen_source_model.csv --context 'O=P(O)(O)' --list './data/scaffold_for_gen.csv' --repeat_times 10000 --batch_size 32 --predict_len 100 --temperature 1 --sample --top_k 10 --mol_csv ./ratio_source_model.csv --data_stoi_name ./train/datasets/117_tokens_stoi.json
def get_bemis_murcko_scaffold(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        return Chem.MolToSmiles(scaffold) if scaffold else None
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {str(e)}")
        return None


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--data_name', type=str, default = 'moses2', help="name of the train dataset", required=False)
        parser.add_argument('--scaffold', action='store_true', default=False, help='whether to use scaffold conditions to generate')
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--csv_name', type=str, help="name to save the generated molecule in csv format", required=False)
        parser.add_argument('--mol_csv', type=str, help="name to save the ratio", required=False)
        parser.add_argument('--context', type=str, default='C', help="specifies the anchoring group of the SAM molecule", required=False)
        parser.add_argument('--scaf', type=str, default='N=c1[nH]c2ccccc2[nH]1', help="specifies the scaffold structure of the SAM molecule", required=False)
        parser.add_argument('--repeat_times', type=int, default=1, help="one SAM molecule product repeat times", required=True)
        parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
        parser.add_argument('--predict_len', type=int, default=200, help="predict max len", required=False)
        parser.add_argument('--temperature', type=float, default=1, help="temperature", required=False)
        parser.add_argument('--sample', action='store_true', default=False, help="sample", required=False)
        parser.add_argument('--top_k', type=int, help="top_k", required=False)
        parser.add_argument('--list', type=str, help="csv name of the scaffold list for molecule generation", required=False)
        parser.add_argument('--data_stoi_name', type=str, default = '117_tokens', help="name of the data_stoi", required=True)
        parser.add_argument('--vocab_size', type=int, default = 117, help="number of layers", required=False)
        parser.add_argument('--block_size', type=int, default = 100, help="number of layers", required=False)
        parser.add_argument('--scaffold_max_len', type=int, default = 100, help="scaffold max len", required=False)
        # parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
        parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)

        args = parser.parse_args()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        context = args.context
        
        if args.list:
            scaffold_cond_list = pd.read_csv(args.list)
            scaffold_cond_list = scaffold_cond_list.drop_duplicates(subset='scaffold_smiles', keep='first').reset_index(drop=True)
            scaffold_cond_list.columns = scaffold_cond_list.columns.str.lower()
            if args.scaffold_max_len:
                for i, d in enumerate(scaffold_cond_list.loc[:,"scaffold_smiles"]):
                    try:
                        if len(d) > args.scaffold_max_len:
                            scaffold_cond_list.loc[i,"scaffold_smiles"] = np.NaN
                    except:
                        continue
            scaffold_cond_list = scaffold_cond_list.dropna(axis=0).reset_index(drop=True)
            scaf_list = scaffold_cond_list.loc[:,"scaffold_smiles"].to_list()
        else:
            scaf_list = [args.scaf]

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        scaf_condition = [scaf + str('<')*(args.scaffold_max_len - len(regex.findall(scaf))) for scaf in scaf_list]

        scaf_condition = [regex.findall(scaf) for scaf in scaf_condition]

        stoi = json.load(open(f"{args.data_stoi_name}", "r"))
        print(stoi)
        itos = { i:ch for ch, i in stoi.items() }

        num_props = len(args.props)
        mconf = GPTConfig(args.vocab_size, args.block_size, num_props=num_props,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                          scaffold=args.scaffold, scaffold_maxlen=args.scaffold_max_len,
                          lstm=args.lstm, lstm_layers=args.lstm_layers)
        model = GPT(mconf)

        model.load_state_dict(torch.load(args.model_weight))
        model.to(device)
        print("Model loaded")

        num_0 = args.repeat_times // args.batch_size
        num_1 = args.repeat_times % args.batch_size
        if num_1:
            num_0 += 1
        
        # 记录总体开始时间
        total_start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"开始生成分子: {start_datetime}")
        
        mol_dict = []
        total_molecules_generated = 0
        
        for idx_, scaf in enumerate(scaf_condition):
            scaffold_start_time = time.time()
            print(f"\n处理第 {idx_+1}/{len(scaf_condition)} 个scaffold: {scaf_list[idx_]}")
            
            for i in tqdm(range(num_0)):
                batch_start_time = time.time()
                
                if (i < num_0 - 1):
                    batch_size = args.batch_size
                elif num_1:
                    batch_size = num_1
                else:
                    batch_size = args.batch_size
                    
                x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(batch_size, 1).to(device)
                p = None
                sca = torch.tensor([stoi[s] for s in scaf], dtype=torch.long)[None,...].repeat(batch_size, 1).to(device)
                
                # 记录采样时间
                sample_start_time = time.time()
                y = sample(model, x, args.block_size, temperature=args.temperature, sample=args.sample, top_k=args.top_k, prop=p, scaffold=sca)
                sample_time = time.time() - sample_start_time
                
                for gen_mol in y:
                    mol_start_time = time.time()
                    completion = "".join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace('<', '')
                    mol = get_mol(completion)
                    if mol:
                        predict_smiles = Chem.MolToSmiles(mol)
                        bm_scaffold = get_bemis_murcko_scaffold(predict_smiles)
                        scaffold_smiles = scaf_list[idx_]
                        mol_time = time.time() - mol_start_time
                        total_molecules_generated += 1
                        mol_dict.append({
                            'scaffold_smiles': scaffold_smiles, 
                            'molecule': mol, 
                            'predict': completion, 
                            'smiles': predict_smiles, 
                            'bm_scaffold': bm_scaffold,
                            'generation_time': mol_time,
                            'batch_id': i,
                            'scaffold_id': idx_
                        })
                
                batch_time = time.time() - batch_start_time
            
            scaffold_time = time.time() - scaffold_start_time
            print(f"Scaffold {idx_+1} 完成，耗时: {scaffold_time:.2f}秒")
                    
            result = pd.DataFrame(mol_dict)
            result.to_csv(args.csv_name)
        
        # 计算总体时间
        total_time = time.time() - total_start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*60}")
        print(f"分子生成完成!")
        print(f"开始时间: {start_datetime}")
        print(f"结束时间: {end_datetime}")
        print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"生成的有效分子数: {total_molecules_generated}")
        if total_molecules_generated > 0:
            print(f"平均每个分子生成时间: {total_time/total_molecules_generated:.4f}秒")
        print(f"{'='*60}\n")
        
        scaffold_smiles_drop = result.drop_duplicates(subset='scaffold_smiles', keep='first').reset_index(drop=True)['scaffold_smiles'].to_list()
        mol_result = []
        for s_s in scaffold_smiles_drop:
            compute_data = result[result.loc[:,"scaffold_smiles"]==s_s]
            compute_data_drop = compute_data.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
            Satisfied_scaffold_condition_ratio = np.round(list(compute_data_drop.loc[:,"scaffold_smiles"]==compute_data_drop.loc[:,"bm_scaffold"]).count(True)/len(compute_data_drop), 3)
            Valid_ratio = np.round(len(compute_data)/(args.repeat_times), 3)
            unique_smiles = list(set([canonic_smiles(s) for s in compute_data.loc[:,'smiles']]))
            Unique_ratio = np.round(len(unique_smiles)/len(compute_data), 3)
            sam_data = pd.read_csv("./train/datasets/sam4tl_len_100.csv")
            novel_ratio = check_novelty(unique_smiles, set(sam_data[sam_data['split']=='train']['smiles']))
            
            # 计算该scaffold的平均生成时间
            scaffold_gen_times = compute_data['generation_time'].values
            avg_gen_time = np.mean(scaffold_gen_times) if len(scaffold_gen_times) > 0 else 0
            
            mol_result.append({
                'scaffold_smiles': s_s, 
                'Satisfied_scaffold_condition_ratio': Satisfied_scaffold_condition_ratio, 
                'Valid_ratio': Valid_ratio, 
                'Unique_ratio': Unique_ratio, 
                'novel_ratio': novel_ratio,
                'avg_generation_time': np.round(avg_gen_time, 6),
                'total_molecules': len(compute_data)
            })

            pd.DataFrame(mol_result).to_csv(args.mol_csv)
        
        # 保存时间统计信息到单独的文件
        if args.csv_name:
            time_log_file = args.csv_name.replace('.csv', '_time_log.txt')
            with open(time_log_file, 'w', encoding='utf-8') as f:
                f.write(f"分子生成时间统计报告\n")
                f.write(f"{'='*60}\n")
                f.write(f"开始时间: {start_datetime}\n")
                f.write(f"结束时间: {end_datetime}\n")
                f.write(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
                f.write(f"生成的有效分子数: {total_molecules_generated}\n")
                if total_molecules_generated > 0:
                    f.write(f"平均每个分子生成时间: {total_time/total_molecules_generated:.4f}秒\n")
                f.write(f"{'='*60}\n")
            print(f"时间统计已保存到: {time_log_file}")