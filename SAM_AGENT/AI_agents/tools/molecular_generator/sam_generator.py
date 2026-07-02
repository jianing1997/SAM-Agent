from pathlib import Path

from tqdm import tqdm
import pandas as pd
from AI_agents.config.paths import GENERATED_DATA_CSV
from AI_agents.tools.molecular_generator.model import GPT, GPTConfig
from AI_agents.tools.molecular_generator.utils_generator import load_stoi, canonic_smiles, get_mol, sample
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
import torch
import re


MODULE_DIR = Path(__file__).resolve().parent



class SAMGenerator:
    # init the paras of the model
    def __init__(self,scaf_condition,anchoring_group,gen_size):
        self.regex=re.compile("(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
        self.prop = None
        # 使用本地weights目录中的权重文件
        self.model_weight = str(MODULE_DIR / 'weights' / 'sam_related_data_transfer_learning_2.pt')
        self.scaffold = True
        self.list = False
        self.scaf_condition = scaf_condition#['c1ccccc1', 'c1ccc2c(c1)[nH]c1ccccc12']
        self.lstm = False
        self.context = anchoring_group # "O=P(O)(O)"
        self.csv_name = 'test.csv'
        # 使用SAM-GPT的词汇表和配置
        self.stoi_name = str(MODULE_DIR / 'weights' / '117_tokens_stoi')
        self.stoi, self.itos = load_stoi(self.stoi_name)
        self.batch_size = gen_size if gen_size < 100 else 100
        self.gen_size = gen_size
        self.vocab_size = 117  # SAM-GPT配置
        self.block_size = 100  # SAM-GPT配置
        self.props = []
        self.num_props = len(self.props)
        self.n_layer = 8
        self.n_head = 8
        self.n_embd = 256
        self.lstm_layers = 2
        self.model = self.load_model()
        
        
    def load_model(self):
        scaffold_max_len = 100 if self.scaffold else 0  # SAM-GPT配置
        num_props = len(self.props)
        config = GPTConfig(self.vocab_size, self.block_size, num_props = num_props,
                           n_layer=self.n_layer, n_head=self.n_head, n_embd = self.n_embd, scaffold = self.scaffold, scaffold_maxlen = scaffold_max_len,
                           lstm = self.lstm, lstm_layers = self.lstm_layers
                           )
        model = GPT(config)
        
        # 智能选择设备：优先使用 GPU，如果不可用则使用 CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_weight, map_location=self.device))
        model.to(self.device)
        
        device_name = 'GPU' if torch.cuda.is_available() else 'CPU'
        print(f'Model loaded on {device_name}')
        return model
    
    
    def _contains_input_scaffold(self, mol, scaf) -> bool:
        scaffold_smiles = scaf.replace('<', '') if scaf else ''
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaffold_mol is None:
            return False
        return mol.HasSubstructMatch(scaffold_mol)

    # Convert the medel logits to the text form.
    def process_output(self, y, scaf):
        molecules = []
        rejected_count = 0
        for gen_mol in y:
            completion = ''.join([self.itos[int(i)] for i in gen_mol]).replace('<', '')
            mol = get_mol(completion)
            if mol and self._contains_input_scaffold(mol, scaf):
                smiles = Chem.MolToSmiles(mol)
                scaffold_smiles = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
                mol_dict = {
                    'smiles': smiles,
                    'scaffold_condition': scaf.replace('<', '') if scaf else None,
                    'scaffold_smiles': scaffold_smiles
                }
                molecules.append(mol_dict)
            else:
                rejected_count += 1
        return molecules, rejected_count    
    
    
    def generate_with_scaffold(self):
        scaffold_max_len = 100 if self.scaffold else 0  # 必须与模型配置保持一致
        scaf_token = [ i + str('<')*(scaffold_max_len - len(self.regex.findall(i))) for i in self.scaf_condition]
        mol_dict = []
        seen_smiles = set()  # 用于跟踪已见过的 SMILES，确保不重复
        total_generated = 0
        
        print(f"开始生成，目标数量: {self.gen_size}, 批次大小: {self.batch_size}, scaffold数量: {len(scaf_token)}")
        
        # 持续生成直到获得足够数量的唯一有效分子
        max_total_generated = max(self.gen_size * 100, self.batch_size * len(scaf_token) * 5)
        while len(seen_smiles) < self.gen_size and total_generated < max_total_generated:
            for scaf in scaf_token:
                print(f"当前唯一有效分子数: {len(seen_smiles)}/{self.gen_size}, 总生成: {total_generated}")
                
                x = torch.tensor([self.stoi[s] for s in self.regex.findall(self.context)], dtype=torch.long)[None,...].repeat(self.batch_size, 1).to(self.device)
                sca = torch.tensor([self.stoi[s] for s in self.regex.findall(scaf)], dtype=torch.long)[None,...].repeat(self.batch_size, 1).to(self.device)
                y = sample(
                    self.model,
                    x,
                    self.block_size,
                    temperature=1,
                    sample=True,
                    top_k=10,
                    prop=None,
                    scaffold=sca
                )
                # Get valid molecules after scaffold filtering.
                valid_mols, rejected_count = self.process_output(y, scaf)
                total_generated += self.batch_size
                
                # 只添加新的、唯一的分子
                new_count = 0
                for mol in valid_mols:
                    smiles = mol['smiles']
                    if smiles not in seen_smiles:
                        seen_smiles.add(smiles)
                        mol_dict.append(mol)
                        new_count += 1
                
                print(f"本批: 生成 {self.batch_size} 个, 通过骨架过滤 {len(valid_mols)} 个, 过滤/无效 {rejected_count} 个, 新增唯一 {new_count} 个")
                
                # 如果已经获得足够的唯一分子，退出循环
                if len(seen_smiles) >= self.gen_size:
                    print(f"✓ 已达到目标数量 {self.gen_size} 个唯一有效分子")
                    break
            
            # 如果已经获得足够的唯一分子，退出外层循环
            if len(seen_smiles) >= self.gen_size:
                break
                

        if not mol_dict:
            raise RuntimeError("Failed to generate valid molecules within the generation limit.")

        results = pd.DataFrame(mol_dict)
        # 不需要再去重，因为已经在生成过程中保证唯一性
        results['scaffold_condition'] = results['scaffold_condition'].str.replace('<', '')
        
        # 精确截取到目标数量
        results = results.head(self.gen_size)
        
        print(f"=" * 60)
        print(f"✓ 最终输出: {len(results)} 个唯一有效分子 (目标: {self.gen_size})")
        print(f"  总生成次数: {total_generated}")
        print(f"  成功率: {len(results)/total_generated*100:.1f}%")
        if len(results) < self.gen_size:
            print(f"  警告: 达到生成上限，仅生成 {len(results)} 个唯一有效分子")
        print(f"=" * 60)
        
        return results
    
def generator_tool(gen_size, scaf_condition, anchoring_group: str = 'O=P(O)(O)'):
    canonical_scaffolds = []
    invalid_scaffolds = []
    for scaffold in scaf_condition:
        canonical = canonic_smiles(scaffold)
        if canonical:
            canonical_scaffolds.append(canonical)
        else:
            invalid_scaffolds.append(scaffold)

    if invalid_scaffolds:
        raise ValueError(
            "Invalid scaffold condition provided. "
            "Molecular_Generator only accepts scaffold SMILES. "
            "Please provide valid scaffold SMILES instead of molecule names or invalid scaffolds: "
            f"{invalid_scaffolds}"
        )
    if not canonical_scaffolds:
        raise ValueError(
            "Invalid scaffold condition provided. Please provide at least one valid scaffold SMILES."
        )
    
    # 如果用户没有指定锚定基团，使用默认的膦酸基团（与 SAM-GPT 训练数据一致）
    if not anchoring_group or anchoring_group.strip() == '':
        anchoring_group = 'O=P(O)(O)'
    
    # 注意：不对锚定基团标准化！保持用户输入的原始格式
    # 这与 SAM-GPT 原始实现一致，模型训练时使用的就是 'O=P(O)(O)' 格式
    # 如果标准化，会变成 'O=[PH](O)O'，与训练数据不一致
    
    generator = SAMGenerator(
        gen_size=gen_size,
        scaf_condition=canonical_scaffolds,
        anchoring_group=anchoring_group,
    )
    results = generator.generate_with_scaffold()
    smiles_df = pd.DataFrame({'SMILES': results['smiles']})
    smiles_df.to_csv(GENERATED_DATA_CSV, index=False)
    return results, str(GENERATED_DATA_CSV)



