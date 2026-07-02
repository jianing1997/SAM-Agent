import argparse
import os
import torch
import sys


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu', type=int, default=1)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1234)

# ==================== dataset ===================== #
parser.add_argument('--test_routes',
                    default='dataset/routes_possible_test_hard.pkl')
parser.add_argument('--starting_molecules', default='dataset/origin_dict.csv')

# ================== value dataset ================= #
parser.add_argument('--value_root', default='dataset')
parser.add_argument('--value_train', default='train_mol_fp_value_step')
parser.add_argument('--value_val', default='val_mol_fp_value_step')

# ================== one-step model ================ #
parser.add_argument('--mlp_model_dump',
                    default='one_step_model/saved_rollout_state_1_2048.ckpt')
parser.add_argument('--mlp_templates',
                    default='one_step_model/template_rules_1.dat')

# ===================== all algs =================== #
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--expansion_topk', type=int, default=50)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--viz_dir', default='viz')

# ===================== model ====================== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_epoch_int', type=int, default=1)
parser.add_argument('--save_folder', default='saved_models')

# ==================== evaluation =================== #
parser.add_argument('--use_value_fn', action='store_true')
parser.add_argument('--value_model', default='best_epoch_final_4.pt')
parser.add_argument('--result_folder', default='results')

if __name__ == '__main__':
    # This block runs if the script is executed directly (e.g., "python parse_args.py")
    # It will parse command-line arguments from sys.argv
    args = parser.parse_args()
else:
    # This block runs if the script is imported as a module
    # (e.g., "from ...common.parse_args import args" in your Jupyter notebook)
    # We parse an empty list, so all arguments will take their default values.
    # This prevents conflicts with arguments from the importing environment (like Jupyter).
    args = parser.parse_args([])

# setup device - 注释掉CUDA设置以避免GPU错误
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
