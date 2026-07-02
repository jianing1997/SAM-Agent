from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from data import DataHub
from models import NNModel
from tasks import Trainer
from utils import YamlHandler
from utils import logger
from utils.util import pad_coords, pad_2d, pad_1d_tokens
import torch
from torch.utils.data import Dataset,  DataLoader as TorchDataLoader
from data import Dictionary
from config import MODEL_CONFIG
from weights import WEIGHT_DIR
from models.nnmodel import TorchDataset

class Data(object):
    def __init__(self,
                 load_model = None,
                 batch_size = None):
        if not load_model:
            raise ValueError("load_model is empty")
        config_path = os.path.join(load_model, 'config.yaml')
        self.config = YamlHandler(config_path).read_yaml()
        self.config.target_cols = [self.config.target_cols]
        if batch_size:
            self.config.batch_size = batch_size
        self.save_path = load_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.config.cuda else "cpu")
        self.task = self.config.task
        if self.config.data_type == 'molecule':
            name = "no_h" if self.config.remove_hs else "all_h"
            name = self.config.data_type + '_' + name
        else:
            name = self.config.data_type
        self.config.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, MODEL_CONFIG['dict'][name]))
        self.config.dictionary.add_symbol("[MASK]", is_special=True)
    def get_data(self,
                data):
        datahub = DataHub(data = data, is_train=True, save_path=self.save_path, **self.config)
        self.data = datahub.data
        self.X = np.asarray(datahub.data['unimol_input'])
        self.y = np.asarray(datahub.data['target'])
        return self.data, self.X, self.y
    def get_batch_data(self):
        dataset = TorchDataset(self.X, self.y)
        dataloader = TorchDataLoader(dataset=dataset,
                                     batch_size=self.config.batch_size,
                                     shuffle=False,
                                     collate_fn=self.batch_collate_fn)
        return dataloader
    def batch_collate_fn(self,
                         samples):
        batch = {}
        padding_idx = self.config.dictionary.pad()
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=padding_idx)
            batch[k] = v
        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        return batch, label
    def decorate_batch(self, batch, feature_name=None):
        return self.decorate_torch_batch(batch)
    def decorate_torch_batch(self, batch):
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(self.device)}, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

# if __name__ == "__main__":
#     load_model = r"./测试_dataset_bs_64"
#     batch_size = 20
#     data_file = r"/home/dell/af/测试/测试数据.csv"
#     datahub = Data(load_model=load_model, batch_size=batch_size)
#     data, X, y = datahub.get_data(data_file)
#     dataloader = datahub.get_batch_data()
#     for i, batch in enumerate(dataloader):
#         net_input, net_target = datahub.decorate_torch_batch(batch)
#     print(data)
        
        
        
        
        
        
        
        
        