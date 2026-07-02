from __future__ import absolute_import, division, print_function

import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import DataHub
from models import NNModel
from tasks import Trainer
from utils import YamlHandler
from utils import logger
from models.unimol import UniMolModel
from models.loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss, MAEwithNan
import matplotlib.pyplot as plt
import cv2

NNMODEL_REGISTER = {
    'unimolv1': UniMolModel,
}
LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': MAEwithNan,
}
ACTIVATION_FN = {
    # predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    # softmax is used for multiclass classification
    'multiclass': lambda x: F.softmax(x, dim=-1),
    'regression': lambda x: x,
    # sigmoid is used for multilabel classification
    'multilabel_classification': lambda x: F.sigmoid(x),
    # no activation function is used for multilabel regression
    'multilabel_regression': lambda x: x,
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}

class UNIMOL(object):
    def __init__(
        self,
        load_model = None,
        checkpoints_path = None,
        output_dim = 1,
        cuda = True,
        **params
    ):
        if not load_model:
            raise ValueError("load_model is empty")
        config_path = os.path.join(load_model, 'config.yaml')
        self.config = YamlHandler(config_path).read_yaml()
        if not checkpoints_path:
            raise ValueError("checkpoints_path is empty")
        self.config.target_cols = self.config.target_cols.split(',')
        self.config.load_model = load_model
        self.config.checkpoints_path = checkpoints_path
        self.config.output_dim = output_dim
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
        try:
            self.config.loss_key = self.config.loss_key
        except:
            self.config.loss_key = params.get('loss_key', None)
            if self.config.loss_key is None:
                self.config.loss_key = 'focal'
        if self.config.task == 'multilabel_classification':
            self.loss_func = LOSS_RREGISTER[self.config.task][self.config.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.config.task]
    def init_model(
        self,
        **params
    ):
        freeze_layers = params.get('freeze_layers', None)
        freeze_layers_reversed = params.get('freeze_layers_reversed', False)
        if self.config.model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[self.config.model_name](**params)
            if isinstance(freeze_layers, str):
                freeze_layers = freeze_layers.replace(' ', '').split(',')
            if isinstance(freeze_layers, list):
                for layer_name, layer_param in model.named_parameters():
                    should_freeze = any(layer_name.startswith(freeze_layer) for freeze_layer in freeze_layers)
                    layer_param.requires_grad = not (freeze_layers_reversed ^ should_freeze)
        else:
            raise ValueError('Unknown model: {}'.format(self.config.model_name))
        return model
    def get_model(self, **params):
        checkpoints_path = self.config.checkpoints_path
        model = self.init_model(**params)
        model.load_state_dict(torch.load(checkpoints_path, map_location=self.config.device)['model_state_dict'])
        model = model.to(self.config.device)
        return model

def GRAD_CAM(
    input_data,
    target_data,
    layer_name,
    model
):
    pre_target = model(input_data)
    loss_fn = nn.MSELoss()  # 假设使用均方误差损失，你可以根据实际需求调整
    loss = loss_fn(pre_target, target_data)
    dense_grad = None
    def get_dense_grad(grad):
        global dense_grad
        dense_grad = grad
    layer = model.get_submodule(layer_name)
    layer.weight.register_hook(get_dense_grad)
    loss.backward()
    return dense_grad

def grad_cam(model,
             metric,
             input,
             target,
             target_layer_name):
    model.eval()
    # 前向传播时保存特征图
    features = None
    gradients = None

    def save_features(module, input, output):
        nonlocal features
        features = output
        output.register_hook(save_gradients)

    def save_gradients(grad):
        nonlocal gradients
        gradients = grad
        
    # 获取 classification_head.dense 层，并注册钩子
    target_layer = model.get_submodule(target_layer_name)
    target_layer.register_forward_hook(save_features)
    
    # 钩子注册在特征图上以保存梯度
    # target_layer.register_hook(save_gradients)

    # 前向传播获取预测值
    pre_target = model(**input)

    # 计算损失
    loss = metric(pre_target, target)

    # 清零梯度
    model.zero_grad()

    # 反向传播，获取梯度
    loss.backward()

    # 获取 target_layer 的特征图和梯度
    # features 是前向传播时的特征图
    # gradients 是反向传播时计算的梯度
    gradients = gradients[0][1:-1, :]  # 梯度的形状是 (batch_size, channels, h, w)，我们取第一个样本
    features = features[0][1:-1, :]    # 同理，取第一个样本

    # 对梯度进行全局平均池化 (Global Average Pooling)
    weights = torch.mean(gradients, dim=(0,))  # 每个通道的权重 (形状是 [channels])

    # 使用权重加权特征图
    cam = torch.zeros(features.shape[0], dtype=torch.float32).to(features.device)  # 初始化 cam 图
    for i, w in enumerate(weights):
        cam += w * features[:, i]  # 权重加权通道

    # ReLU 激活函数
    cam_r = F.relu(cam)

    # 将 cam 缩放到 [0, 1]
    cam_r -= cam_r.min()
    cam_r /= cam_r.max()

    # 返回 Grad-CAM 热力图
    return cam_r.detach().cpu().numpy(), cam.detach().cpu().numpy()

# 生成 Grad-CAM 可视化图像
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    plt.imshow(np.uint8(255 * cam))
    plt.show()

# if __name__ == "__main__":
#     unimol = UNIMOL(load_model=r"./测试_dataset_bs_64", checkpoints_path="./测试_dataset_bs_64/model_0.pth")
#     model = unimol.get_model(**unimol.config)
#     layer_name = "classification_head.dense"
#     # layer = model.get_submodule(layer_name)
#     g_c = grad_cam(model=model, input=None, target=None, target_layer_name=layer_name)
#     print(g_c)