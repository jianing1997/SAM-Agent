from DATA import Data
from MODEL import UNIMOL, grad_cam
import torch
import os
import numpy as np
import pandas as pd
from torch.nn import functional as F
import torch


def print_layer_outputs(model, input_data):
    hooks = []

    # 定义一个钩子函数，用于打印层名和输出的 shape
    def hook_fn(module, input, output):
        class_name = module.__class__.__name__  # 获取层的类型名称
        if isinstance(output, torch.Tensor):
            print(f"Layer: {class_name}, Output Shape: {output.shape}")
        elif isinstance(output, (list, tuple)):  # 处理多个输出的情况
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"Layer: {class_name}, Output {i} Shape: {out.shape}")
        else:
            print(f"Layer: {class_name}, Output: {output}")

    # 给每一层注册一个 forward hook
    for name, layer in model.named_modules():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # 前向传播以触发钩子
    with torch.no_grad():  # 禁用梯度计算，提升推理速度
        model(**input_data)

    # 取消钩子
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    # load_model = r"./测试_dataset_bs_64"
    columns = ["SMILES", "TARGET", "MODEL_0", "MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4", "MODEL_ALL"]
    data_save = pd.DataFrame(columns = columns)
    load_model = r"./visualization/homo_weights"
    model_paths = [os.path.join(load_model, p) for p in os.listdir(load_model) if p.endswith(".pth")]
    batch_size = 1
    # data_file = r"/home/dell/af/测试/测试数据.csv"
    data_file = r"./visualization/test_data/sam_atten_test_select.csv"
    data_info = pd.read_csv(data_file)
    save_file = r"./visualization/result/sam_atten_test_select_cams.csv"
    # checkpoints_path = r"./测试_dataset_bs_64/model_1.pth"
    unimols, metrics, models = [], [], []
    for checkpoints_path in model_paths:
        unimol = UNIMOL(load_model=load_model, checkpoints_path=checkpoints_path)
        metric = unimol.loss_func
        model = unimol.get_model(**unimol.config)
        model.eval()
        unimols.append(unimol)
        metrics.append(metric)
        models.append(model)
    datahub = Data(load_model=load_model, batch_size=batch_size)
    data = datahub.get_data(data=data_file)
    dataloader = datahub.get_batch_data()
    atoms_info = []
    for i, batch in enumerate(dataloader):
        net_input, net_target = datahub.decorate_torch_batch(batch)
        # with torch.no_grad():
        #     pred = model(**net_input)
        # for name, module in model.named_modules():
        #     print(f"Direct Layer Name: {name}")
        cams = []
        c_info = []
        for j in range(len(unimols)):
            cam, c = grad_cam(model=models[j], metric=metrics[j], input=net_input, target=net_target, target_layer_name="encoder.layers.7.self_attn_layer_norm")
            cams.append(cam)
            c_info.append(c)
        cams = np.array(cams)
        c_info = np.array(c_info)
        cam_all_info = (np.sum(c_info,axis=0)/len(c_info))
        cam_all_info = F.relu(torch.tensor(cam_all_info))
        cam_all_info = cam_all_info.detach().numpy()
        cam_all_info -= cam_all_info.min()
        cam_all_info /= cam_all_info.max()
        atoms_info.append((cams, cam_all_info))
        data_save = pd.concat([data_save, pd.DataFrame([[data_info.loc[i, "SMILES"], data_info.loc[i, "TARGET"], cams[0], cams[1], cams[2], cams[3], cams[4], cam_all_info]], columns=columns)])
        # pred = model(**net_input)
        # print(pred, net_target)
        # print_layer_outputs(model, net_input)
        # "encoder.layers.7.self_attn_layer_norm", "encoder.layers.7.final_layer_norm"
    data_save.to_csv(save_file, index=False)