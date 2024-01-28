import os
import shutil
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

def get_lap(input, poolsize, device):
    avg_pool_layer = nn.AvgPool2d(kernel_size=poolsize, stride=poolsize)
    lap_X = avg_pool_layer(input)
    lap_filter = torch.tensor([
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        ]).float().to(device)
    lap_X = F.conv2d(lap_X, lap_filter.unsqueeze(0))
    #print("lap_X:", lap_X.shape)
    return lap_X

def move_data():
    # 源文件夹路径
    src_folder = "./style-data/train"
    # 目标文件夹路径
    dest_folder = "./myTrain/style"

    # 获取源文件夹中所有的文件名
    file_names = os.listdir(src_folder)

    # 随机选择要移动的文件名
    n = 10000  # 要移动的文件数量
    selected_file_names = random.sample(file_names, n)

    # 遍历所有选择的文件名
    for file_name in selected_file_names:
        # 拼接源文件路径
        src_file_path = os.path.join(src_folder, file_name)
        # 拼接目标文件路径
        dest_file_path = os.path.join(dest_folder, file_name)
        # 移动文件
        shutil.move(src_file_path, dest_file_path)

    print("移动完成！")
    
def transform_tar2pth():
    filename = './experiments_original/decoder_iter_160000.pth.tar'
    # filename = './models/decoder_iter_100.pth'

    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        # print(checkpoint.keys())
        state_dict = {}
        for key in checkpoint.keys():
            if not key.startswith('optimizer') and not key.startswith('scheduler'):
                new_key = key.replace('module.', '')  # 处理多GPU模型参数的前缀'module.'
                state_dict[new_key] = checkpoint[key]
            
        torch.save(state_dict, './models/decoder_original_16.pth')
    else:
        print(f"File {filename} not found.")
        