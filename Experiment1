import time
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pprintpp import pprint as pp
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def count_flops_params(model, x):
    with torch.no_grad():
        parameters = parameter_count_table(model)
        flops = FlopCountAnalysis(model, x)
    #print('total flops and parameters:\n')
    #print(parameters)
    print(flops.total())
    # pp(flops.by_module())
    return None


class Multi_branch(nn.Module):
    def __init__(self, in_channel, out_channel, branch, channel_equal=False):
        super(Multi_branch, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.branch = branch
        self.channel_equal = channel_equal

        if channel_equal:
            self.sub_channel = self.out_channel // self.branch
        else:
            self.sub_channel = out_channel

        self.modulelist = nn.ModuleList()

        self.modulelist.extend(
            [nn.Conv2d(self.in_channel, self.sub_channel, kernel_size=(3, 3), stride=(1, 1), padding=1) for i in range(self.branch)])

    def forward(self, x):
        if self.channel_equal:
            out = []
            for i in range(self.branch):
                out.append(self.modulelist[i](x))
            out = torch.cat(out, dim=1)
        else:
            for i in range(self.branch):
                if i == 0:
                    out = self.modulelist[i](x)
                else:
                    out += self.modulelist[i](x)
        return out


class dnn(nn.Module):
    def __init__(self, in_channel, out_channel, block_num, branch=1, channel_equal=False, with_shuffle=False):
        super(dnn, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block_num = block_num
        self.branch = branch
        self.channel_equal = channel_equal
        self.with_shuffle = with_shuffle

        self.model_sequential = nn.Sequential()

        for i in range(block_num):
            self.model_sequential.add_module('stage{}'.format(i),
                                             Multi_branch(self.in_channel,
                                                          self.out_channel,
                                                          self.branch,
                                                          self.channel_equal))

    def forward(self, x):
        out = self.model_sequential(x)
        return out


def test(branch, C, M, block_num, data_batch, channel_equal, DEVICE='cpu', warm_up_num=100, cyc=10):
    model = dnn(C, C, block_num, branch=branch, channel_equal=channel_equal)

    data = torch.randn(data_batch, C, M, M).float()
    warm_up = torch.randn(data_batch, C, M, M).float()
    randn = torch.randn(1, C, M, M).float()

    if DEVICE == 'gpu':
        model.to('cuda:0')
        data = data.cuda()
        warm_up = warm_up.cuda()
        randn = randn.cuda()

    model.eval()
    count_flops_params(model, randn)
    # warm up
    for _ in tqdm(range(warm_up_num)):
        _ = model(warm_up)

    t = 0
    for _ in range(cyc):
        t1 = time.time()
        for _ in tqdm(range(1280)):
            _ = model(data)
        t2 = time.time() - t1
        t += t2

    t = round(1280 / (t / cyc), 3)

    print('每秒计算{}个batch'.format(t))

# 单元计算量对齐
# gpu
# test(branch=1, C=32, M=32, block_num=24, data_batch=64, channel_equal=False, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(branch=2, C=32, M=32, block_num=12, data_batch=64, channel_equal=False, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(branch=4, C=32, M=32, block_num=6, data_batch=64, channel_equal=False, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(branch=8, C=32, M=32, block_num=3, data_batch=64, channel_equal=False, DEVICE='gpu', warm_up_num=100, cyc=10) 
# cpu
# test(branch=1, C=32, M=32, block_num=24, data_batch=64, channel_equal=False, DEVICE='cpu', warm_up_num=100, cyc=1) 
# test(branch=2, C=32, M=32, block_num=12, data_batch=64, channel_equal=False, DEVICE='cpu', warm_up_num=100, cyc=1) 
# test(branch=4, C=32, M=32, block_num=6, data_batch=64, channel_equal=False, DEVICE='cpu', warm_up_num=100, cyc=1)
# test(branch=8, C=32, M=32, block_num=3, data_batch=64, channel_equal=False, DEVICE='cpu', warm_up_num=100, cyc=1)

# 模块计算量对齐
# gpu
# test(branch=1, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(branch=2, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(branch=4, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(branch=8, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='gpu', warm_up_num=100, cyc=10) 
# cpu
# test(branch=1, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='cpu', warm_up_num=100, cyc=1) 
# test(branch=2, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='cpu', warm_up_num=100, cyc=1) 
# test(branch=4, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='cpu', warm_up_num=100, cyc=1) 
# test(branch=8, C=32, M=32, block_num=24, data_batch=64, channel_equal=True, DEVICE='cpu', warm_up_num=100, cyc=1) 
