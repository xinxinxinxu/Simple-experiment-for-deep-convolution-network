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
    # print('total flops and parameters:\n')
    # print(parameters)
    print(flops.total())
    # pp(flops.by_module())
    return None


class dnn(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, layer_num=10):
        super(dnn, self).__init__()

        self.padding = kernel_size // 2

        self.nn = nn.Sequential()

        for i in range(layer_num):
            self.nn.add_module('stage{}'.format(i),
                               nn.Conv2d(in_channel,
                                         out_channel,
                                         kernel_size=kernel_size,
                                         stride=(1, 1),
                                         padding=self.padding))
    
    def forward(self, x):
        return self.nn(x)

def test(K, C, M, layer_num, data_batch,DEVICE='gpu', warm_up_num = 100, cyc = 10):
    model = dnn(K, C, C, layer_num)
        
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

    t = round(1280/(t/cyc), 3)
    
    print('每秒计算{}个batch'.format(t))

# 通道数对齐
# gpu
# test(K=1, C=72, M=32, layer_num=10, data_batch=64, DEVICE='gpu') 
# test(K=3, C=24, M=32, layer_num=10, data_batch=64, DEVICE='gpu')
# test(K=9, C=8, M=32, layer_num=10, data_batch=64, DEVICE='gpu')
# cpu
# test(K=1, C=72, M=32, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)
# test(K=3, C=24, M=32, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)
# test(K=9, C=8, M=32, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)


# 分辨率对齐
# gpu
# test(K=1, C=8, M=288, layer_num=10, data_batch=64, DEVICE='gpu') 
# test(K=3, C=8, M=96, layer_num=10, data_batch=64, DEVICE='gpu')
# test(K=9, C=8, M=32, layer_num=10, data_batch=64, DEVICE='gpu')
# cpu
# test(K=1, C=8, M=288, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)
# test(K=3, C=8, M=96, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)
# test(K=9, C=8, M=32, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)

# 深度对齐
# gpu
# test(K=1, C=8, M=32, layer_num=810, data_batch=64, DEVICE='gpu') 
# test(K=3, C=8, M=32, layer_num=90, data_batch=64, DEVICE='gpu')
# test(K=9, C=8, M=32, layer_num=10, data_batch=64, DEVICE='gpu')
# cpu
# test(K=1, C=8, M=32, layer_num=810, data_batch=64, DEVICE='cpu', cyc=1)
# test(K=3, C=8, M=32, layer_num=90, data_batch=64, DEVICE='cpu', cyc=1)
test(K=9, C=8, M=32, layer_num=10, data_batch=64, DEVICE='cpu', cyc=1)
