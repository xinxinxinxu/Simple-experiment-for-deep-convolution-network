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


class dnn(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num, with_shuffle, shuffle_group = 4):
        super(dnn, self).__init__()
    
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer_num = layer_num
        self.with_shuffle = with_shuffle
        self.shuffle_group = shuffle_group

        self.net = nn.ModuleList()

        for i in range(layer_num):
            self.net.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), padding=(1,1)))
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batchsize, groups, 
            channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    

    def forward(self, x):
        out = None
        for i in range(self.layer_num):
            if out != None:
                out = self.net[i](out)
            else:
                out = self.net[i](x)
            
            if self.with_shuffle:
                out = self.channel_shuffle(out, self.shuffle_group)
        
        return out


def test(C, M, layer_num, with_shuffle ,data_batch, DEVICE='cpu', warm_up_num=100, cyc=10):
    model = dnn(C, C, layer_num, with_shuffle, shuffle_group = 4)

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


# with shuffle
# gpu
# test(C=32, M=32, layer_num=5, with_shuffle=True ,data_batch=64, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=20, with_shuffle=True ,data_batch=64, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=100, with_shuffle=True ,data_batch=64, DEVICE='gpu', warm_up_num=100, cyc=10) 
# cpu
# test(C=32, M=32, layer_num=5, with_shuffle=True ,data_batch=64, DEVICE='cpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=20, with_shuffle=True ,data_batch=64, DEVICE='cpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=100, with_shuffle=True ,data_batch=64, DEVICE='cpu', warm_up_num=100, cyc=10) 

# without shuffle
# gpu
# test(C=32, M=32, layer_num=5, with_shuffle=False ,data_batch=64, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=20, with_shuffle=False ,data_batch=64, DEVICE='gpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=100, with_shuffle=False ,data_batch=64, DEVICE='gpu', warm_up_num=100, cyc=10) 
# cpu
# test(C=32, M=32, layer_num=5, with_shuffle=False ,data_batch=64, DEVICE='cpu', warm_up_num=100, cyc=10) 
test(C=32, M=32, layer_num=20, with_shuffle=False ,data_batch=64, DEVICE='cpu', warm_up_num=100, cyc=10) 
# test(C=32, M=32, layer_num=100, with_shuffle=False ,data_batch=64, DEVICE='cpu', warm_up_num=100, cyc=10) 
