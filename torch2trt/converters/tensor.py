from torch2trt.torch2trt import *
import numpy as np


@tensorrt_converter('torch.tensor')
# @tensorrt_converter('torch.Tensor')
@tensorrt_converter('torch.from_numpy')
def convert_tensor(ctx):
    input = ctx.method_args[0]
    input = torch.tensor(input,**ctx.method_kwargs)
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    output._trt = input_trt


@tensorrt_converter('torch.Tensor.cuda')
def convert_cuda(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    output._trt = input_trt
