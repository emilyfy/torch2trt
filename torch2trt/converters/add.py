from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__radd__')
def convert_radd(ctx):
    input_a = ctx.method_args[1]  # flipped for radd
    input_b = ctx.method_args[0]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)
    

class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_basic():
    return Add()


class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_iadd():
    return IAdd()


class TorchAdd(torch.nn.Module):
    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torchadd():
    return TorchAdd()


class RAddInt(torch.nn.Module):
    def __init__(self):
        super(RAddInt, self).__init__()

    def forward(self, x):
        return 1 + x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_int():
    return RAddInt()


class RAddFloat(torch.nn.Module):
    def __init__(self):
        super(RAddFloat, self).__init__()

    def forward(self, x):
        return 1.0 + x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_float():
    return RAddFloat()