import torch
from string import Template
from collections import namedtuple
import cupy

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy.memoize()
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    # kernel_code = cupy.cuda.compile_with_cache(code)
    kernel_code = cupy.RawModule(code=code, backend='nvrtc')
    return kernel_code.get_function(kernel_name)