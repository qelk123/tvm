from typing import Callable, Optional

from numpy import int8

import tvm

import typing

from tvm import te
from tvm import topi




#(matrix, cur_layer, visited)
def same_output_tensor(
    size: tvm.te.Tensor
):
    # s = tvm.te.var("num_elem")
    return te.compute(size.shape, lambda i: size[i])