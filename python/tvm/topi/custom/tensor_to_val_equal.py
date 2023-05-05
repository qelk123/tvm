from typing import Callable, Optional

from numpy import int8

import tvm

import typing

from tvm import te
from tvm import topi




#(matrix, cur_layer, visited)
def tensor_to_val_equal(
    matrix: tvm.te.Tensor,
    val: tvm.te.var,
):
    return te.compute((), lambda: matrix(0)==val)