from typing import Callable, Optional

from numpy import int8

import tvm

import typing

from tvm import te
from tvm import topi




#(matrix, cur_layer, visited)
def matrix_to_val(
    matrix: tvm.te.Tensor
):
    return te.compute((), lambda: matrix(0)+10)