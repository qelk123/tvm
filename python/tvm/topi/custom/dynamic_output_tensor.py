from typing import Callable, Optional

from numpy import int8

import tvm

import typing

from tvm import te
from tvm import topi




#(matrix, cur_layer, visited)
def dynamic_output_tensor(
    size: tvm.te.var
):
    s = tvm.te.var("num_elem")
    return te.compute((s,), lambda i: i)