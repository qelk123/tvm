from typing import Callable, Optional

from numpy import int8

import tvm

import typing

from tvm import te
from tvm import topi


#(matrix, cur_layer, visited)
def updatepi(
    W: tvm.te.Tensor,
    Pi: tvm.te.Tensor,
    layer
):
  return te.compute(W.shape,lambda x:te.if_then_else(te.all(W[x] > 0 , Pi[x] == -1),layer,Pi[x]))