from typing import Callable, Optional

from numpy import int8

import tvm

import typing

from tvm import te
from tvm import topi

def spmv(G:te.Tensor,W:te.Tensor,Pi:te.Tensor):

  k=te.reduce_axis((0,W.shape[0]),"k")
  # W2=te.compute(W.shape,lambda x:te.sum(te.if_then_else(x > 0,te.if_then_else(x<W.shape[0],G[k,x-1]*W[k]+G[k,x]*W[k]+G[k,x+1]*W[k],G[k,x]*W[k]+G[k,x+1]*W[k]),
  #                                                           G[k,x]*W[k]+G[k,x+1]*W[k]),axis=k))
  W2=te.compute(W.shape,lambda x:te.sum(G[x,k]*W[k],axis=k),name="new_W")
  
  # Pi2=te.compute(W.shape,lambda x:te.if_then_else(te.all(W[x] > 0 , Pi[x] == -1),depth,Pi[x]))
  W3=te.compute(W.shape,lambda x:te.if_then_else(te.all(W2[x] > 0 , Pi[x] == -1),W2[x],0))
  return W3


#(matrix, cur_layer, visited)
def bfs(
    G: tvm.te.Tensor,
    W: tvm.te.Tensor,
    Pi: tvm.te.Tensor,
    layer: tvm.te.var
):
    return spmv(G,W,Pi)