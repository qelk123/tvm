"""Neural network operations."""
from tvm.relay import expr

from ...expr import Constant, Expr, const,TupleWrapper
from ..dyn.nn import _make as _dyn_make
from . import _make
# from .utils import get_pad_tuple1d, get_pad_tuple2d, get_pad_tuple3d


def custombfs(matrix, cur_layer, visited, layer):
    return _make.custombfs(matrix, cur_layer, visited, layer)
  
def updatepi(cur_layer, visited, layer):
    return _make.updatepi(cur_layer, visited, layer)

def tensor_to_val_equal(matrix, val):
    return _make.tensor_to_val_equal(matrix, val)

def matrix_to_val(matrix):
    return _make.matrix_to_val(matrix)

def dynamic_output_tensor(size):
    return _make.dynamic_output_tensor(size)

def same_output_tensor(size):
    return _make.same_output_tensor(size)

def tensor_ana(S_M):
    return _make.tensor_ana(S_M)

def split_tensor(S_M,split_info):
    return TupleWrapper(_make.split_tensor(S_M,split_info),2)

def spmv_cpu(S_M,V):
    return _make.spmv_cpu(S_M,V)

def spmv_gpu(S_M,V):
    return _make.spmv_gpu(S_M,V)

def concat_vector(split_info,V1,V2):
    return _make.concat_vector(split_info,V1,V2)

def bfs_pprocess(V,Pi,i):
    return _make.bfs_pprocess(V,Pi,i)