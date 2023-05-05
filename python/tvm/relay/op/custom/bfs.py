"""Backend compiler related feature registration"""
from __future__ import absolute_import
import re

from tvm import relay, topi
from tvm.runtime import convert
from tvm.te.hybrid import script
from tvm.script import tir as T
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple

from ....ir import container
from ....tir import expr
from ...transform import LayoutConfig
from .. import op as reg

from .. import strategy
from .._tensor import elemwise_shape_func
from ..op import OpPattern
from ..strategy.generic import is_depthwise_conv2d

# reg.register_injective_schedule("liuyn.")
# reg.register_pattern("liuyn.liuynadd", OpPattern.BROADCAST)

reg.register_strategy(
    "custom.custombfs",
    strategy.bfs_strategy,
)
reg.register_strategy(
    "custom.updatepi",
    strategy.updatepi_strategy,
)
reg.register_strategy(
    "custom.tensor_to_val_equal",
    strategy.tensor_to_val_equal_strategy,
)
reg.register_strategy(
    "custom.matrix_to_val",
    strategy.matrix_to_val_strategy,
)
reg.register_strategy(
    "custom.dynamic_output_tensor",
    strategy.dynamic_output_tensor_strategy,
)

reg.register_strategy(
    "custom.same_output_tensor",
    strategy.same_output_tensor_strategy,
)

@script
def _dynamic_output_tensor_shape_func(size):
    out = output_tensor((1,), "int64")
    out[0] = int64(size[()])
    return out

# @T.prim_func
# def _dynamic_output_tensor_shape_func(size:T.handle,ret:T.handle):
#     R=T.match_buffer(ret,(1,),"int64")
#     S=T.match_buffer(size,(1,),"int64")
#     R[0]=S[0]
    


@reg.register_shape_func("custom.dynamic_output_tensor", True)
def dynamic_output_tensor_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return [_dynamic_output_tensor_shape_func(*inputs)]



@script
def _split_tensor_shape_func(g,s_info):
    out = output_tensor((2,), "int64")
    out1 = output_tensor((2,), "int64")
    out[1] = int64(g.shape[1])
    out1[1] = int64(g.shape[1])
    dim1 = int64(0)
    dim2 = int64(0)
    for index in range(g.shape[0]):
        dim1=dim1+int64(s_info[0,index])
        dim2=dim2+int64(s_info[1,index])
    out[0] = dim1
    out1[0] = dim2
    return (out,out1)

@reg.register_shape_func("custom.split_tensor", True)
def split_tensor_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return _split_tensor_shape_func(*inputs)




@script
def _spmv_cpu_shape_func(g,w_in):
    out = output_tensor((1,), "int64")
    out[0] = int64(g.shape[0])
    return out

@reg.register_shape_func("custom.spmv_cpu", True)
def spmv_cpu_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return [_spmv_cpu_shape_func(*inputs)]

@reg.register_shape_func("custom.spmv_gpu", True)
def spmv_gpu_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return [_spmv_cpu_shape_func(*inputs)]


@script
def _tensor_ana_shape_func(g):
    out = output_tensor((2,), "int64")
    out[0] = int64(2)
    out[1] = int64(g.shape[0])
    return out


@reg.register_shape_func("custom.tensor_ana", True)
def tensor_ana_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return [_tensor_ana_shape_func(*inputs)]



@script
def _concat_vector_shape_func(s_info,w1,w2):
    out = output_tensor((1,), "int64")
    out[0] = int64(s_info.shape[1])
    return out


@reg.register_shape_func("custom.concat_vector", True)
def concat_vector_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return [_concat_vector_shape_func(*inputs)]


@script
def _bfs_pprocess_shape_func(W,Pi,i_tensor):
    out = output_tensor((1,), "int64")
    out[0] = int64(Pi.shape[0])
    return out


@reg.register_shape_func("custom.bfs_pprocess", True)
def bfs_pprocess_shape_func(attrs, inputs, _):
    """
    Shape func for custom.dynamic_output_tensor
    """
    return [_bfs_pprocess_shape_func(*inputs)]

# @reg.register_shape_func("custom.dynamic_output_tensor", True)
# def dynamic_output_tensor_shape_func(attrs, inputs, _):
#     """
#     Shape func for custom.dynamic_output_tensor
#     """
#     return [_dynamic_output_tensor_shape_func(*inputs)]
