import os
print(os.getpid())
import tvm
import tvm.testing
from tvm.script import tir as T
import tvm.sparse


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    d: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    cwm: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(cwm)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    
    with T.sp_iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]
    # DD = T.dense_fixed(m)
    # J = T.sparse_variable(DD, (m, nnz), (indptr, indices), "int32")
    # D = T.match_sparse_buffer(d, (DD, J), "float32")
    # A = T.match_buffer(a, (m, m), "float32")
    # B = T.match_buffer(b, (m, m), "float32")
    # C = T.match_buffer(c, (m, m), "float32")
    # for i0, i1, i2 in T.grid(m, m, m):
    #     with T.block("csrmm"):
    #         vi, vj, vk = T.axis.remap("SSR", [i0, i1, i2])
            
    #         with T.init():
    #             C[vi, vk] = 0.0
    #         C[vi, vk] = C[vi, vk] + B[vi, vj] * B[vj, vk]



@T.prim_func
def tir_func(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, num_tiles: T.int32, nnz: T.int32, cwm: T.int32):
    T.func_attr({"sparse_tir_level": 0, "tir.noalias": T.bool(True)})
    _data = T.match_buffer(a, (nnz,), strides=(1,))
    _data_1 = T.match_buffer(b, (n * num_tiles * cwm * 32,), strides=(1,))
    _data_2 = T.match_buffer(c, (m * num_tiles * cwm * 32,), strides=(1,))
    J_indptr_data = T.match_buffer(indptr, (m + 1,), "int32", strides=(1,))
    J_indices_data = T.match_buffer(indices, (nnz,), "int32", strides=(1,))
    # with T.block("root"):
    for _i in range(m):
        with T.block("csrmm0"):
            vi = T.axis.spatial(m, _i)
            # j = T.int32()
            # i = T.int32()
            # k2 = T.int32()
            # k1 = T.int32()
            # k3 = T.int32()
            # T.reads(J_indptr_data[0:m + 1], _data[_j + J_indptr_data[_i]], _data_1[_k2 * 32 + J_indices_data[j + J_indptr_data[i]] * num_tiles * cwm * 32 + k1 * cwm * 32 + k3], J_indices_data[j + J_indptr_data[i]])
            # T.writes(_data_2[k2 * 32 + i * num_tiles * cwm * 32 + k1 * cwm * 32 + k3])
            T.block_attr({"sparse": T.bool(True)})
            for _j, _k1, _k2, _k3 in T.grid(J_indptr_data[vi + 1] - J_indptr_data[vi], num_tiles, cwm, 32):
                with T.block("csrmm1"):
                    vj = T.axis.reduce(n, _j)
                    vk1, vk2, vk3 = T.axis.remap("SSS", [_k1, _k2, _k3])
                    T.reads(_data[vj + J_indptr_data[vi]], _data_1[vk2 * 32 + J_indices_data[vj + J_indptr_data[vi]] * num_tiles * cwm * 32 + vk1 * cwm * 32 + vk3], J_indices_data[vj + J_indptr_data[vi]])
                    T.writes(_data_2[vk2 * 32 + vi * num_tiles * cwm * 32 + vk1 * cwm * 32 + vk3])
                    T.block_attr({"sparse": T.bool(True)})
                    with T.init():
                        _data_2[vk2 * 32 + vi * num_tiles * cwm * 32 + vk1 * cwm * 32 + vk3] = T.float32(0)
                    _data_2[vk2 * 32 + vi * num_tiles * cwm * 32 + vk1 * cwm * 32 + vk3] = _data_2[vk2 * 32 + vi * num_tiles * cwm * 32 + vk1 * cwm * 32 + vk3] + _data[vj + J_indptr_data[vi]] * _data_1[vk2 * 32 + J_indices_data[vj + J_indptr_data[vi]] * num_tiles * cwm * 32 + vk1 * cwm * 32 + vk3]


def bench_hyb(*args, **kwargs):
    # mod = tvm.IRModule.from_expr(tir_func)
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.sparse.lower_sparse_iter(mod)
    mod = tvm.sparse.lower_sparse_buffer(mod)
    # print(mod)
    
    f = tvm.build(mod, target="llvm")



if __name__ == "__main__":
    bench_hyb()
