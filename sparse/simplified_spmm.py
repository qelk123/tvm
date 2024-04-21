import tvm
import tvm.testing
from tvm.script import tir as T
import tvm.sparse


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
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




def bench_hyb(*args, **kwargs):
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.sparse.lower_sparse_iter(mod)
    mod = tvm.sparse.lower_sparse_buffer(mod)
    print(mod)
    f = tvm.build(mod, target="llvm")
    
    import numpy as np
    nnz = 8
    # data_nd = tvm.nd.array(np.ones((nnz)).astype("float32"))
    data_nd = tvm.nd.array(np.random.randn((nnz)).astype("float32"))
    n = 8
    m = 3
    num_tiles = 2
    cwm = 2
    index_nd = tvm.nd.array(np.array([1, 2, 4, 2, 5, 1, 3, 4], dtype="int32"))
    indptr_nd = tvm.nd.array(np.array([0, 3, 5, 8], dtype="int32"))
    b_nd = tvm.nd.array(np.random.randn((n * num_tiles * cwm * 32)).astype("float32"))
    c_nd = tvm.nd.array(np.zeros((m * num_tiles * cwm * 32)).astype("float32"))
    f(data_nd, b_nd, c_nd, indptr_nd, index_nd, m, n, num_tiles, nnz, cwm)
    c_np: np.ndarray = c_nd.asnumpy()
    c_np = c_np.reshape((m, num_tiles, cwm, 32))
    
    c_np_ref = np.zeros_like(c_np)
    b_np_ref = b_nd.asnumpy().reshape((n, num_tiles, cwm, 32))
    index_np = index_nd.asnumpy()
    indptr_np = indptr_nd.asnumpy()
    data_np = data_nd.asnumpy()
    for i in range(m):
        for j in range(indptr_np[i], indptr_np[i + 1]):
            for k1 in range(num_tiles):
                for k2 in range(cwm):
                    for k3 in range(32):
                        c_np_ref[i, k1, k2, k3] += data_np[j] * b_np_ref[index_np[j], k1, k2, k3]

    tvm.testing.assert_allclose(c_np, c_np_ref, rtol=1e-5, atol=1e-5)



if __name__ == "__main__":
    bench_hyb()
