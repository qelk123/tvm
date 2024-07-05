import tvm
import tvm.testing
from tvm.script import tir as T
import tvm.sparse
from utils import ell, ell2, csr2ell_index_map, csr2ell_inv_index_map
import numpy as np

@T.prim_func
def easier_kernel(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    M = T.dense_fixed(m)
    K = T.sparse_variable(M, (n, nnz), (indptr, indices), "int32")
    K_detach = T.dense_fixed(n)
    A = T.match_sparse_buffer(a, (M, K), "float32")
    B = T.match_sparse_buffer(b, (K_detach,), "float32")
    C = T.match_sparse_buffer(c, (M,), "float32")
    with T.sp_iter([M, K], "SR", "csrmm") as [i, k]:
        with T.init():
            C[i] = 0.0
        C[i] = C[i] + A[i, k] * B[k]


def test_easier_reduce(*args, **kwargs):
  
    bucket_sizes = [4]
    rewrites = []
    for bucket_id, bucket_size in enumerate(bucket_sizes):
        rewrites.append(
            tvm.sparse.FormatRewriteRule(
                str(bucket_id),
                ell2.specialize({ell2.params[-1]: bucket_size}),
                ["A"],
                ["M", "K"],
                ["O", "I", "J"],
                {"M": ["O", "I"], "K": ["J"]},
                csr2ell_index_map,
                csr2ell_inv_index_map,
            )
        )
  
  
  
    mod = tvm.IRModule.from_expr(easier_kernel)
    mod = tvm.sparse.format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)
    
    mod = tvm.sparse.lower_sparse_iter(mod)
    print(mod)
    mod = tvm.sparse.lower_sparse_buffer(mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    sch = tvm.tir.Schedule(mod)
    block_b = sch.get_block("csrmm_01")
    [row_iter, col_iter] = sch.get_loops(block_b)
    row_o, _, row_i = sch.split(row_iter, [1, None, 128])
    sch.bind(row_o,"blockIdx.x")
    sch.bind(row_i,"threadIdx.x")
    print("before:", sch.mod)
    print("after:", sch.mod)
    store = sch.cache_write(block_b, 0, "local")
    sch.reverse_compute_at(store, row_i)
    sch.annotate(store, "atomic", True)

    print(sch.mod)

    f = tvm.build(sch.mod, target="cuda")
    print(f.imported_modules[0].get_source("cu"))

    # prepare input
    m = 128
    n = 128
    non_zero_m = 1024
    input_x_nd = tvm.nd.array(np.random.randn((n)).astype("float32"), tvm.cuda())
    output_y_nd = tvm.nd.array(np.zeros((m)).astype("float32"), tvm.cuda())
    input_A_nd = tvm.nd.array(np.random.randn((non_zero_m * 4)).astype("float32"), tvm.cuda())
    indptr_i_nd = tvm.nd.array(np.array([0, non_zero_m], dtype="int32"), tvm.cuda())
    indices_i_nd = tvm.nd.array(np.random.randint(0, m, non_zero_m).astype("int32"), tvm.cuda())
    indices_j_nd = tvm.nd.array(np.random.randint(0, n, non_zero_m * 4).astype("int32"), tvm.cuda())
    print(output_y_nd.asnumpy())
    f(input_x_nd, output_y_nd, m, n, input_A_nd, indptr_i_nd, indices_i_nd, indices_j_nd, non_zero_m, non_zero_m)
    print(output_y_nd.asnumpy())
    output_ref = np.zeros((m)).astype("float32")
    indices_i_np = indices_i_nd.asnumpy()
    indices_j_np = indices_j_nd.asnumpy()
    input_x_np = input_x_nd.asnumpy()
    input_A_np = input_A_nd.asnumpy()


    for i in range(non_zero_m):
        for j in range(4):
            output_ref[indices_i_np[i]] += input_A_np[i * 4 + j] * input_x_np[indices_j_np[i * 4 + j]]
    
    print(output_ref)
    tvm.testing.assert_allclose(output_y_nd.asnumpy(), output_ref, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_easier_reduce()
