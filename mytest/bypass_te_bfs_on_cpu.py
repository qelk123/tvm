import os
# print(os.getpid())
import tvm
from tvm import relay
from tvm.script import tir as T
from tvm.relay import vm
import numpy as np
from tvm import runtime
from tvm.contrib import utils
import relay_multi_target

# export lib test
compile_needed = True


if compile_needed:
  @tvm._ffi.register_func("relay.backend.spmv_cpu")
  def spmv_cpu():
      @T.prim_func
      def spmv_cpu(g: T.handle,w_in: T.handle,w_out: T.handle) -> None:
        col_all=T.var("int32")
        row_all=T.var("int32")
        G=T.match_buffer(g,(row_all,col_all),"int32")
        WI=T.match_buffer(w_in,(col_all,),"int32")
        WO=T.match_buffer(w_out,(row_all,),"int32")

        for i in range(row_all):
          with T.block("b_1"):
            vi = T.axis.remap("S", [i])
            T.reads(G[vi, 0:col_all],WI[0:col_all])
            T.writes(WO[vi])
            with T.init():
              WO[vi]= 0
            for j in range(col_all):
              WO[vi]= WO[vi]+ WI[j]* G[vi,j]
      return spmv_cpu

  @tvm._ffi.register_func("relay.backend.bfs_pprocess")
  def bfs_pprocess():
      @T.prim_func
      def bfs_pprocess(w: T.handle,pi: T.handle,i: T.handle,pi_n: T.handle) -> None:
        size=T.var("int32")
        W=T.match_buffer(w,(size,),"int32")
        PI=T.match_buffer(pi,(size,),"int32")
        I=T.match_buffer(i,(1,),"int32")
        PIN=T.match_buffer(pi_n,(size,),"int32")

        for i in range(size):
          with T.block("b_1"):
            vi = T.axis.remap("S", [i])
            if W[vi]>0 and PI[vi]==-1:
              PIN[vi]=I[0]
            else:
              PIN[vi]=PI[vi]
              W[vi]=0
      return bfs_pprocess

  G = relay.var("G", shape=(relay.Any(), relay.Any()), dtype="int32")
  W = relay.var("W", shape=(relay.Any(),), dtype="int32")
  Pi = relay.var("Pi", shape=(relay.Any(),), dtype="int32")

  step_n = relay.var("step_n", shape=(), dtype="int32")
  i = relay.var("i", shape=(), dtype="int32")
  i_tensor = relay.var("i_tensor", shape=(1,), dtype="int32")

  def body(Pi, W,i_tensor, i):
      G_C= relay.annotation.on_device(G,tvm.cpu(),constrain_body=False,constrain_result=True)
      W_c = relay.annotation.on_device(relay.custom.spmv_cpu(G,W),tvm.cpu())
      Pi_n = relay.annotation.on_device(relay.custom.bfs_pprocess(W_c,Pi,i_tensor),tvm.cpu())
      j = relay.add(i, relay.const(1, "int32"))
      i_tensor_n = relay.add(i_tensor, relay.const(1, "int32"))
      return Pi_n,W_c,i_tensor_n, j
  def cond(Pi, W,i_tensor, i):
      return relay.less(i,step_n)

  myloop = relay.loops.while_loop(cond, [Pi, W,i_tensor, i], body)
  z = myloop(Pi,W,i_tensor, relay.const(0, "int32"))
  result = relay.Function([G,W,Pi,i_tensor,step_n], relay.TupleGetItem(z, 0))




  module = tvm.IRModule.from_expr(result)
  
  # print(module)
  with tvm.transform.PassContext(opt_level=3, config={"relay.fallback_device_type": tvm.cpu().device_type,"relay.backend.use_tvm_script": True}):
      exe = relay.vm.compile(
        module, target={"cpu": tvm.target.Target("llvm"), "cuda": tvm.target.Target("cuda")}
      )
      code, lib = exe.save()
      with open("bypass_te_cpu.ll", "w") as external_file:
        print(lib.get_source(),file=external_file)
        
  # serialize.
  assert isinstance(code, bytearray)

  # save and load the code and lib file.
  tmp = os.getcwd()
  path_lib = tmp+"/lib.so"
  lib.export_library(path_lib)
  with open(tmp+"/code.ro", "wb") as fo:
      fo.write(code)

# load module
loaded_lib = tvm.runtime.load_module(os.getcwd()+"/lib.so")
loaded_code = bytearray(open(os.getcwd()+"/code.ro", "rb").read())

# deserialize.
exe2 = runtime.vm.Executable.load_exec(loaded_code, loaded_lib)


# prepare input

dim_size = 5
step = 5

inG = np.random.randint(2, size=[dim_size,dim_size]).astype("int32")
n_inG = inG.transpose()
inW = np.zeros([dim_size,]).astype("int32")
i_t = np.zeros([1,]).astype("int32")
inW[0]=1
inPi = np.ones([dim_size,]).astype("int32")
inPi = inPi*-1
inPi[0]=0


vm_my = runtime.vm.VirtualMachine(exe2, [tvm.cpu(),tvm.cuda()])  
vm_my.set_input("main", **{"G":n_inG,"W":inW,"Pi":inPi,"i_tensor":i_t,"step_n":step})
tvm_res = vm_my.run()
# re = vm.invoke("main",a,np.ones((5), dtype='int32'),np.ones((5), dtype='int32'),np.ones((1), dtype='int32'),np.zeros((5), dtype='int32'))
print("input matrix:\n",inG)
print("input vector:\n",inW)
print("result:\n",tvm_res)
