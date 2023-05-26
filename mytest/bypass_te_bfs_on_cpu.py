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
compile_needed = True
dim_size = 5
step = 5
if compile_needed:
  # @tvm._ffi.register_func("relay.backend.tensor_ana")
  # def tensor_ana():
  #     @T.prim_func
  #     def tensor_ana(g: T.handle,s_info: T.handle) -> None:
  #       m=T.var("int32")
  #       n=T.var("int32")
  #       G=T.match_buffer(g,(m,n),"int32")
  #       S=T.match_buffer(s_info,(2,m),"int32")
  #       for i,j in T.grid(2,m):
  #         with T.block("b_1"):
  #           vi, vj = T.axis.remap("SS", [i, j])
  #           T.writes(S[vi, vj])
  #           if vi<1:
  #             S[vi, vj]=vj%2
  #           else:
  #             S[vi, vj]=(vj+1)%2
  #     return tensor_ana
              
      
  # @tvm._ffi.register_func("relay.backend.split_tensor")
  # def split_tensor():
  #     @T.prim_func
  #     def split_tensor(g: T.handle,s_info: T.handle,cpu_p: T.handle,gpu_p: T.handle) -> None:
  #       col_all=T.var("int32")
  #       row_all=T.var("int32")
  #       row_cpu=T.var("int32")
  #       row_gpu=T.var("int32")
  #       G=T.match_buffer(g,(row_all,col_all),"int32")
  #       CP=T.match_buffer(cpu_p,(row_cpu,col_all),"int32")
  #       GP=T.match_buffer(gpu_p,(row_gpu,col_all),"int32")
  #       S=T.match_buffer(s_info,(2,row_all),"int32")
  #       CUR_R=T.alloc_buffer((2,),"int32")
  #       for o in range(2):
  #         CUR_R[o]= 0
          
  #       for i in range(row_all):
  #         with T.block("b_1"):
  #           vi = T.axis.remap("R", [i])
  #           T.reads(G[vi, 0:col_all],S[0,vi])
  #           T.writes(CP[0:row_cpu, 0:col_all],GP[0:row_gpu, 0:col_all])
  #           if S[0,vi]< 1:
  #             for j in range(col_all):
  #               GP[CUR_R[1],j]= G[vi,j]
  #             CUR_R[1]= CUR_R[1]+ 1
  #           else:
  #             for j in range(col_all):
  #               CP[CUR_R[0],j]= G[vi,j]
  #             CUR_R[0]= CUR_R[0]+ 1
  #     return split_tensor

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
    
  # @tvm._ffi.register_func("relay.backend.spmv_gpu")
  # def spmv_gpu():
  #     @T.prim_func
  #     def spmv_gpu(g: T.handle,w_in: T.handle,w_out: T.handle) -> None:
  #       col_all=T.var("int32")
  #       row_all=T.var("int32")
  #       G=T.match_buffer(g,(row_all,col_all),"int32")
  #       WI=T.match_buffer(w_in,(col_all,),"int32")
  #       WO=T.match_buffer(w_out,(row_all,),"int32")
        
  #       for block_idx_x in T.thread_binding(0, 1, "blockIdx.x"):
  #         with T.block("b_1"):
  #           vi = T.axis.remap("S", [block_idx_x])
  #           for i in range(row_all):
  #             with T.block("b_2"):
  #               with T.init():
  #                 WO[vi*1+i]=0
  #               for j in range(col_all):
  #                 WO[vi*1+i]=WO[vi*1+i] + WI[j]*G[vi*1+i,j]
                
  #     return spmv_gpu

  # @tvm._ffi.register_func("relay.backend.concat_vector")
  # def concat_vector():
  #     @T.prim_func
  #     def concat_vector(s_info: T.handle,w_c: T.handle,w_g: T.handle,w_out: T.handle) -> None:
  #       s_c=T.var("int32")
  #       s_g=T.var("int32")
  #       s_a=T.var("int32")
  #       WC=T.match_buffer(w_c,(s_c,),"int32")
  #       WG=T.match_buffer(w_g,(s_g,),"int32")
  #       WO=T.match_buffer(w_out,(s_a,),"int32")
  #       SI=T.match_buffer(s_info,(2,s_a),"int32")
  #       CUR_R=T.alloc_buffer((2,),"int32")
        
  #       for o in range(2):
  #         CUR_R[o]=0
          
  #       for i in range(s_a):
  #         with T.block("b_1"):
  #           vi = T.axis.remap("S", [i])
  #           if SI[0,vi]<1:
  #             WO[vi]=WG[CUR_R[1]]
  #             CUR_R[1]=CUR_R[1]+1
  #           else:
  #             WO[vi]=WC[CUR_R[0]]
  #             CUR_R[0]=CUR_R[0]+1
  #     return concat_vector


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

  # print(tensor_ana())
  # print(split_tensor())
  # print(spmv_cpu())
  # print(spmv_gpu())
  # print(concat_vector())
  # print(bfs_ppocess())

  G = relay.var("G", shape=(relay.Any(), relay.Any()), dtype="int32")
  W = relay.var("W", shape=(relay.Any(),), dtype="int32")
  Pi = relay.var("Pi", shape=(relay.Any(),), dtype="int32")
  # G = relay.var("G", shape=(dim_size, dim_size), dtype="int32")
  # W = relay.var("W", shape=(dim_size,), dtype="int32")
  # Pi = relay.var("Pi", shape=(dim_size,), dtype="int32")

  step_n = relay.var("step_n", shape=(), dtype="int32")
  i = relay.var("i", shape=(), dtype="int32")
  i_tensor = relay.var("i_tensor", shape=(1,), dtype="int32")

  def body(Pi, W,i_tensor, i):
      # S_info = relay.annotation.on_device(relay.custom.tensor_ana(G),tvm.cpu())
      # G_after_split = relay.custom.split_tensor(G,S_info)
      G_C= relay.annotation.on_device(G,tvm.cpu(),constrain_body=False,constrain_result=True)
      # G_G= relay.annotation.on_device(G_after_split[1],tvm.cuda(),constrain_body=False,constrain_result=True)
      # W_copy = relay.device_copy(W,tvm.cpu(),tvm.cuda())
      # W_g = relay.annotation.on_device(relay.custom.spmv_gpu(G_G,W_copy),tvm.cuda())
      W_c = relay.annotation.on_device(relay.custom.spmv_cpu(G,W),tvm.cpu())
      # W_n = relay.annotation.on_device(relay.custom.concat_vector(S_info,W_c,W_g),tvm.cpu())
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
  
  # relay_multi_target.visualizer(module)
  print(module)
  # print(relay.transform.InferType(module))
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

loaded_lib = tvm.runtime.load_module(os.getcwd()+"/lib.so")
loaded_code = bytearray(open(os.getcwd()+"/code.ro", "rb").read())

    # deserialize.
exe2 = runtime.vm.Executable.load_exec(loaded_code, loaded_lib)
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
print(inG)
print(inW)
print(tvm_res)
