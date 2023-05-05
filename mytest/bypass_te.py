import os
print(os.getpid())
import tvm
from tvm import relay
from tvm.script import tir as T
from tvm.relay import vm
import numpy as np
from tvm import runtime


@tvm._ffi.register_func("relay.backend.custombfs")
def custombfs():
    @T.prim_func
    def custombfs(g: T.handle,w: T.handle, pi: T.handle,i: T.handle,ret: T.handle) -> None:
      m=T.var("int32")
      # G=T.match_buffer(g,(m,m),"int32")
      W=T.match_buffer(w,(m,),"int32")
      Pi=T.match_buffer(pi,(m,),"int32")
      RET=T.match_buffer(ret,(m,),"int32")
      for i in range(m):
        with T.block("outer_1"):
          if i<2:
            RET[i]=W[i]-Pi[i]
            Pi[i]=99
          else:
            RET[i]=987654321
            
    
    mod = tvm.IRModule.from_expr(custombfs)
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("outer_1")
    i = sch.get_loops(blk)
    print(len(i))
    io, ii = sch.split(i[0], [None, 2])
    print(sch.mod["main"])
    return sch.mod["main"]


custombfs()


G = relay.var("G", shape=(5, 5), dtype="int32")
W = relay.var("W", shape=(5,), dtype="int32")
Pi = relay.var("Pi", shape=(5,), dtype="int32")
i = relay.var("i", shape=(1,), dtype="int32")
mid = relay.op.annotation.on_device(relay.custom.custombfs(G,W,Pi,i), tvm.cpu())
mid2 = relay.op.annotation.on_device(relay.custom.custombfs(G,mid,Pi,i), tvm.cpu())
z = relay.var("z", shape=(5,), dtype="int32")
# z2 = relay.var("z", shape=(5,), dtype="int32")
res = relay.add(z,mid2)
res2 = relay.op.annotation.on_device(res, tvm.cuda())

func = relay.Function([G,W,Pi,i,z], res2)




module = tvm.IRModule.from_expr(func)
with tvm.transform.PassContext(opt_level=3, config={"relay.fallback_device_type": tvm.cuda().device_type,"relay.backend.use_tvm_script": True}):
    exe = relay.vm.compile(
      module, target={"cpu": tvm.target.Target("llvm"), "cuda": tvm.target.Target("cuda")}
    )
    code, lib = exe.save()
    with open("bypass_te_cpu.ll", "w") as external_file:
      print(lib.get_source(),file=external_file)
    vm = runtime.vm.VirtualMachine(exe, [tvm.cuda(), tvm.cpu()])  
    a=np.ones((5,5), dtype='int32')
    re = vm.invoke("main",a,np.ones((5), dtype='int32'),np.ones((5), dtype='int32'),np.ones((1), dtype='int32'),np.zeros((5), dtype='int32'))
print(a)
print(re)
