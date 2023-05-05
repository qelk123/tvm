from tvm import relay, target
import tvm
import numpy as np
from tvm import relay, auto_scheduler
from tvm.runtime.vm import VirtualMachine
import time
from tvm import meta_schedule as ms
from tvm.meta_schedule import ApplyHistoryBest, postproc, schedule_rule
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm.meta_schedule.tune import tune_extracted_tasks


dim_size=5
step=5

def get_np_array(var, dtype):
    return np.random.randn(*[int(x) for x in var.type_annotation.shape]).astype(dtype)


G = relay.var("G", shape=(dim_size, dim_size), dtype="int32")
W = relay.var("W", shape=(dim_size,), dtype="int32")
Pi = relay.var("Pi", shape=(dim_size,), dtype="int32")
i = relay.var("i", shape=tuple(), dtype="int32")

def myfun(Pi, W, i):
    W2 = relay.custom.custombfs(G,W,Pi,i)
    Pi2 = relay.custom.updatepi(W2,Pi,i)
    j = relay.add(i, relay.const(1, "int32"))
    return Pi2,W2, j
def cond(Pi, W, i):
    return relay.less(i, relay.const(step, "int32"))

myloop = relay.loops.while_loop(cond, [Pi, W, i], myfun)
z = myloop(Pi,W, relay.const(0, "int32"))
result = relay.Function([G,W,Pi], relay.TupleGetItem(z, 0))

print(result.astext())
zz = result
module = tvm.IRModule.from_expr(zz)
print(module["main"])

print("Extract tasks...")
G_p, W_p, Pi_p = get_np_array(G, "int32"), get_np_array(W, "int32"), get_np_array(Pi, "int32")
target2 = tvm.target.Target("llvm -num-cores 32")


tasks, task_weights = auto_scheduler.extract_tasks(module["main"], params={"G":G_p,"W":W_p,"Pi":Pi_p}, target=target2)
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
    


use_sparse = False
log_file = "bfs_relay.json"
def run_tuning():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)
        
# run_tuning()

# print(tasks[0].print_best(log_file))




extracted_tasks = extract_task_from_relay(module, params={"G":G_p,"W":W_p,"Pi":Pi_p}, target=target2)

# tune_tasks = list(
#     filter(
#         lambda task: op_name in task.task_name,
#         extracted_tasks,
#     )
# )
import tempfile
CONFIG = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=32,
    max_trials_per_task=32,
    max_trials_global=20000,
)
with tempfile.TemporaryDirectory() as work_dir:
    database = tune_extracted_tasks(
        extracted_tasks,
        CONFIG,
        work_dir=work_dir,
        # sch_rules=lambda: sch_rules,
        # postprocs=lambda: postprocs,
    )


#-----------------------compile and eval-------------------------
inG = np.random.randint(2, size=[dim_size,dim_size]).astype("int32")
n_inG = inG.transpose()
inW = np.zeros([dim_size,]).astype("int32")
inW[0]=1
inPi = np.ones([dim_size,]).astype("int32")
inPi = inPi*-1
inPi[0]=0


with ApplyHistoryBest(database):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"], config={"relay.backend.use_meta_schedule": True}):
        vm_exec = relay.vm.compile(module, target="llvm", params={"G":inG,"W":inW,"Pi":inPi})


# with auto_scheduler.ApplyHistoryBest(log_file):
#     with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"], config={"relay.backend.use_auto_scheduler": True}):
#         vm_exec = relay.vm.compile(module, target="llvm", params={"G":inG,"W":inW,"Pi":inPi})
         
with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"], config={"relay.backend.use_auto_scheduler": False}):
    vm_exec2 = relay.vm.compile(module, target="llvm", params={"G":n_inG,"W":inW,"Pi":inPi})
    
dev = tvm.cpu()
vm = VirtualMachine(vm_exec, dev)
vm2 = VirtualMachine(vm_exec2, dev)
# print("Inference time of model after tuning: {:0.4f}".format(time.time() - start_t))

vm2.set_input("main", **{"G":n_inG,"W":inW,"Pi":inPi})
tvm_res2 = vm2.run()

start_t = time.time()
for i in range(100):
    tvm_res2 = vm2.run()
print(inG)
print("Inference time of model before tuning: {:0.4f}".format(time.time() - start_t))
print(tvm_res2)

vm.set_input("main", **{"G":inG,"W":inW,"Pi":inPi})
tvm_res = vm.run()

start_t = time.time()
for i in range(100):
    tvm_res = vm.run()
print("Inference time of model after tuning: {:0.4f}".format(time.time() - start_t))
print(tvm_res)

# print(tvm_res[0].numpy().tolist())

# with auto_scheduler.ApplyHistoryBest(log_file):
#     with relay.build_config(opt_level=3):
#         ex = relay.create_executor("vm", mod=module,  target="llvm")  #"debug"

# with relay.build_config(opt_level=3):
#     ex2 = relay.create_executor("vm", mod=module,  target="llvm")  #"debug"

# inW=np.zeros(dim_size).astype("int32")
# inW[0]=1
# inW = tvm.nd.array(inW)
# inPi = np.ones(dim_size).astype("int32")
# inPi = inPi*-1
# inPi[0] = 0

# inPi = tvm.nd.array(inPi)
# import time
# start_time = time.time()
# re = ex.evaluate()(inG,inW,inPi)
# end_time = time.time()
# print("latency:"+str(end_time-start_time))
# start_time = time.time()
# re2 = ex2.evaluate()(inG,inW,inPi)
# end_time = time.time()
# print("latency:"+str(end_time-start_time))
# print(inW)
# print(inPi)
# print(inG)
# print(re)