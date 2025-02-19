import os
os.chdir('/home/ly/workspace/MtPinn/')
import sys
if len(sys.argv) < 3:
    print("Need Kernel Name and MPS Configuration")
    sys.exit()
else:
    net_name = str(sys.argv[1])
    MPS = str(sys.argv[2])
    print('net:%s, MPS:%s' % (net_name, MPS))

cu_mask = [None]*4
# 30, 60, 90, 120
if int(MPS) == 25:
    mask4 = "0001"
elif int(MPS) == 50:
    mask4 = "0011"
elif int(MPS) == 75:
    mask4 = "0111"
elif int(MPS) == 100:
    mask4 = "1111"

mask128 = '00000000'
# 当步长为负时，从右向左
for x in mask4[4::-1]:
    mask128 += x*30
cu_mask = [None]*4
# 转换为16进制
cu_mask[3] = hex(int(mask128[:32],2))
cu_mask[2] = hex(int(mask128[32:64],2))
cu_mask[1] = hex(int(mask128[64:96],2))
cu_mask[0] = hex(int(mask128[96:128],2))


import os
os.environ['ENABLE_CU_MASK'] = "1"
os.environ['CU_MASK_0'] = cu_mask[0]
os.environ['CU_MASK_1'] = cu_mask[1]
os.environ['CU_MASK_2'] = cu_mask[2]
os.environ['CU_MASK_3'] = cu_mask[3]

if net_name == 'FNN':
    input_size = (32, 1)
elif net_name == 'MsFFN':
    input_size = (32, 1)
elif net_name == 'STMsFFN':
    input_size = (32, 2)
elif net_name == 'CNN':
    input_size = (1024*8, 21*21)
elif net_name == 'ResNet':
    input_size = (32, 1)
import numpy as np
input_data = np.random.rand(*input_size).astype(np.float32)

model_path = 'onnx_model/%s.onnx' % net_name
import onnx
onnx_model = onnx.load(model_path)

import tvm
from tvm import relay
input_name = 'data0'
shape_dict = {input_name:input_size}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

target = "rocm"
dev = tvm.device(target, 0)
# 新增stream作为默认stream
stream = dev.create_raw_stream()
dev.set_raw_stream(stream)
os.environ['StreamID'] = "0"


log_file = 'ansor_log/rocm-MI100/%s/fix-%s-%s-%s.json' % (net_name, target, net_name, MPS)
from tvm import auto_scheduler
'''
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file = log_file)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=500,  # change this to 20000 to achieve the best performance
    early_stopping=200,
    runner = auto_scheduler.LocalRunner(timeout=50),
    builder = auto_scheduler.LocalBuilder(timeout=150),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=1,
)
# Run auto-tuning (search)
tuner.tune(tune_option)
'''

# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
from tvm.contrib import graph_executor
module = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
module.set_input(input_name, tvm.nd.array(input_data))
# Execute
module.run()
# Get outputs
#print(module.get_output(0).numpy()[:5])
# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
