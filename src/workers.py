import tvm
from tvm import relay
import onnx
import numpy as np
import os
from typing import List
from pyhip import hip

class Workers:
    def __init__(self, net_names:List[str], mpses:List[int], stream_mask4:List[str], target='rocm', deviceId=0) -> None:
        self.target = target
        self.dev = tvm.device(target, deviceId)
        self.modules = {}
        self.create_modules(net_names, mpses)
        self.streams = {}
        self.events = {}
        self.__init_streams__(stream_mask4)
    def module_run(self, net_name:str, mps:int, mask4:str, repeat_num:int):
        self.__set_stream__(mask4)
        for i in range(repeat_num):
            self.modules[net_name][mps].run()
        hip.hipEventRecord(self.events[mask4], self.streams[mask4])

    def stream_query(self, mask4:str)->bool:
        return hip.hipEventQuery(self.events[mask4])
    def create_modules(self, net_names:List[str], mpses:List[int]):
        for net_name in net_names:
            self.modules[net_name] = {}
            for mps in mpses:
                self.modules[net_name][mps] = self.create_single_module(net_name, mps)
    def create_single_module(self, net_name: str, mps: int):
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
    
        input_data = np.random.rand(*input_size).astype(np.float32)

        model_path = 'onnx_model/%s.onnx' % net_name
        onnx_model = onnx.load(model_path)
        input_name = 'data0'
        shape_dict = {input_name:input_size}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        from tvm import auto_scheduler
        log_file = 'ansor_log/rocm-MI100/%s/%s-%s-%s.json' % (net_name, self.target, net_name, mps)
        # Compile with the history best
        print("Compile..., net:%s, mps:%s" % (net_name, mps))
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=self.target, params=params)
        # Create graph executor
        from tvm.contrib import graph_executor
        module = graph_executor.GraphModule(lib["default"](self.dev))
        # Set inputs
        module.set_input(input_name, tvm.nd.array(input_data))
        # Execute
        module.run()
        # Get outputs
        # print(module.get_output(0).numpy()[:5])
        # Evaluate
        # print("Evaluate inference time cost...")
        #print(module.benchmark(self.dev, repeat=3, min_repeat_ms=500))
        return module
    def __init_streams__(self, stream_mask4):
        for mask4 in stream_mask4:
            self.__set_cu_env__(mask4)
            stream = self.dev.create_raw_stream()
            self.streams[mask4] = stream
            self.events[mask4] = hip.hipEventCreate()
    def __set_stream__(self, mask4:str):
        self.dev.set_raw_stream(self.streams[mask4])
    def __set_cu_env__(self, mask4:str):
        '''
        mask4: Each represents 25% of the computing resources, 30 CUs on MI100.
        mask128: Each represents 1/120 of the computing resources, 1 CU on MI100.
        '''
        mask128 = '00000000'
        for x in mask4[4::-1]:
            mask128 += x*30
        cu_mask = [None]*4
        # 转换为16进制
        cu_mask[3] = hex(int(mask128[:32],2))
        cu_mask[2] = hex(int(mask128[32:64],2))
        cu_mask[1] = hex(int(mask128[64:96],2))
        cu_mask[0] = hex(int(mask128[96:128],2))
        os.environ['ENABLE_CU_MASK'] = "1"
        os.environ['CU_MASK_0'] = cu_mask[0]
        os.environ['CU_MASK_1'] = cu_mask[1]
        os.environ['CU_MASK_2'] = cu_mask[2]
        os.environ['CU_MASK_3'] = cu_mask[3]
    def __del__(self):
        del self.modules
