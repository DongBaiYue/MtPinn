import os
root_dir = "/home/ly/workspace/MtPinn"
os.chdir(root_dir)

from workers import Workers
from typing import List, Dict
import time
stream_mask4 = ['0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']

class WorkersManager:
    def __init__(self, net_names: List[str], mpses: List[int], stream_mask4=stream_mask4):
        self.workers = Workers(net_names, mpses, stream_mask4)
        self.streams:List[str] = stream_mask4
        self.running_streams_requestIDs:Dict[str, int] = {}
        self.free_gpu = 100
        self.free_stream_masks = [True, True, True, True]
    def update_free_gpu(self):
        done_request_ids = []
        for stream in self.running_streams_requestIDs.keys():
            if self.workers.stream_query(stream):
                # update running_streams、running_request_ids and free gpu 、free_stream_masks
                request_id = self.running_streams_requestIDs.pop(stream)
                self.__add_free_stream__(stream)
                done_request_ids.append(request_id)
                break
        return done_request_ids

    def worker_run(self, request_id:int, net_name:str, mps:int, repeat_num:int):
        # 空闲资源量检查
        if self.free_gpu < mps:
            ValueError('illegal free gpu!')
        # 获取合适mask
        stream = self.__get_free_stream__(mps)
        # print(stream)
        self.workers.module_run(net_name, mps, stream, repeat_num)
        # update running_streams、running_request_ids and free gpu 、free_stream_masks
        self.running_streams_requestIDs[stream] = request_id
        self.__delete_free_stream__(stream)

    def __get_free_stream__(self, mps):
        stream = ""
        for i in range(3, -1, -1):
            if mps > 0:
                if self.free_stream_masks[i]:
                    stream='1'+stream
                    mps -= 25
                else:
                    stream='0'+stream
            else:
                stream='0'+stream
        return stream
    def __delete_free_stream__(self, stream):
        for i in range(4):
            if stream[i] == '1':
                self.free_stream_masks[i] = False
                self.free_gpu -= 25
    def __add_free_stream__(self, stream):
        for i in range(4):
            if stream[i] == '1':
                self.free_stream_masks[i] = True
                self.free_gpu += 25
    def __del__(self):
        del self.workers
         

if __name__ == "__main__":
    net_names = ['FNN', 'MsFFN']
    mpss = [100]
    workers_manager = WorkersManager(net_names, mpss)
    start_time = time.perf_counter()
    workers_manager.worker_run(1, 'FNN', 100, 10)
    #workers_manager.worker_run(2, 'MsFFN', 25, 10)
    while len(workers_manager.update_free_gpu())==0:
        pass
    end_time = time.perf_counter()
    time_str = 'total: %.3fms\n' % ((end_time-start_time)*1000)
    print(time_str)

