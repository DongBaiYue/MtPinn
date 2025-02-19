from multiprocessing import Process, connection
import time
import json
from workers_manager import WorkersManager
from latency_predictor import LatencyPredictor

class Request:
    request_number = 0
    def __init__(self, net_name:str, qos_latency:float, receive_time:float):
        Request.request_number += 1
        self.request_id:int = Request.request_number
        self.net_name:str = net_name
        self.qos_latency:float = qos_latency
        self.receive_time:float = receive_time
        self.mps:int = None
        self.predict_latency:float = None
        self.max_start_time:float = None
        self.start_time:float = None
        self.predict_end_time:float = None
        self.end_time:float = None
        self.mode:int = 0 #0表示waiting，1表示running, 2表示done
        self.qos_satisfy:bool = True
    
class Scheduler(Process):
    def __init__(
        self,
        pipe:connection,
        repeat_num:int,
        record_file_path:str,
        single_latencys_file_path: str
    ):
        super().__init__()
        self.pipe:connection = pipe
        self.wait_request_list:list[Request] = []
        self.running_request_list:list[Request] = []
        self.done_request_list:list[Request] = []
        self.timeout_request_list:list[Request] = []
        self.free_gpu:int = 100 #空闲GPU百分比
        self.start_time:float = None
        self.nowtime:float = None
        self.end_flag:bool = False #为1结束
        self.latencyPredictor = None
        self.workersManager:WorkersManager = None
        self.failQOSnum:int = 0 #未能满足qos的请求数量
        self.repeat_num:int = repeat_num
        self.last_receive_time = None
        self.record_file_path:str = record_file_path
        self.single_latencys_file_path:str = single_latencys_file_path
    def run(self) -> None:
        self.init_workers()
        self.init_personal()
        self.pipe.send('scheduler complete initialization')
        self.nowtime = 0
        self.start_time = time.perf_counter()
        while True:
            if self.free_gpu!=100:
                self.update_gpu_state()
            if self.end_flag:
                if self.free_gpu==100:
                    self.record_to_file()
                    # 结束该进程
                    self.pipe.send('ok')
                    self.pipe.close()
                    del self.workersManager
                    break
                else:
                    continue
            # 更新wait_request_list
            if len(self.wait_request_list)>0:
                self.update_wait()
            while self.pipe.poll():
                request_str = self.pipe.recv()
                if request_str == 'end':
                    self.end_flag = True
                else:
                    net_name, qos_latency = request_str.split(':')
                    qos_latency = 2*float(qos_latency)
                    self.__update_nowtime__()
                    request = Request(net_name, qos_latency, self.nowtime)
                    self.selectMPS(request)
                    self.insert_to_wait(request)
                    self.last_receive_time = self.nowtime
            if len(self.wait_request_list) == 0:
                if self.free_gpu > 0 and len(self.timeout_request_list)>0:
                    self.schedule_timeout()
                    pass
                #是否需要补充
                continue
            # 调度执行
            # 第一步：调度所有资源满足的队列头request
            while len(self.wait_request_list)>0 and self.free_gpu >= self.wait_request_list[0].mps:
                self.wait_to_runing(0)
            # 第二步：计算队列头request何时资源满足，寻找可在此之前完成的request
            if self.free_gpu > 0:
                self.schedule_free()
            
    def init_workers(self, net_names = ['FNN', 'MsFFN', 'STMsFFN', 'CNN', 'ResNet'], mps_list = [25, 50, 75, 100]):
        self.workersManager = WorkersManager(net_names, mps_list)
        # self.workersManager.warmup()
    def init_personal(self):
        pass
    def __update_nowtime__(self):
        self.nowtime = (time.perf_counter() - self.start_time)*1000/self.repeat_num
    def update_gpu_state(self):
        request_ids = self.workersManager.update_free_gpu()
        for request_id in request_ids:
            self.running_to_done(request_id)
    def selectMPS(self, request:Request):
        '''
        子类必须实现！
        '''
        ValueError('子类必须实现！')
    def insert_to_wait(self, request:Request):
        '''
        子类必须实现！
        '''
        ValueError('子类必须实现！')
    def wait_to_runing(self, wait_index):
        request = self.wait_request_list.pop(wait_index)
        request.mode=1
        #发送GPU推理请求
        self.__update_nowtime__()
        request.start_time = self.nowtime
        if request.predict_latency != None:
            request.predict_end_time = request.start_time + request.predict_latency
        self.workersManager.worker_run(request.request_id, request.net_name, request.mps, self.repeat_num)
        self.running_request_list.append(request)
        self.free_gpu -= request.mps
        self.print_state()

    def running_to_done(self, request_id):
        for request in self.running_request_list:
            if request.request_id == request_id:
                self.__update_nowtime__()
                self.running_request_list.remove(request)
                request.end_time = self.nowtime
                request.mode = 2
                self.free_gpu += request.mps
                if request.end_time > request.receive_time+request.qos_latency:
                    request.qos_satisfy = False
                    self.failQOSnum += 1
                else:
                    request.qos_satisfy = True
                self.done_request_list.append(request)
                break
        self.print_state()
    def schedule_free(self):
        '''
        调度空闲GPU碎片
        '''
        pass
    def update_wait(self):
        '''
        随时间推移更新wait_request_list
        '''
        pass
    def print_state(self):
        '''
        print('waiting')
        for request in self.wait_request_list:
            print('net:%s, mps:%s' % (request.net_name, request.mps), end='')
            if request.max_start_time != None:
                print(', max_start_time:%s' % request.max_start_time, end='')
            print()
        print('running')
        for request in self.running_request_list:
            print('net:%s, mps:%s' % (request.net_name, request.mps))
        print()
        '''
    def schedule_timeout(self):
        '''
        调度timeout队列
        '''
        pass
    def record_to_file(self):
        self.__update_nowtime__()
        throughput = len(self.done_request_list) / self.nowtime *1000 #每秒多少次请求
        qos_satisfy_percentage = (len(self.done_request_list) - self.failQOSnum) / (len(self.done_request_list) + len(self.timeout_request_list) + len(self.wait_request_list))
        request_list = []
        for request in self.done_request_list:
            request_info = {}
            request_info['request_id'] = request.request_id
            request_info['net_name'] = request.net_name
            request_info['receive_time'] = round(request.receive_time, 2)
            request_info['qos_latency'] = request.qos_latency
            request_info['start_time'] = round(request.start_time, 2)
            request_info['mps'] = request.mps
            if request.max_start_time != None:
                request_info['max_start_time'] = request.max_start_time
                request_info['predict_latency'] = request.predict_latency
                request_info['predict_end_time'] = round(request.predict_end_time, 2)
            request_info['end_time'] = round(request.end_time, 2)
            request_info['qos_satisfy'] = True if request.receive_time+request.qos_latency > request.end_time else False
            request_list.append(request_info)
        with open(self.record_file_path, encoding='utf-8', mode='w+') as f:
            f.write('throughput:%s 次/s\n' % (throughput))
            f.write('qos_satisfy_percentage:%s\n' % (qos_satisfy_percentage))
            f.write('done_request_num:%s \n' % (len(self.done_request_list)))
            f.write('wait_request_num:%s \n' % (len(self.wait_request_list)))
            f.write('timeout_request_num:%s \n' % (len(self.timeout_request_list)))
            for request_info in request_list:
                f.write(json.dumps(request_info))
                f.write('\n')

class MtPinnScheduler(Scheduler):
    def init_personal(self):
        self.latencyPredictor = LatencyPredictor(self.single_latencys_file_path)
    def selectMPS(self, request:Request):
        #为request分配mps
        if request.mps == None:
            if self.latencyPredictor.SatisfyMps(request.net_name, request.qos_latency):
                request.mps = self.latencyPredictor.selectMps(request.net_name, request.qos_latency)
                request.predict_latency = self.latencyPredictor.predict(request.net_name, request.mps)
                request.max_start_time = request.receive_time+request.qos_latency - request.predict_latency
            else:
                ValueError('')
    def insert_to_wait(self, request:Request):
        # 按照max_start_time从小到大排序
        if len(self.wait_request_list) == 0:
            self.wait_request_list.append(request)
        elif request.max_start_time >= self.wait_request_list[-1].max_start_time:
            self.wait_request_list.append(request)
        else:
            for wait_index in range(len(self.wait_request_list)):
                if request.max_start_time < self.wait_request_list[wait_index].max_start_time:
                    self.wait_request_list.insert(wait_index, request)
                    break
        self.print_state()
    def wait_to_runing(self, wait_index):
        '''
        由于cal_until_free_time，running必须按照predict_end_time排序
        '''
        request = self.wait_request_list.pop(wait_index)
        request.mode=1
        #发送GPU推理请求
        self.__update_nowtime__()
        request.start_time = self.nowtime
        request.predict_end_time = request.start_time + request.predict_latency
        self.workersManager.worker_run(request.request_id, request.net_name, request.mps, self.repeat_num)
        #按照predict_end_time从小到大排序
        if len(self.running_request_list) == 0 or request.predict_end_time >= self.running_request_list[-1].predict_end_time:
            self.running_request_list.append(request)
        else:
            for run_index in range(len(self.running_request_list)):
                if request.predict_end_time < self.running_request_list[run_index].predict_end_time:
                    self.running_request_list.insert(run_index, request)
                    break
        self.free_gpu -= request.mps
        self.print_state()
    def schedule_free(self):
        wait_index = 1
        if self.free_gpu > 0 and wait_index < len(self.wait_request_list):
            until_free_time = self.cal_until_free_time(self.wait_request_list[0].mps)
        while self.free_gpu > 0 and wait_index < len(self.wait_request_list):
            request = self.wait_request_list[wait_index]
            if self.free_gpu >= request.mps and self.nowtime+request.predict_latency<=until_free_time:
                self.wait_to_runing(wait_index)
            else:
                wait_index += 1
    def cal_until_free_time(self, need_free_gpu):
        need_new_free_gpu = need_free_gpu - self.free_gpu
        # 无需等待
        if need_new_free_gpu <= 0:
            return self.nowtime
        for request in self.running_request_list:
            need_new_free_gpu -= request.mps
            if need_new_free_gpu <= 0:
                return max(self.nowtime, request.predict_end_time)
    def insert_to_timeout(self, request:Request):
        # 按照predict_latency从小到大排序
        if len(self.timeout_request_list) == 0:
            self.timeout_request_list.append(request)
        elif request.predict_latency >= self.timeout_request_list[-1].predict_latency:
            self.timeout_request_list.append(request)
        else:
            for wait_index in range(len(self.timeout_request_list)):
                if request.predict_latency < self.timeout_request_list[wait_index].predict_latency:
                    self.timeout_request_list.insert(wait_index, request)
                    break
    def update_wait(self):
        # 更新wait_request_list
        # 切片创建新列表用于迭代
        for request in self.wait_request_list[:]:
            if request.qos_satisfy and request.max_start_time < self.nowtime:
                old_mps = request.mps
                if self.latencyPredictor.SatisfyMps(request.net_name, request.receive_time + request.qos_latency-self.nowtime):
                    request.mps = self.latencyPredictor.selectMps(request.net_name, request.receive_time + request.qos_latency-self.nowtime)
                    request.predict_latency = self.latencyPredictor.predict(request.net_name, request.mps)
                    request.max_start_time = request.receive_time + request.qos_latency-request.predict_latency
                    self.wait_request_list.remove(request)
                    self.insert_to_wait(request)
                    print('nowtime:%s, net:%s, old mps:%s, new mps:%s, new_max_start_time:%s, qos_satisfy:%s' % (self.nowtime, request.net_name, old_mps, request.mps, request.max_start_time, request.qos_satisfy))
                else:
                    # 是否修改mps？
                    #request.mps = 25
                    #request.predict_latency = self.latencyPredictor.predict(request.net_name, request.mps)
                    request.qos_satisfy = False
                    self.wait_request_list.remove(request)
                    # self.timeout_request_list.append(request)
                    self.insert_to_timeout(request)
                    print('nowtime:%s, net:%s, old mps:%s, new mps:%s, qos_satisfy:%s' % (self.nowtime, request.net_name, old_mps, request.mps, request.qos_satisfy))
                self.print_state()
    def schedule_timeout(self):
        inteval = 10
        next_receive_time = self.last_receive_time + inteval
        request = self.timeout_request_list[0]
        self.__update_nowtime__()
        if self.free_gpu >= request.mps and self.nowtime+request.predict_latency < next_receive_time:
            self.timeout_request_list.remove(request)
            self.wait_request_list.append(request)
            self.wait_to_runing(0)

    

class ParallelFCFsScheduler(Scheduler):
    def init_personal(self):
        self.latencyPredictor = LatencyPredictor(self.single_latencys_file_path)
    def selectMPS(self, request: Request):
        #为request分配mps
        if request.mps == None:
            request.mps = max(50, self.latencyPredictor.selectMaxThroughput(request.net_name))
            # request.predict_latency = self.latencyPredictor.predict(request.net_name, request.mps)
    def insert_to_wait(self, request:Request):
        self.wait_request_list.append(request)
        self.print_state()
    
class SerialFCFsScheduler(Scheduler):
    def selectMPS(self, request: Request):
        '''
        为request分配mps
        '''
        request.mps=100
        
    def insert_to_wait(self, request:Request):
        self.wait_request_list.append(request)
        self.print_state()

class SerialQosScheduler(Scheduler):
    def init_personal(self):
        self.latencyPredictor = LatencyPredictor(self.single_latencys_file_path)
    def selectMPS(self, request: Request):
        '''
        为request分配mps
        '''
        request.mps=100
    def insert_to_wait(self, request:Request):
        request.predict_latency = self.latencyPredictor.predict(request.net_name, request.mps)
        request.max_start_time = request.receive_time+request.qos_latency - request.predict_latency
        
        if len(self.wait_request_list) == 0:
            self.wait_request_list.append(request)
            return
        # 按照max_start_time从小到大排序
        if request.max_start_time >= self.wait_request_list[-1].max_start_time:
            self.wait_request_list.append(request)
        else:
            for wait_index in range(len(self.wait_request_list)):
                if request.max_start_time < self.wait_request_list[wait_index].max_start_time:
                    self.wait_request_list.insert(wait_index, request)
                    break
        self.print_state()
    def insert_to_timeout(self, request:Request):
        # 按照predict_latency从小到大排序
        if len(self.timeout_request_list) == 0:
            self.timeout_request_list.append(request)
        elif request.predict_latency >= self.timeout_request_list[-1].predict_latency:
            self.timeout_request_list.append(request)
        else:
            for wait_index in range(len(self.timeout_request_list)):
                if request.predict_latency < self.timeout_request_list[wait_index].predict_latency:
                    self.timeout_request_list.insert(wait_index, request)
                    break
    def update_wait(self):
        # 更新wait_request_list
        # 切片创建新列表用于迭代
        for request in self.wait_request_list[:]:
            if request.qos_satisfy and request.max_start_time < self.nowtime:
                request.qos_satisfy = False
                self.wait_request_list.remove(request)
                self.insert_to_timeout(request)
                print('nowtime:%s, net:%s, qos_satisfy:%s' % (self.nowtime, request.net_name, request.qos_satisfy))
                self.print_state()
    def schedule_timeout(self):
        inteval = 10
        next_receive_time = self.last_receive_time + inteval
        request = self.timeout_request_list[0]
        self.__update_nowtime__()
        if self.free_gpu >= request.mps and self.nowtime+request.predict_latency < next_receive_time:
            self.timeout_request_list.remove(request)
            self.wait_request_list.append(request)
            self.wait_to_runing(0)
