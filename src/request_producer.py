import random
from scipy import stats
import multiprocessing
from utils import read_single_latencys
from scheduler import Scheduler, MtPinnScheduler, ParallelFCFsScheduler, SerialQosScheduler, SerialFCFsScheduler


def random_request_produce(requests_file_path:str, single_latencys_file_path:str, net_posses={'FNN':20, 'MsFFN':20, 'STMsFFN':20, 'CNN':20, 'ResNet':20}):
    # qos latency可能为100%下时延的若干倍，key为倍数，value为poss
    qos_posses = {2:1, 5:3, 10:2}
    qos_size = 1000
    single_latencys = read_single_latencys(single_latencys_file_path)
    random.seed(1234)
    with open(requests_file_path, encoding='utf-8', mode='w+') as f:
        for _ in range(qos_size):
            #确定net_name
            random_x = random.randint(0, sum(net_posses.values()))
            sum_poss = 0
            for net_name, poss in net_posses.items():
                sum_poss += poss
                if random_x < sum_poss:
                    break
            #确定qos
            random_x = random.randint(0, sum(qos_posses.values()))
            sum_poss = 0
            for qos, poss in qos_posses.items():
                sum_poss += poss
                if random_x < sum_poss:
                    break
            qos_latency = qos * single_latencys[net_name][100]
            qos_latency = round(qos_latency, 2)
            f.write('%s:%s, ' % (net_name, qos_latency))

def poisson_request_produce(requests_file_path:str, average_batch_size:int, batch_num:int, record_file_path:str, Scheduler:Scheduler, single_latencys_file_path:str):
    time_interval = 0.01 #s
    # average_batch_size *= 2
    time_interval *= 2
    repeat_num = 10
    with open(requests_file_path, encoding='utf-8', mode='r') as f:
        line = f.readline()
    request_list = line.split(', ')[:-1]
    poisson_sample = stats.poisson.rvs(mu=average_batch_size, size=batch_num, random_state=0)

    pipe_parent, pipe_child = multiprocessing.Pipe()
    sheduler = Scheduler(pipe_child, repeat_num, record_file_path, single_latencys_file_path)
    sheduler.start()
    pipe_child.close()
    print(pipe_parent.recv())

    request_index = 0
    for request_num in poisson_sample:
        if(request_index+request_num<=len(request_list)):
            requests = request_list[request_index:request_index+request_num]
            request_index+=request_num
        elif request_index < len(request_list):
            requests = request_list[request_index:]
            request_index = len(request_list)
        else:
            break
        for request in requests:
            pipe_parent.send(request)
        import time
        time.sleep(time_interval*repeat_num)
    pipe_parent.send('end')
    # 确认子进程结束
    print(pipe_parent.recv())

def request_produce():
    single_latencys_file_path = "/home/ly/workspace/MtPinn/data/rocm-MI100/input/single_model_latency_no_collocation.csv"
    requests_file_path = '/home/ly/workspace/MtPinn/data/rocm-MI100/input/requests/single_STMsFFN-1000.txt'
    equal_net_posses = {'FNN': 20,'MsFFN': 20,'STMsFFN': 20,'CNN': 20,'ResNet': 20}
    short_skew_net_posses = {'FNN': 40,'MsFFN': 5,'STMsFFN': 5,'CNN': 10,'ResNet': 40}
    middle_skew_net_posses = {'FNN': 5,'MsFFN': 40,'STMsFFN': 40,'CNN': 10,'ResNet': 5}
    long_skew_net_posses = {'FNN': 5,'MsFFN': 5,'STMsFFN': 5,'CNN': 80,'ResNet': 5}
    net_posses= {'STMsFFN': 100}
    random_request_produce(requests_file_path, single_latencys_file_path, net_posses)
def scheduler_run(scheduler_mode, average_batch_size, input_name):
    single_latencys_file_path = "/home/ly/workspace/MtPinn/data/rocm-MI100/input/single_model_latency_no_collocation.csv"
    requests_file_path = '/home/ly/workspace/MtPinn/data/rocm-MI100/input/requests/%s-1000.txt' % input_name
    average_batch_size = average_batch_size
    batch_num = 20
    scheduler_mode = scheduler_mode # MtPINN、FF、SF、SQ

    record_file_path = '/home/ly/workspace/MtPinn/data/rocm-MI100/output/new/%s/%s-%s-%s.json'  % (input_name, input_name, average_batch_size, scheduler_mode)
    scheduler_modes = {'MtPINN':MtPinnScheduler, 'FF':ParallelFCFsScheduler, 'SF':SerialFCFsScheduler, 'SQ':SerialQosScheduler}
    poisson_request_produce(requests_file_path, average_batch_size, batch_num, record_file_path, scheduler_modes[scheduler_mode], single_latencys_file_path)

def max_arrival():
    # ['SF', 'FF', 'SQ', 'MtPINN']
    for scheduler_mode in ['MtPINN']:
        for input_name in ['short_skew']:
            for average_batch_size in range(16, 20):
                scheduler_run(scheduler_mode, average_batch_size, input_name)
                record_file_path = '/home/ly/workspace/MtPinn/data/rocm-MI100/output/new/%s/%s-%s-%s.json'  % (input_name, input_name, average_batch_size, scheduler_mode)
                with open(record_file_path, mode='r') as f:
                    f.readline()
                    qos = float(f.readline().strip().split(":")[1].split()[0])
                    if(qos < 0.95):
                        break
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    import os
    os.chdir('/home/ly/workspace/MtPinn/')
    max_arrival()

'''
if __name__ == '__main__':
    # request_produce()
    import os
    os.chdir('/home/ly/workspace/MtPinn/')
    import sys
    if len(sys.argv) < 4:
        print("Need scheduler_mode and average_batch_size and net name")
        sys.exit()
    else:
        scheduler_mode = str(sys.argv[1])
        average_batch_size = int(sys.argv[2])
        input_name = str(sys.argv[3])
        print('scheduler_mode:%s, average_batch_size:%s, input_name:%s' % (scheduler_mode, average_batch_size, input_name))
    scheduler_run(scheduler_mode, average_batch_size, input_name)
'''