from scipy import stats
import multiprocessing
from scheduler import Scheduler, MtPinnScheduler, ParallelFCFsScheduler, SerialQosScheduler, SerialFCFsScheduler

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
            for average_batch_size in range(1, 20):
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