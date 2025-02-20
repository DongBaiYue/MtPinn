import csv
import time
from workers_manager import WorkersManager

def read_single_latencys(single_latencys_file_path:str):
    # 读取single_model_latency_no_collocation.csv
    single_latencys = {}
    with open(single_latencys_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        linecount = 0
        for row in reader:
            if linecount == 0:
                mpsList = [int(mps) for mps in row[1:]]
            else:
                net_name = row[0]
                single_latencys[net_name] = {}
                latencys = [round(float(latency), 2) for latency in row[1:]]
                for i in range(len(mpsList)):
                    single_latencys[net_name][mpsList[i]] = latencys[i]
            linecount += 1
    return single_latencys
def evaluate_single_model(workers_manager:WorkersManager, net_name, mps, iter_num):
    print(net_name)
    print(mps)
    start_time = time.perf_counter()
    workers_manager.worker_run(0, net_name, mps, iter_num)
    while len(workers_manager.update_free_gpu())==0:
        pass
    end_time = time.perf_counter()
    latency = (end_time-start_time)/iter_num*1000 # ms
    return latency

def write_single_latencys(single_latencys_file_path:str):
    net_names = ['FNN', 'MsFFN', 'STMsFFN', 'CNN', 'ResNet']
    mpses = [25, 50, 75, 100]
    workers_manager = WorkersManager(net_names, mpses)
    iter_num = 100
    latencys = {}
    for net_name in net_names:
        latencys[net_name] = {}
        for mps in mpses:
            latency = evaluate_single_model(workers_manager, net_name, mps, iter_num)
            print(latency)
            latencys[net_name][mps] = round(latency, 2)
    #写入csv文件
    headline = ['net name', '25', '50', '75', '100']
    with open(single_latencys_file_path, encoding='utf-8', mode='w+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headline)
        for net_name in net_names:
            line = [net_name] + [str(latencys[net_name][mps]) for mps in mpses]
            writer.writerow(line)
'''
if __name__=="__main__":
    single_latencys_file_path = "/home/ly/workspace/MtPinn/data/rocm-MI100/input/single_model_latency_no_collocation.csv"
    # write_single_latencys(single_latencys_file_path)
    # print(read_single_latencys(single_latencys_file_path))
'''