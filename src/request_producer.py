import random
from latency_evaluate import read_single_latencys


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

def request_produce():
    single_latencys_file_path = "/home/ly/workspace/MtPinn/data/rocm-MI100/input/single_model_latency_no_collocation.csv"
    requests_file_path = '/home/ly/workspace/MtPinn/data/rocm-MI100/input/requests/single_STMsFFN-1000.txt'
    equal_net_posses = {'FNN': 20,'MsFFN': 20,'STMsFFN': 20,'CNN': 20,'ResNet': 20}
    short_skew_net_posses = {'FNN': 40,'MsFFN': 5,'STMsFFN': 5,'CNN': 10,'ResNet': 40}
    middle_skew_net_posses = {'FNN': 5,'MsFFN': 40,'STMsFFN': 40,'CNN': 10,'ResNet': 5}
    long_skew_net_posses = {'FNN': 5,'MsFFN': 5,'STMsFFN': 5,'CNN': 80,'ResNet': 5}
    net_posses= {'STMsFFN': 100}
    random_request_produce(requests_file_path, single_latencys_file_path, net_posses)
