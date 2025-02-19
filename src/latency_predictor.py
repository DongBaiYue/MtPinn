from utils import read_single_latencys

class LatencyPredictor:
    def __init__(self, single_latencys_file_path:str):
        self.single_latencys = read_single_latencys(single_latencys_file_path)
        self.optional_mps = [25, 50, 75, 100] # 升序
    def predict(self, net_name, mps):
        latency = self.single_latencys[net_name][mps]
        if mps == 25:
            latency = self.single_latencys[net_name][mps] * 1.2
        elif mps == 50:
            latency = self.single_latencys[net_name][mps] * 1.1
        elif mps == 75:
            latency = self.single_latencys[net_name][mps] * 1.05
        #latency = round(latency, 2)
        return latency
    def selectMaxThroughput(self, net_name):
        '''
        return: 资源利用率最大的mps
        '''
        throughputs = [self.predict(net_name, mps) * mps for mps in self.optional_mps]
        select_mps = self.optional_mps[throughputs.index(min(throughputs))]
        return select_mps
    def selectMinSatisfyQos(self, net_name, qos_latency):
        '''
        return: 满足qos的最小mps
        '''
        for mps in self.optional_mps:
            if self.predict(net_name, mps) < qos_latency:
                return mps
        ValueError('Qos cannot statisfy!')
    def SatisfyMps(self, net_name, qos_latency)->bool:
        '''
        return: QoS是否可满足
        '''
        return self.predict(net_name, 100) < qos_latency
    def selectMps(self, net_name, qos_latency):
        return max(self.selectMaxThroughput(net_name), self.selectMinSatisfyQos(net_name, qos_latency))
'''
if __name__=="__main__":
    single_latencys_file_path = "/home/ly/workspace/MtPinn/data/rocm-MI100/input/single_model_latency_no_collocation.csv"
    latency_predictor = LatencyPredictor(single_latencys_file_path)
    net_names = ['FNN', 'MsFFN', 'STMsFFN', 'CNN', 'ResNet']
    for net_name in net_names:
        print(latency_predictor.selectMaxThroughput(net_name))
'''