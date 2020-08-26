#pragma once

#include "Config.h"
#include "CpuNumaUtils.h"

#include <algorithm>
#include <numa.h>
#include <string>
#include <vector>

#ifdef CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace rdma {

class GpuNumaUtils {
public: 

    static std::vector<std::string> get_gpu_to_pci_map(){
        std::vector<std::string> gpu_map;
        #ifdef CUDA_ENABLED
            int deviceCount = 0;
            cudaGetDeviceCount(&deviceCount);
            for(int deviceId=0; deviceId<deviceCount; deviceId++){
                char pciBusId[255];
                if(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), deviceId) == cudaSuccess){
                    gpu_map.push_back(std::string(pciBusId));
                } else {
                    gpu_map.push_back(std::string(""));
                }
            }
        #endif
        return gpu_map;
    }
    
    static int get_cuda_device_index_by_cpu(int cpu){
        // TODO automate somehow with get_gpu_to_pci_map() and remove Config entry
        for(size_t gpuIndex = 0; gpuIndex < Config::GPUS_TO_CPU_AFFINITY.size(); gpuIndex++){
            std::vector<int> cpus = Config::GPUS_TO_CPU_AFFINITY[gpuIndex];
            if(std::find(cpus.begin(), cpus.end(), cpu) != cpus.end()){
                return gpuIndex;
            }
        } return -1;
    }
    static int get_cuda_device_index_by_cpu(){
        return get_cuda_device_index_by_cpu(rdma::CpuNumaUtils::get_current_cpu());
    }

    static int get_cuda_device_index_by_numa(int numaNode){
        std::vector<int> cpus = rdma::Config::NUMA_THREAD_CPUS[numaNode];
        for(int &cpu : cpus){
            int gpuIndex = get_cuda_device_index_by_cpu(cpu);
            if(gpuIndex >= 0) return gpuIndex;
        } return -1;
    }

    static int get_cuda_device_index_by_numa(){
        return get_cuda_device_index_by_numa(rdma::Config::RDMA_NUMAREGION);
    }

};
}