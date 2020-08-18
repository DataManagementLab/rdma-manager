#ifndef GPUNUMAUTILS_HPP_
#define GPUNUMAUTILS_HPP_

#include "Config.h"

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

    static int get_current_numa_node(){
        int cpu = sched_getcpu();
        return numa_node_of_cpu(cpu);
    };

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
    };

    
    static int get_cuda_device_index_for_numa_node(int node){
        // TODO automate somehow with get_gpu_to_pci_map() and remove Config entry
        for(size_t gpuIndex = 0; gpuIndex < Config::NUMA_GPUS.size(); gpuIndex++){
            std::vector<int> numas = Config::NUMA_GPUS[gpuIndex];
            if(std::find(numas.begin(), numas.end(), node) != numas.end())
                return gpuIndex;
        } return -1;
    };
    static int get_cuda_device_index_for_numa_node(){
        return get_cuda_device_index_for_numa_node(get_current_numa_node());
    };

};
}

#endif