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

    static int get_current_cpu(){
        return sched_getcpu();
    }

    static int get_current_numa_node(){
        return numa_node_of_cpu(get_current_cpu());
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

    
    static int get_cuda_device_index_by_cpu(int cpu){
        // TODO automate somehow with get_gpu_to_pci_map() and remove Config entry
        for(size_t gpuIndex = 0; gpuIndex < Config::GPUS_TO_CPU_AFFINITY.size(); gpuIndex++){
            std::vector<int> numas = Config::GPUS_TO_CPU_AFFINITY[gpuIndex];
            if(std::find(numas.begin(), numas.end(), cpu) != numas.end()){
                return gpuIndex;
            }
        } return -1;
    };
    static int get_cuda_device_index_by_cpu(){
        return get_cuda_device_index_by_cpu(get_current_cpu());
    };

};
}

#endif