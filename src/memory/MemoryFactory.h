#pragma once

#include "../utils/Config.h"
#include "MainMemory.h"
#include "CudaMemory.h"
#include "LocalMainMemoryStub.h"
#include "LocalMainMemoryStub.h"
#include "LocalCudaMemoryStub.h"

#ifndef HUGEPAGE
#define HUGEPAGE false
#endif

namespace rdma {

class MainFactory {
public:

    /**
     * Creates allocated memory that is directly registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @return  Allocated memory object that is directly registered with IBV
     */
    static BaseMemory* createMemory(MEMORY_TYPE mem_type, size_t size){
        return createMemory(mem_type, size, HUGEPAGE, Config::RDMA_NUMAREGION, -1);
    }

    /**
     * Creates allocated memory that is directly registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param huge[in]      Only for main memory if memory should be mapped
     * @param numa_node[in] Only for main memory on which numa node it should be allocated
     * @return  Allocated memory object that is directly registered with IBV
     */
    static BaseMemory* createMemory(MEMORY_TYPE mem_type, size_t size, bool huge, int numa_node){
        return createMemory(mem_type, size, HUGEPAGE, Config::RDMA_NUMAREGION, -1);
    }

    /**
     * Creates allocated memory that is directly registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param ib_numa[in]   Only for GPU to known on which numa node IB is located
     * @return  Allocated memory object that is directly registered with IBV
     */
    static BaseMemory* createMemory(MEMORY_TYPE mem_type, size_t size, int ib_numa){
        return createMemory(mem_type, size, HUGEPAGE, Config::RDMA_NUMAREGION, ib_numa);
    }

    /**
     * Creates allocated memory that is directly registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param huge[in]      Only for main memory if memory should be mapped
     * @param numa_node[in] Only for main memory on which numa node it should be allocated
     * @param ib_numa[in]   Only for GPU to known on which numa node IB is located (-1 to auto detect)
     * @return  Allocated memory object that is directly registered with IBV
     */
    static BaseMemory* createMemory(MEMORY_TYPE mem_type, size_t size, bool huge, int numa_node, int ib_numa){
        return createMemory(mem_type, size, huge, numa_node, ib_numa, true);
    }



    /**
     * Creates raw allocated memory that will not be registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @return  Raw allocated memory object
     */
    static BaseMemory* createRawMemory(MEMORY_TYPE mem_type, size_t size){
        return createRawMemory(mem_type, size, HUGEPAGE, Config::RDMA_NUMAREGION, -1);
    }

    /**
     * Creates raw allocated memory that will not be registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param huge[in]      Only for main memory if memory should be mapped
     * @param numa_node[in] Only for main memory on which numa node it should be allocated
     * @return  Raw allocated memory object
     */
    static BaseMemory* createRawMemory(MEMORY_TYPE mem_type, size_t size, bool huge, int numa_node){
        return createRawMemory(mem_type, size, HUGEPAGE, Config::RDMA_NUMAREGION, -1);
    }

    /**
     * Creates raw allocated memory that will not be registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param ib_numa[in]   Only for GPU to known on which numa node IB is located
     * @return  Raw allocated memory object
     */
    static BaseMemory* createRawMemory(MEMORY_TYPE mem_type, size_t size, int ib_numa){
        return createRawMemory(mem_type, size, HUGEPAGE, Config::RDMA_NUMAREGION, ib_numa);
    }

    /**
     * Creates raw allocated memory that will not be registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param huge[in]      Only for main memory if memory should be mapped
     * @param numa_node[in] Only for main memory on which numa node it should be allocated
     * @param ib_numa[in]   Only for GPU to known on which numa node IB is located (-1 to auto detect)
     * @return  Raw allocated memory object
     */
    static BaseMemory* createRawMemory(MEMORY_TYPE mem_type, size_t size, bool huge, int numa_node, int ib_numa){
        return createMemory(mem_type, size, huge, numa_node, ib_numa, false);
    }



    /**
     * Creates allocated memory that can be directly registered with IBV.
     * Returned object must be manually deleted
     * @param mem_type[in]  Type of memory
     * @param size[in]      How big the allocated memory should be (in bytes)
     * @param huge[in]      Only for main memory if memory should be mapped
     * @param numa_node[in] Only for main memory on which numa node it should be allocated
     * @param ib_numa[in]   Only for GPU to known on which numa node IB is located (-1 to auto detect)
     * @param register_ibv[in]  If allocated memory should be registered with IBV
     * @return  Allocated memory object that is directly registered with IBV
     */
    static BaseMemory* createMemory(MEMORY_TYPE mem_type, size_t size, bool huge, int numa_node, int ib_numa, bool register_ibv){
        switch(mem_type){
            case MEMORY_TYPE::GPU_NUMA:
            case MEMORY_TYPE::GPU_DEFAULT:
            case MEMORY_TYPE::GPU_0:
            case MEMORY_TYPE::GPU_1:
            case MEMORY_TYPE::GPU_2:
            case MEMORY_TYPE::GPU_3:
            case MEMORY_TYPE::GPU_4: 
                #ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
                    return (BaseMemory*) new CudaMemory(true, mem_size, mem_type, ib_numa);
                #endif

            case MEMORY_TYPE::MAIN: return (BaseMemory*) new MainMemory(true, size, huge, numa_node);

            default: return nullptr;
        }
    }


    /**
     * Creates a memory stub that operates on a given buffer.
     * Memory stub must be deleted manually
     * @param mem_type[in]  Memory type of the given buffer
     * @param buffer[in]    Pointer to the allocated buffer
     * @param size[in]      How big the useable space should be
     * @param freeFunc[in]  Function that gets executed on deletion (optional)
     */
    static LocalBaseMemoryStub* createMemoryStub(MEMORY_TYPE mem_type, void* buffer, size_t size, std::function<void(const void* buffer)> freeFunc=nullptr){
        return createMemoryStub(mem_type, buffer, 0, size, freeFunc);
    }

    /**
     * Creates a memory stub that operates on a given buffer.
     * Memory stub must be deleted manually
     * @param mem_type[in]  Memory type of the given buffer
     * @param buffer[in]    Pointer to the allocated buffer
     * @param offset[in]    Base offset the stub uses for all operations
     * @param size[in]      How big the useable space should be
     * @param freeFunc[in]  Function that gets executed on deletion (optional)
     */
    static LocalBaseMemoryStub* createMemoryStub(MEMORY_TYPE mem_type, void* buffer, size_t offset, size_t size, std::function<void(const void* buffer)> freeFunc=nullptr){
        switch(mem_type){
            case MEMORY_TYPE::GPU_NUMA:
            case MEMORY_TYPE::GPU_DEFAULT:
            case MEMORY_TYPE::GPU_0:
            case MEMORY_TYPE::GPU_1:
            case MEMORY_TYPE::GPU_2:
            case MEMORY_TYPE::GPU_3:
            case MEMORY_TYPE::GPU_4: 
                #ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
                    return (LocalBaseMemoryStub*) new LocalCudaMemoryStub(buffer, offset, size, (int)mem_type, freeFunc);
                #endif

            case MEMORY_TYPE::MAIN: return (LocalBaseMemoryStub*) new LocalMainMemoryStub(buffer, offset, size, freeFunc);

            default: return nullptr;
        }
    }

};

}