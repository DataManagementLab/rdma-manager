#include "./Memory.h"

#pragma once

#ifndef HUGEPAGE
#define HUGEPAGE false
#endif

namespace rdma {

class MemoryFactory {
    public:

        static Memory* createMemory(size_t memSize, MEMORY_TYPE memType){
            return (int)memType <= (int)MEMORY_TYPE::MAIN ? 
                    new Memory(memSize, HUGEPAGE) : // main memory
                    new Memory(memSize, memType) ; // gpu memory
        }


        /* Function createMainMemory
         * --------------
         * Allocates main memory that can be used for RDMA.
         * Uses therefore malloc() or mmap()
         *
         * memSize:  size how much memory should be allocated
         */
        static Memory* createMainMemory(size_t memSize){
            return new Memory(memSize);
        }

        /* Function createMainMemory
         * --------------
         * Allocates main memory that can be used for RDMA.
         * Uses therefore malloc() or mmap()
         *
         * memSize:  size how much memory should be allocated
         * huge:      If true then a huge memory block can be 
         *            allocated (uses therefore mmap())
         */
        static Memory* createMainMemory(size_t memSize, bool huge){
            return new Memory(memSize, huge);
        }

        /* Function createMainMemory
         * --------------
         * Allocates main memory that can be used for RDMA.
         * Uses therefore malloc() or mmap()
         *
         * memSize:  size how much memory should be allocated
         * numaNode: Index of the NUMA node where the memory
         *            should be allocated on (LINUX only)
         */
        static Memory* createMainMemory(size_t memSize, int numaNode){
            return new Memory(memSize, HUGEPAGE, numaNode);
        }

        /* Function createMainMemory
         * --------------
         * Allocates main memory that can be used for RDMA.
         * Uses therefore malloc() or mmap()
         *
         * memSize:  size how much memory should be allocated
         * huge:      If true then a huge memory block can be 
         *            allocated (uses therefore mmap())
         * numaNode: Index of the NUMA node where the memory
         *            should be allocated on (LINUX only)
         */
        static Memory* createMainMemory(size_t memSize, bool huge, int numaNode){
            return new Memory(memSize, huge, numaNode);
        }

        /* Function createMainMemory
         * --------------
         * Allocates main memory that can be used for RDMA.
         * Uses therefore malloc() or mmap()
         *
         * memSize:  size how much memory should be allocated
         * huge:      If true then a huge memory block can be 
         *            allocated (uses therefore mmap())
         * numaNode: Index of the NUMA node where the memory
         *            should be allocated on (LINUX only)
         * registerIbv:  If memory should be registered with IBV (true in order to use RDMA)
         * ibPort:  InfiniBand port to use
         */
        static Memory* createMainMemory(size_t memSize, bool huge, int numaNode, bool registerIbv, int ibPort=Config::RDMA_IBPORT){
            return new Memory(registerIbv, memSize, huge, numaNode, ibPort);
        }




        /* Function createCudaMemory
         * --------------
         * Allocates CUDA (GPU) memory  that can be used for RDMA / GPUDIRECT
         *
         * memSize:      size how much memory should be allocated
         *
         */
        static Memory* createCudaMemory(size_t memSize){
            return new Memory(memSize, MEMORY_TYPE::GPU_DEFAULT);
        }

        /* Function createCudaMemory
         * --------------
         * Allocates CUDA (GPU) memory  that can be used for RDMA / GPUDIRECT
         *
         * memSize:      size how much memory should be allocated
         * memType:      enum instead of device index. Type MAIN is not allowed!
         *
         */
        static Memory* createCudaMemory(size_t memSize, MEMORY_TYPE memType){
            return new Memory(memSize, memType);
        }

        /* Function createCudaMemory
         * --------------
         * Allocates CUDA (GPU) memory  that can be used for RDMA / GPUDIRECT
         *
         * memSize:      size how much memory should be allocated
         * deviceIndex:  index of the GPU device that should be used to
         *                allocate the memory. If -1 the currently 
         *                selected device will be used. If -2 a device 
         *                will be selected based on the preferred NUMA region
         *
         */
        static Memory* createCudaMemory(size_t memSize, int deviceIndex){
            return new Memory(memSize, deviceIndex);
        }

        /* Function createCudaMemory
         * --------------
         * Allocates CUDA (GPU) memory  that can be used for RDMA / GPUDIRECT
         *
         * memSize:      size how much memory should be allocated
         * memType:      enum instead of device index. Type MAIN is not allowed!
         * ibNuma:       Defines on which NUMA region the memory should be 
         *                registered for IBV. (-1 to automatically detect NUMA region)
         *
         */
        static Memory* createCudaMemory(size_t memSize, MEMORY_TYPE memType, int ibNuma){
            return new Memory(memSize, memType, ibNuma);
        }

        /* Function createCudaMemory
         * --------------
         * Allocates CUDA (GPU) memory  that can be used for RDMA / GPUDIRECT
         *
         * memSize:      size how much memory should be allocated
         * deviceIndex:  index of the GPU device that should be used to
         *                allocate the memory. If -1 the currently 
         *                selected device will be used. If -2 a device 
         *                will be selected based on the preferred NUMA region
         * registerIbv:  If memory should be registered with IBV (true in order to use RDMA)
         * ibNuma:       Defines on which NUMA region the memory should be 
         *                registered for IBV. (-1 to automatically detect NUMA region)
         *
         */
        static Memory* createCudaMemory(size_t memSize, int deviceIndex, bool registerIbv, int ibNuma){
            return new Memory(registerIbv, memSize, deviceIndex, ibNuma);
        }
};

}