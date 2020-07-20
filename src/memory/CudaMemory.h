#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef CudaMemory_H_
#define CudaMemory_H_

#include "AbstractCudaMemory.h"
#include "LocalCudaMemoryStub.h"
#include "BaseMemory.h"

namespace rdma {
    
class CudaMemory : virtual public AbstractCudaMemory, virtual public BaseMemory {

public:
    
    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory based of the preferred NUMA region
     *
     * mem_size:      size how much memory should be allocated
     *
     */
    CudaMemory(size_t mem_size);

    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * mem_size:      size how much memory should be allocated
     * device_index:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected base on the NUMA region
     *
     */
    CudaMemory(size_t mem_size, int device_index);

    // destructor
    virtual ~CudaMemory();

    LocalBaseMemoryStub *malloc(size_t size) override;
};

} // namespace rdma

#endif /* CudaMemory_H_ */
#endif /* CUDA support */