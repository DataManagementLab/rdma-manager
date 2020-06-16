#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef CudaMemory_H_
#define CudaMemory_H_

#include "BaseMemory.h"
#include "AbstractCudaMemory.h"

namespace rdma {
    
class CudaMemory : virtual public AbstractCudaMemory, virtual public BaseMemory {

public:
    
    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory on the currently selected GPU
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
     *                allocate the memory. If negative the currently 
     *                selected device will be used
     *
     */
    CudaMemory(size_t mem_size, int device_index);

    // destructor
    virtual ~CudaMemory();

};

} // namespace rdma

#endif /* CudaMemory_H_ */
#endif /* CUDA support */