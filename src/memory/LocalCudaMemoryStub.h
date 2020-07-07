#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef LocalCudaMemoryStub_H_
#define LocalCudaMemoryStub_H_

#include "LocalBaseMemoryStub.h"
#include "AbstractCudaMemory.h"

namespace rdma {
    
class LocalCudaMemoryStub : virtual public AbstractCudaMemory, virtual public LocalBaseMemoryStub {

public:

    /* Constructor
     * --------------
     * Handles CUDA memory.
     *
     * buffer:  pointer to CUDA memory that should be handled
     * mem_size:  size of CUDA memory that should be handled
     * device_index: index of CUDA device
     * freeFunc:  function handle to release buffer
     *
     */
    LocalCudaMemoryStub(void* buffer, size_t mem_size, int device_index, std::function<void(const void* buffer)> freeFunc);

};

} // namespace rdma

#endif /* LocalCudaMemoryStub_H_ */
#endif /* CUDA support */