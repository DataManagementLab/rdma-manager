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
     * Handles CUDA memory part.
     *
     * rootBuffer:  pointer of whole memory that contains memory part
     * rootOffset:  offset from the whole memory pointer where memory part begins
     * mem_size:  how big the memory part is (beginning from buffer+offset)
     * freeFunc:  function handle to release memory part
     *
     */
    LocalCudaMemoryStub(void* rootBuffer, size_t rootOffset, size_t mem_size, int device_index, std::function<void(const void* buffer)> freeFunc=nullptr);

    LocalBaseMemoryStub *createStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc=nullptr) override;
};

} // namespace rdma

#endif /* LocalCudaMemoryStub_H_ */
#endif /* CUDA support */