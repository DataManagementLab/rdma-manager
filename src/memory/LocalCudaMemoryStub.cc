#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "LocalCudaMemoryStub.h"

using namespace rdma;

// constructor
LocalCudaMemoryStub::LocalCudaMemoryStub(void* rootBuffer, size_t rootOffset, size_t mem_size, int device_index, std::function<void(const void* buffer)> freeFunc) : AbstractBaseMemory((void*)((size_t)rootBuffer+rootOffset), mem_size), AbstractCudaMemory((void*)((size_t)rootBuffer+rootOffset), mem_size, device_index), LocalBaseMemoryStub(rootBuffer, rootOffset, mem_size, freeFunc){}

LocalBaseMemoryStub *LocalCudaMemoryStub::createStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc){
    return (LocalBaseMemoryStub*) new LocalCudaMemoryStub(rootBuffer, rootOffset, mem_size, this->device_index, freeFunc);
}

#endif /* CUDA support */