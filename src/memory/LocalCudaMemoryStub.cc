#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "LocalCudaMemoryStub.h"

using namespace rdma;

// constructor
LocalCudaMemoryStub::LocalCudaMemoryStub(void* buffer, size_t mem_size, std::function<void(const void* buffer)> freeFunc) : AbstractBaseMemory(buffer, mem_size), AbstractCudaMemory(buffer, mem_size), LocalBaseMemoryStub(buffer, mem_size, freeFunc){}

#endif /* CUDA support */