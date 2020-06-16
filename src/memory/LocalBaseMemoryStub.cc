#include "LocalBaseMemoryStub.h"
#include "LocalMainMemoryStub.h"
#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
#include "LocalCudaMemoryStub.h"
#endif /* CUDA support */

using namespace rdma;

// constructor
LocalBaseMemoryStub::LocalBaseMemoryStub(void* buffer, size_t mem_size, std::function<void(const void* buffer)> freeFunc) : AbstractBaseMemory(buffer, mem_size){
    this->freeFunc = freeFunc;
}

LocalBaseMemoryStub::~LocalBaseMemoryStub(){
    this->freeFunc(this->buffer);
}