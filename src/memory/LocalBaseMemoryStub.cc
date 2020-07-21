#include "LocalBaseMemoryStub.h"
#include "LocalMainMemoryStub.h"
#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
#include "LocalCudaMemoryStub.h"
#endif /* CUDA support */

using namespace rdma;

// constructor
LocalBaseMemoryStub::LocalBaseMemoryStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc) : AbstractBaseMemory((void*)((size_t)rootBuffer+rootOffset), mem_size){
    this->rootBuffer = rootBuffer;
    this->rootOffset = rootOffset;
    this->freeFunc = freeFunc;
}

LocalBaseMemoryStub::~LocalBaseMemoryStub(){
    this->freeFunc(this->buffer);
}


void* LocalBaseMemoryStub::getRootPointer(){
    return this->rootBuffer;
}

void* LocalBaseMemoryStub::getRootPointer(const size_t &offset){
    return (void*)((size_t)this->rootBuffer + offset);
}

size_t LocalBaseMemoryStub::getRootOffset(){
    return this->rootOffset;
}