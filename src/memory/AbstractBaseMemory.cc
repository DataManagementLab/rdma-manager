#include "AbstractBaseMemory.h"
#include "AbstractMainMemory.h"
#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
#include "AbstractCudaMemory.h"
#endif /* CUDA support */

#include <stdio.h>

using namespace rdma;

// constructors
AbstractBaseMemory::AbstractBaseMemory(size_t mem_size){
    this->buffer = nullptr;
    this->mem_size = mem_size;
}
AbstractBaseMemory::AbstractBaseMemory(void* buffer, size_t mem_size){
    this->buffer = buffer;
    this->mem_size = mem_size;
}

size_t AbstractBaseMemory::getSize(){
    return this->mem_size;
}

void* AbstractBaseMemory::pointer(){
    return this->buffer;
}