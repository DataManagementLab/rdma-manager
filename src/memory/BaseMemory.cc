#include "BaseMemory.h"
#include "MainMemory.h"
#include "CudaMemory.h"

using namespace rdma;

// constructor
BaseMemory::BaseMemory(size_t mem_size){
    this->mem_size = mem_size;
}

size_t BaseMemory::getSize(){
    return this->mem_size;
}

void* BaseMemory::pointer(){
    return this->buffer;
}