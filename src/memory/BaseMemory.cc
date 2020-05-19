#include "BaseMemory.h"
#include "MainMemory.h"
#include "CudaMemory.h"

using namespace rdma;

// constructor
template<typename P>
BaseMemory<P>::BaseMemory(size_t mem_size){
    this->mem_size = mem_size;
}

template<typename P>
size_t BaseMemory<P>::getSize(){
    return this->mem_size;
}

template<typename P>
size_t BaseMemory<P>::getSizeInBytes(){
    return this->mem_size * sizeof(P);
}

template<typename P>
P* BaseMemory<P>::ptr(){
    return this->buf;
}