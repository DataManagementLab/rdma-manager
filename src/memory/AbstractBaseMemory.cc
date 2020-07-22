#include "AbstractBaseMemory.h"
#include "AbstractMainMemory.h"
#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
#include "AbstractCudaMemory.h"
#endif /* CUDA support */

#include <stdio.h>
#include <iostream>
#include <sstream>

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

std::string AbstractBaseMemory::toString(){
    return toString(0, this->mem_size);
}

std::string AbstractBaseMemory::toString(size_t offset, size_t length){
    std::ostringstream oss;
    oss << "[";
    bool next = false;
    for(size_t i=offset; i < length; i++){
        if(next){ oss << ", "; } else { next = true; }
        oss << ((int)getChar(i));
    }
    oss << "]";
    return oss.str();
}

void AbstractBaseMemory::print(){
    std::cout << toString() << std::endl;
}

void AbstractBaseMemory::print(size_t offset, size_t length){
    std::cout << toString(offset, length) << std::endl;
}
