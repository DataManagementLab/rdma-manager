#include "AbstractMainMemory.h"
#include <stdlib.h>
#include <string.h>

using namespace rdma;

// constructors
AbstractMainMemory::AbstractMainMemory(size_t mem_size) : AbstractBaseMemory(mem_size){}
AbstractMainMemory::AbstractMainMemory(void* buffer, size_t mem_size) : AbstractBaseMemory(buffer, mem_size){}

void AbstractMainMemory::setMemory(int value){
    setMemory(value, this->mem_size);
}

void AbstractMainMemory::setMemory(int value, size_t num){
    memset(this->buffer, value, num);
}

void AbstractMainMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void AbstractMainMemory::copyTo(void *destination, size_t num){
    memcpy(destination, this->buffer, num);
}

void AbstractMainMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void AbstractMainMemory::copyFrom(const void *source, size_t num){
    memcpy(this->buffer, source, num);
}