#include "BaseMemory.h"
#include "MainMemory.h"
#include "CudaMemory.h"
#include <stdio.h>

using namespace rdma;

// constructor
BaseMemory::BaseMemory(size_t mem_size){
    this->mem_size = mem_size;
}

BaseMemory::~BaseMemory() = default;

size_t BaseMemory::getSize(){
    return this->mem_size;
}

void* BaseMemory::pointer(){
    return this->buffer;
}


// TODO THIS MUST BE REMOVED
// ---[ Added because linking causes errors ]---------------
void BaseMemory::setMemory(int value){
    fprintf(stderr, "BaseMemory::setMemory(%i) is an abstract function\n", value);
}

void BaseMemory::setMemory(int value, size_t num){
    fprintf(stderr, "BaseMemory::setMemory(%i, %zu) is an abstract function\n", value, num);
}

void BaseMemory::copyTo(void *destination){
    fprintf(stderr, "BaseMemory::copyTo(%p) is an abstract function\n", destination);
}

void BaseMemory::copyTo(void *destination, size_t num){
    fprintf(stderr, "BaseMemory::copyTo(%p, %zu) is an abstract function\n", destination, num);
}

void BaseMemory::copyFrom(void *source){
    fprintf(stderr, "BaseMemory::copyFrom(%p) is an abstract function\n", source);
}

void BaseMemory::copyFrom(void *source, size_t num){
    fprintf(stderr, "BaseMemory::copyFrom(%p, %zu) is an abstract function\n", source, num);
}
// ---------------------------------------------------------