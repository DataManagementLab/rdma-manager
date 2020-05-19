#include "MainMemory.h"
#include <stdlib.h>
#include <string.h>

using namespace rdma;

// constructor
MainMemory::MainMemory(size_t mem_size) : BaseMemory(mem_size){
    this->buffer = malloc(mem_size);
}

// destructor
MainMemory::~MainMemory(){
    free(this->buffer);
}

void MainMemory::setMemory(int value){
    setMemory(value, this->mem_size);
}

void MainMemory::setMemory(int value, size_t num){
    memset(this->buffer, value, num);
}

void MainMemory::copyTo(void *destination){
    copyTo(destination, this->mem_size);
}

void MainMemory::copyTo(void *destination, size_t num){
    memcpy(destination, this->buffer, num);
}

void MainMemory::copyFrom(void *source){
    copyFrom(source, this->mem_size);
}

void MainMemory::copyFrom(void *source, size_t num){
    memcpy(this->buffer, source, num);
}