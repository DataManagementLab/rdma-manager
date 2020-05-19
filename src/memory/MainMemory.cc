#include "MainMemory.h"
#include <stdlib.h>
#include <string.h>

using namespace rdma;

// constructor
MainMemory::MainMemory(size_t mem_size) : BaseMemory<void>(mem_size){
    this->buf = malloc(mem_size);
}

// destructor
MainMemory::~MainMemory(){
    free(this->buf);
}

void MainMemory::setMemory(int value){
    setMemory(value, this->mem_size);
}

void MainMemory::setMemory(int value, size_t num){
    memset(this->buf, value, num);
}

void MainMemory::copyTo(void *destination){
    copyTo(destination, this->mem_size);
}

void MainMemory::copyTo(void *destination, size_t num){
    memcpy(destination, this->buf, num);
}

void MainMemory::copyFrom(void *source){
    copyFrom(source, this->mem_size);
}

void MainMemory::copyFrom(void *source, size_t num){
    memcpy(this->buf, source, num);
}