#include "MainMemory.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

using namespace rdma;

// constructors
MainMemory::MainMemory(size_t mem_size) : MainMemory(mem_size, false){}
MainMemory::MainMemory(size_t mem_size, bool huge) : BaseMemory(mem_size){
    this->huge = huge;
    if(huge){
        this->buffer = mmap(NULL, mem_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        madvise(this->buffer, mem_size, MADV_HUGEPAGE);
    } else {
        this->buffer = malloc(mem_size);
    }
    #ifdef LINUX
    numa_tonode_memory(this->buffer, this->mem_size, Config::RDMA_NUMAREGION);
    #endif
    memset(this->buffer, 0, this->mem_size);
    if (this->buffer == 0) {
        throw runtime_error("Cannot allocate memory! Requested size: " + to_string(this->mem_size));
    }

    this->init();
}

// destructor
MainMemory::~MainMemory(){
    if(this->huge){
        munmap(this->buffer, this->mem_size);
    } else {
        free(this->buffer);
    }
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