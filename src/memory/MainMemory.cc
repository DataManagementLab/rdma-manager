#include "MainMemory.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

using namespace rdma;

#ifndef HUGEPAGE
#define HUGEPAGE false
#endif

// constructors
MainMemory::MainMemory(size_t mem_size) : MainMemory(mem_size, HUGEPAGE){}
MainMemory::MainMemory(size_t mem_size, bool huge) : MainMemory(mem_size, huge, Config::RDMA_NUMAREGION){}
MainMemory::MainMemory(size_t mem_size, int numa_node) : MainMemory(mem_size, HUGEPAGE, numa_node){}
MainMemory::MainMemory(size_t mem_size, bool huge, int numa_node) : BaseMemory(mem_size){
    this->huge = huge;
    this->numa_node = numa_node;
    // allocate memory
    #ifdef LINUX
        if(huge){
            this->buffer = mmap(NULL, this->mem_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            madvise(this->buffer, this->mem_size, MADV_HUGEPAGE);
            numa_tonode_memory(this->buffer, this->mem_size, this->numa_node);
        } else {
            this->buffer = numa_alloc_onnode(this->mem_size, this->numa_node);
        }
    #else
        this->buffer = malloc(this->mem_size);
    #endif

    if (this->buffer == 0) {
        throw runtime_error("Cannot allocate memory! Requested size: " + to_string(this->mem_size));
    }

    memset(this->buffer, 0, this->mem_size);

    this->init();
}

// destructor
MainMemory::~MainMemory(){
    // release memory
    #ifdef LINUX
        if(this->huge){
            munmap(this->buffer, this->mem_size);
        }
        numa_free(this->buffer, this->mem_size);
    #else
        free(this->buffer);
    #endif
    this->buffer = nullptr;
}

bool MainMemory::isHuge(){
    return this->huge;
}

int MainMemory::getNumaNode(){
    return this->numa_node;
}

void MainMemory::setMemory(int value){
    setMemory(value, this->mem_size);
}

void MainMemory::setMemory(int value, size_t num){
    memset(this->buffer, value, num);
}

void MainMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void MainMemory::copyTo(void *destination, size_t num){
    memcpy(destination, this->buffer, num);
}

void MainMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void MainMemory::copyFrom(const void *source, size_t num){
    memcpy(this->buffer, source, num);
}