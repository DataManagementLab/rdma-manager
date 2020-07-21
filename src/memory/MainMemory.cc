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
MainMemory::MainMemory(size_t mem_size, bool huge, int numa_node) : AbstractBaseMemory(mem_size), AbstractMainMemory(mem_size), BaseMemory(mem_size){
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
        } // TODO else {
        numa_free(this->buffer, this->mem_size);
        // TODO }
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

LocalBaseMemoryStub *MainMemory::malloc(size_t size){
    size_t rootOffset = (size_t)alloc(size) - (size_t)this->buffer;
    return (LocalBaseMemoryStub*) new LocalMainMemoryStub(this->buffer, rootOffset, size, [this](const void* ptr){
      free(ptr);
    });
}