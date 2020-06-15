#include "LocalMainMemoryStub.h"

using namespace rdma;

// constructor
LocalMainMemoryStub::LocalMainMemoryStub(void* buffer, size_t mem_size, void freeFunc(void* buffer)) : AbstractBaseMemory(buffer, mem_size), AbstractMainMemory(buffer, mem_size), LocalBaseMemoryStub(buffer, mem_size, freeFunc){}