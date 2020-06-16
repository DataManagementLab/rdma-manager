#include "LocalMainMemoryStub.h"

using namespace rdma;

// constructor
LocalMainMemoryStub::LocalMainMemoryStub(void* buffer, size_t mem_size, std::function<void(const void* buffer)> freeFunc) : AbstractBaseMemory(buffer, mem_size), AbstractMainMemory(buffer, mem_size), LocalBaseMemoryStub(buffer, mem_size, freeFunc){}