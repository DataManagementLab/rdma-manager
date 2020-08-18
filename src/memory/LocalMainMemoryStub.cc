#include "LocalMainMemoryStub.h"

using namespace rdma;

// constructor
LocalMainMemoryStub::LocalMainMemoryStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc) : AbstractBaseMemory((void*)((size_t)rootBuffer+rootOffset), mem_size), AbstractMainMemory((void*)((size_t)rootBuffer+rootOffset), mem_size), LocalBaseMemoryStub(rootBuffer, rootOffset, mem_size, freeFunc){}