#ifndef LocalMainMemoryStub_H_
#define LocalMainMemoryStub_H_

#include "LocalBaseMemoryStub.h"
#include "AbstractMainMemory.h"

namespace rdma {
    
class LocalMainMemoryStub : virtual public AbstractMainMemory, virtual public LocalBaseMemoryStub {

public:

    /* Constructor
     * --------------
     * Handles main memory.
     *
     * buffer:  pointer to main memory that should be handled
     * mem_size:  size of main memory that should be handled
     * freeFunc:  function handle to release buffer
     *
     */
    LocalMainMemoryStub(void* buffer, size_t mem_size, void freeFunc(void* buffer));


    // destructor
    ~LocalMainMemoryStub();

};

} // namespace rdma

#endif /* LocalMainMemoryStub_H_ */