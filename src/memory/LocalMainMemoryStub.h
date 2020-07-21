#ifndef LocalMainMemoryStub_H_
#define LocalMainMemoryStub_H_

#include "LocalBaseMemoryStub.h"
#include "AbstractMainMemory.h"

namespace rdma {
    
class LocalMainMemoryStub : virtual public AbstractMainMemory, virtual public LocalBaseMemoryStub {

public:

    /* Constructor
     * --------------
     * Handles main memory part.
     *
     * rootBuffer:  pointer of whole memory that contains memory part
     * rootOffset:  offset from the whole memory pointer where memory part begins
     * mem_size:  how big the memory part is (beginning from buffer+offset)
     * freeFunc:  function handle to release memory part
     *
     */
    LocalMainMemoryStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc);

};

} // namespace rdma

#endif /* LocalMainMemoryStub_H_ */