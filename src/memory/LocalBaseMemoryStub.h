#ifndef LocalBaseMemoryStub_H_
#define LocalBaseMemoryStub_H_

#include "AbstractBaseMemory.h"

namespace rdma {

class LocalBaseMemoryStub : virtual public AbstractBaseMemory {

protected:
    void (*freeFunc)(void* buffer);

public:

    /* Constructor
     * --------------
     * Base class to handle specific memory
     *
     * rdma:  Interface that created buffer
     * buffer:  pointer to memory that should be handled
     * freeFunc:  function handle to release buffer
     *
     */
    LocalBaseMemoryStub(void* buffer, size_t mem_size, void freeFunc(void* buffer));

    /* Destructor
     * -------------
     * Releases the allocated memory
     */
    ~LocalBaseMemoryStub();
};

} // namespace rdma

#endif /* LocalBaseMemoryStub_H_ */