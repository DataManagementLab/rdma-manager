#ifndef LocalBaseMemoryStub_H_
#define LocalBaseMemoryStub_H_

#include "AbstractBaseMemory.h"
#include <functional>

namespace rdma {

class LocalBaseMemoryStub : virtual public AbstractBaseMemory {

protected:
    std::function<void(const void* buffer)> freeFunc;

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
    LocalBaseMemoryStub(void* buffer, size_t mem_size, std::function<void(const void* buffer)> freeFunc);

    /* Destructor
     * -------------
     * Releases the allocated memory
     */
    ~LocalBaseMemoryStub();
};

} // namespace rdma

#endif /* LocalBaseMemoryStub_H_ */