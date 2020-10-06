#ifndef LocalBaseMemoryStub_H_
#define LocalBaseMemoryStub_H_

#include "AbstractBaseMemory.h"
#include <functional>

namespace rdma {

class LocalBaseMemoryStub : virtual public AbstractBaseMemory {

protected:
    void* rootBuffer;
    size_t rootOffset;
    std::function<void(void* buffer)> freeFunc;

public:

    /* Constructor
     * --------------
     * Base class to handle specific memory part
     *
     * rootBuffer:  pointer of whole memory that contains memory part
     * rootOffset:  offset from the whole memory pointer where memory part begins
     * mem_size:  how big the memory part is (beginning from buffer+offset)
     * freeFunc:  function handle to release memory part
     * 
     */
    LocalBaseMemoryStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc);

    /* Destructor
     * -------------
     * Releases the allocated memory part
     */
    virtual ~LocalBaseMemoryStub();

    /* Function:  getRootPointer
     * -------------
     * Returns the pointer of the whole memory buffer 
     * where this memory part is allocated in
     * 
     * return:  pointer of whole memory buffer
     */
    void* getRootPointer();

    /* Function:  getRootPointer
     * -------------
     * Returns the pointer of the whole memory buffer 
     * where this memory part is allocated in
     * 
     * offset:  offset that will be added to pointer
     * return:  (pointer+offset) of whole memory buffer
     */
    void* getRootPointer(const size_t &offset);

    /* Function:  getRootOffset
     * -------------
     * Returns the offset of this memory part 
     * inside of the whole memory buffer
     * 
     * return:  offset inside whole memory buffer
     */
    size_t getRootOffset();

    /* Function: createStub
     * ---------------
     * Creates an instance of a memory stub that 
     * can handle the same memory type as this object
     * rootBuffer:  pointer of whole memory that contains memory part
     * rootOffset:  offset from the whole memory pointer where memory part begins
     * mem_size:  how big the memory part is (beginning from buffer+offset)
     * freeFunc:  function handle to release memory part
     */
    virtual LocalBaseMemoryStub *createStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc) = 0;
};

} // namespace rdma

#endif /* LocalBaseMemoryStub_H_ */