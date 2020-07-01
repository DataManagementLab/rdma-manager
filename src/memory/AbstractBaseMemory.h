#ifndef AbstractBaseMemory_H_
#define AbstractBaseMemory_H_

#include <stdio.h>

namespace rdma {

class AbstractBaseMemory {

protected:
    size_t mem_size;
    void *buffer = NULL;

    /* Constructor
     * --------------
     * Base class to handle specific memory
     *
     * mem_size:  size how much memory should be allocated
     *
     */
    AbstractBaseMemory(size_t mem_size);

public:

    /* Constructor
     * --------------
     * Base class to handle specific memory
     *
     * mem_size:  size how much memory should be allocated
     *
     */
    AbstractBaseMemory(void* buffer, size_t mem_size);

    /* Function:  getSize
     * ---------------------
     * Returns the size of the handled memory
     * 
     * return:  size of handled memory
     */
    size_t getSize();

    /* Function:  pointer
     * ---------------------
     * Returns the pointer of the handled memory.
     * What type of memory and how to handle the pointer is
     * implementation dependent 
     *
     * return:  pointer of the handled memory
     */
    void* pointer();

    /* Function:  pointer
     * ---------------------
     * Returns the pointer of the handled memory with a given offset added.
     * What type of memory and how to handle the pointer is
     * implementation dependent 
     *
     * offset:  offset that should be added onto the pointer in bytes
     * return:  pointer of the handled memory
     */
    void* pointer(const size_t offset){
        return ((char*)buffer + offset);
    }

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the handled memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     *
     */
    virtual void setMemory(int value) = 0;

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the handled memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     * num:    how many bytes should be set
     *
     */
    virtual void setMemory(int value, size_t num) = 0;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     *
     */
    virtual void copyTo(void *destination) = 0;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * num:          how many bytes should be copied
     *
     */
    virtual void copyTo(void *destination, size_t num) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     *
     */
    virtual void copyFrom(const void *source) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * num:     how many bytes should be copied
     *
     */
    virtual void copyFrom(const void *source, size_t num) = 0;

};

} // namespace rdma

#endif /* AbstractBaseMemory_H_ */