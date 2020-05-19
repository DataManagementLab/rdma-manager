#ifndef BaseMemory_H_
#define BaseMemory_H_

#include <stdio.h>

namespace rdma {

template<typename P>
class BaseMemory {

protected:
    size_t mem_size;
    P *buf = NULL;

public:

    /* Constructor
     * --------------
     * Base class to allocate specific memory
     *
     * mem_size:  size how much memory should be allocated
     *
     */
    BaseMemory(size_t mem_size);

    /* Destructor
     * -------------
     * Releases the allocated memory
     */
    virtual ~BaseMemory();

    /* Function:  getSize
     * ---------------------
     * Returns the size of the allocated memory
     * 
     * return:  size of allocated memory
     */
    size_t getSize();

    /* Function:  getSizeInBytes
     * ---------------------
     * Returns the size in bytes of the allocated memory
     * 
     * return:  size in bytes of allocated memory
     */
    size_t getSizeInBytes();

    /* Function:  ptr
     * ---------------------
     * Returns the pointer of the allocated memory.
     * What type of memory and how to handle the pointer is
     * implementation dependent 
     *
     * return:  pointer of the allocated memory
     */
    P* ptr();

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the allocated memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     *
     */
    virtual void setMemory(int value);

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the allocated memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     * num:    how many bytes should be set
     *
     */
    virtual void setMemory(int value, size_t num);

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the allocated memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the allocated memory should be copied to
     *
     */
    virtual void copyTo(void *destination);

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the allocated memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the allocated memory should be copied to
     * num:          how many bytes should be copied
     *
     */
    virtual void copyTo(void *destination, size_t num);

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the allocated memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     *
     */
    virtual void copyFrom(void *source);

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the allocated memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * num:     how many bytes should be copied
     *
     */
    virtual void copyFrom(void *source, size_t num);

};

} // namespace rdma

#endif /* BaseMemory_H_ */