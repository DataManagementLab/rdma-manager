#ifndef AbstractBaseMemory_H_
#define AbstractBaseMemory_H_

#include <stdio.h>
#include <cstdint>

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

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the handled memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     * offset: where to start writing the bytes
     * num:    how many bytes should be set
     *
     */
    virtual void setMemory(int value, size_t offset, size_t num) = 0;

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

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * destOffset:   offset where to start writing bytes at destination
     * srcOffset:    offset where to start reading bytes at this buffer
     * num:          how many bytes should be copied
     *
     */
    virtual void copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num) = 0;

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

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * srcOffset:    offset where to start reading bytes at source
     * destOffset:   offset where to start writing bytes at this buffer
     * num:     how many bytes should be copied
     *
     */
    virtual void copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num) = 0;

    /* Function:  getChar
     * ---------------------
     * Reads a char
     * 
     * offset: where to start reading
     * return: read char
     */
    virtual char getChar(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a char
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(char value, size_t offset) = 0;

    /* Function:  getInt16
     * ---------------------
     * Reads a 16 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    virtual int16_t getInt16(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a 16 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(int16_t value, size_t offset) = 0;

    /* Function:  getInt32
     * ---------------------
     * Reads a 32 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    virtual int32_t getInt32(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a 32 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(int32_t value, size_t offset) = 0;

    /* Function:  getInt64
     * ---------------------
     * Reads a 64 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    virtual int64_t getInt64(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a 64 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(int64_t value, size_t offset) = 0;

};

} // namespace rdma

#endif /* AbstractBaseMemory_H_ */