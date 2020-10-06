#ifndef AbstractBaseMemory_H_
#define AbstractBaseMemory_H_

#include <stdio.h>
#include <cstdint>
#include <string>
#include <stdexcept>

#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
#include <cuda_runtime_api.h>
#endif

namespace rdma {


enum MEMORY_TYPE { MAIN=-3, GPU_NUMA=-2, GPU_DEFAULT=-1, GPU_0=0, GPU_1=1, GPU_2=2, GPU_3=3, GPU_4=4};


class AbstractBaseMemory {

protected:

#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
    /* Function:  checkCudaError
     * ---------------------
     * Handles CUDA errors
     *
     * code:  CUDA error code that should be handled
     * msg:   Message that should be printed if code is an error code
     *
     */
    static void checkCudaError(cudaError_t code, const char* msg){
        if(code != cudaSuccess){
            //fprintf(stderr, "CUDA-Error(%i): %s", code, msg);
            throw std::runtime_error("CUDA-Error("+std::to_string(code)+") has occurred: "+msg);
        }
    }
#endif



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

    /* Function:  setSize
     * ---------------------
     * Sets the size of the memory this 
     * object handles
     * 
     * mem_size:  size of the handled memory
     */
    void setSize(size_t mem_size);

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

    /* Function:  isMainMemory
     * --------------------
     * Returns true if the memory is allocated in main memory
     */ 
    virtual bool isMainMemory() {
        return false;
    }

    /* Function:  isGPUMemory
     * --------------------
     * Returns true if the memory is allocated on a GPU
     */ 
    virtual bool isGPUMemory() {
        return false;
    }

    /* Function: openContext
     * ---------------------
     * Must be called before using the raw pointer if it 
     * is not main memory!
     * Can be called if many calls of the 
     * set(), get() or copy() methods follow to boost 
     * performance but not necessary.
     * 
     * Multiple openContext() can be called as long as 
     * same amount of closeContext() are called again.
     */
    virtual void openContext() = 0;

    /* Function: closeContext
     * ---------------------
     * Closes the opened context again such that 
     * another memory operating on the same device 
     * can use it again
     */
    virtual void closeContext() = 0;

    /* Function: toString
     * ----------------------
     * Returns all bytes of this memory as string
     * 
     * return:  byte values as string
     */
    virtual std::string toString();

    /* Function: toString
     * ----------------------
     * Returns the byte values of a certain section 
     * of this memory as string
     * 
     * offset:  offset from where to start reading bytes
     * length:  how many bytes should be returned
     * 
     * return:  byte values as string
     */
    virtual std::string toString(size_t offset, size_t length);

    /* Function: print
     * ----------------------
     * Prints all bytes of this memory
     * 
     */
    virtual void print();

    /* Function: print
     * ----------------------
     * Prints the byte values of a certain section of this memory
     * 
     * offset:  offset from where to start printing bytes
     * length:  how many bytes should be printed
     * 
     */
    virtual void print(size_t offset, size_t length);

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
     * memtype:      memory type of the destination (default to main memory, which GPU doesn't matter)
     */
    virtual void copyTo(void *destination, MEMORY_TYPE memtype=MEMORY_TYPE::MAIN) = 0;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * num:          how many bytes should be copied
     * memtype:      memory type of the destination (default to main memory, which GPU doesn't matter)
     */
    virtual void copyTo(void *destination, size_t num, MEMORY_TYPE memtype=MEMORY_TYPE::MAIN) = 0;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * destOffset:   offset where to start writing bytes at destination
     * srcOffset:    offset where to start reading bytes at this buffer
     * num:          how many bytes should be copied
     * memtype:      memory type of the destination (default to main memory, which GPU doesn't matter)
     */
    virtual void copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num, MEMORY_TYPE memtype=MEMORY_TYPE::MAIN) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * memtype: memory type of the source (default from main memory, which GPU doesn't matter)
     */
    virtual void copyFrom(const void *source, MEMORY_TYPE memtype=MEMORY_TYPE::MAIN) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * num:     how many bytes should be copied
     * memtype: memory type of the source (default from main memory, which GPU doesn't matter)
     */
    virtual void copyFrom(const void *source, size_t num, MEMORY_TYPE memtype=MEMORY_TYPE::MAIN) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * srcOffset:    offset where to start reading bytes at source
     * destOffset:   offset where to start writing bytes at this buffer
     * num:     how many bytes should be copied
     * memtype: memory type of the source (default from main memory, which GPU doesn't matter)
     */
    virtual void copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num, MEMORY_TYPE memtype=MEMORY_TYPE::MAIN) = 0;


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

    /* Function:  getUInt16
     * ---------------------
     * Reads a 16 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    virtual uint16_t getUInt16(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a 16 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(uint16_t value, size_t offset) = 0;

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

    /* Function:  getUInt32
     * ---------------------
     * Reads a 32 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    virtual uint32_t getUInt32(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a 32 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(uint32_t value, size_t offset) = 0;

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

    /* Function:  getUInt64
     * ---------------------
     * Reads a 64 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    virtual uint64_t getUInt64(size_t offset) = 0;

    /* Function:  set
     * ---------------------
     * Writes a 64 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    virtual void set(uint64_t value, size_t offset) = 0;
};

} // namespace rdma

#endif /* AbstractBaseMemory_H_ */