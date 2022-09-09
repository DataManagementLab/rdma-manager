#include "../utils/Config.h"

#include <condition_variable>
#include <infiniband/verbs.h>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#ifndef NO_CUDA
#include <cuda_runtime_api.h>
#endif

#pragma once

namespace rdma {


enum MEMORY_TYPE {
    /** main memory / RAM */
    MAIN=-3,

    /** selects the GPU closest to the current CPU core */
    GPU_NUMA=-2,

    /** selects the default GPU defined by CUDA */
    GPU_DEFAULT=-1,

    /** select GPU with index 0 */
    GPU_0=0,

    /** select GPU with index 1 */
    GPU_1=1,

    /** select GPU with index 2 */
    GPU_2=2,

    /** select GPU with index 3 */
    GPU_3=3,

    /** select GPU with index 4 */
    GPU_4=4
};


/* Abstraction for slicing large RDMA buffer into smaller virtual chunks (rdma_mem_t) */
struct rdma_mem_t {
  size_t size; // size of memory region
  bool free;
  size_t offset;
  bool isnull;

  rdma_mem_t(size_t initSize, bool initFree, size_t initOffset)
      : size(initSize), free(initFree), offset(initOffset), isnull(false) {}

  rdma_mem_t() : size(0), free(false), offset(0), isnull(true) {}
};



class Memory {
protected:
    size_t memSize;
    void *buffer = nullptr;
    const bool mainMem;
    int numaNode;

    // sub memory management
    Memory* parent;

    // main memory only
    const bool huge;

    // gpu memory only
    int deviceIndex, previousDeviceIndex, openContextCounter = 0;

    // RDMA
    bool m_ibv;
    struct ibv_pd *pd; // ProtectionDomain handle
    struct ibv_mr *mr; // MemoryRegistration handle for buffer

    // Device attributes
    int ib_port;
    struct ibv_port_attr port_attr; // IB port attributes
    struct ibv_context *ib_ctx;     // device handle

    // Memory management
    std::list<rdma_mem_t> m_rdmaMem;
    std::unordered_map<size_t, rdma_mem_t> m_usedRdmaMem; // <offset, memory-segment>
    static rdma_mem_t s_nillmem;

    // Thread safe alloc/free
    std::recursive_mutex m_lockMem;

    void preInit();
    void postInit();
    void mergeFreeMem(std::list<rdma_mem_t>::iterator &iter);


    #ifndef NO_CUDA
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


    /*
     * Used by alloc() method to create child memory on top of parent memory
     */
    Memory(Memory *parent, size_t offset, size_t memSize);


public:

    /* Constructor for MAIN memory
     * --------------
     * Allocates main memory that can be used for RDMA.
     * Uses therefore malloc() or mmap()
     *
     * memSize:  size of buffer
     *
     */
    Memory(size_t memSize);

    /* Constructor for MAIN memory
     * --------------
     * Allocates main memory that can be used for RDMA.
     * Uses therefore malloc() or mmap()
     *
     * memSize:  size of buffer
     * huge:    If true then a huge memory block can be 
     *            allocated (uses therefore mmap())
     *
     */
    Memory(size_t memSize, bool huge);

    /* Constructor for MAIN memory
     * --------------
     * Allocates main memory that can be used for RDMA.
     * Uses therefore malloc() or mmap()
     *
     * memSize:  size of buffer
     * huge:    If true then a huge memory block can be 
     *            allocated (uses therefore mmap())
     * numaNode: Index of the NUMA node where the memory
     *            should be allocated on (LINUX only)
     *
     */
    Memory(size_t memSize, bool huge, int numaNode);

    /* Constructor for MAIN memory
     * --------------
     * Allocates main memory that can be used for RDMA.
     * Uses therefore malloc() or mmap()
     *
     * registerIbv:  If memory should be registered with IBV (true in order to use RDMA)
     * memSize:  size how much memory should be allocated
     * huge:      If true then a huge memory block can be 
     *            allocated (uses therefore mmap())
     * numaNode: Index of the NUMA node where the memory
     *            should be allocated on (LINUX only)
     * ibPort:  InfiniBand port to use
     */
    Memory(bool registerIbv, size_t memSize, bool huge, int numaNode, int ibPort=Config::RDMA_IBPORT);



     /* Constructor for GPU memory
     * --------------
     * Allocates CUDA (GPU) memory that can be used for RDMA / GPUDIRECT
     *
     * memSize:      size how much memory should be allocated
     * memoryType:   enum instead of device index. Type MAIN is not allowed!
     *
     */
    Memory(size_t memSize, MEMORY_TYPE memoryType);

    /* Constructor for GPU memory
     * --------------
     * Allocates CUDA (GPU) memory that can be used for RDMA / GPUDIRECT
     *
     * memSize:      size how much memory should be allocated
     * deviceIndex:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected based on the preferred NUMA region
     *
     */
    Memory(size_t memSize, int deviceIndex);

    /* Constructor for GPU memory
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * memSize:      size how much memory should be allocated
     * memoryType:   enum instead of device index. Type MAIN is not allowed!
     * ibNuma:       Defines on which NUMA region the memory should be 
     *                registered for IBV. (-1 to automatically detect NUMA region)
     *
     */
    Memory(size_t memSize, MEMORY_TYPE memoryType, int ibNuma);

    /* Constructor for GPU memory
     * --------------
     * Allocates CUDA (GPU) memory that can be used for RDMA / GPUDIRECT.
     *
     * memSize:      size how much memory should be allocated
     * deviceIndex:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected based on the preferred NUMA region
     * ibNuma:       Defines on which NUMA region the memory should be 
     *                registered for IBV. (-1 to automatically detect NUMA region)
     *
     */
    Memory(size_t memSize, int deviceIndex, int ibNuma);

    /* Constructor for GPU memory
     * --------------
     * Allocates CUDA (GPU) memory that can be used for RDMA / GPUDIRECT
     *
     * registerIbv:  If memory should be registered with IBV (true in order to use RDMA)
     * memSize:      size how much memory should be allocated
     * deviceIndex:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected based on the preferred NUMA region
     * ibNuma:       Defines on which NUMA region the memory should be 
     *                registered for IBV. (-1 to automatically detect NUMA region)
     *
     */
    Memory(bool registerIbv, size_t memSize, int deviceIndex, int ibNuma);


    ~Memory();


    /* Function:  isChild
     * ---------------------
     * Returns if this memory object has no own allocated memory 
     * but instead borrows a reserved portion of a parent memory object.
     * 
     * return:  true if child of a parent memory object
     */
    bool isChild();


    /* Function:  isRootParent
     * ---------------------
     * Returns if this memory object has allocated physical memory
     * that is responsible for deallocating it again.
     * 
     * return: true if not child of a parent memory object
     */
    bool isRoot();

    /* Function:  getRootOffset
     * ---------------------
     * Returns the offset of the (child) memory compared
     * to the underlying physical buffer.
     * If this memory is already the root then zero will be returned.
     * 
     * return: offset to underlying physical buffer of this abstract memory object
     */
    size_t getRootOffset();

    /* Function:  getParent
     * ---------------------
     * Returns the parent memory object that created this 
     * child memory object. If this memory object is a 
     * root memory object it has no parent and therefore
     * returned value will be nullptr!
     * 
     * return: Parent or nullptr if already root parent
     */
    Memory* getParent();


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
     * memSize:  size of the handled memory
     */
    void setSize(size_t memSize);

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
    void* pointer(const size_t offset);

    /* Function:  isMainMemory
     * --------------------
     * Returns true if the memory is allocated in main memory
     */ 
    bool isMainMemory();

    /* Function:  isGPUMemory
     * --------------------
     * Returns true if the memory is allocated on a GPU
     */ 
    bool isGPUMemory();

    /* Function: getNumaNode
     * -------------
     * Returns on which NUMA node the memory is allocated
     * 
     * return numa node
     */
    int getNumaNode();


    /* Function:  isHuge
     * --------------------
     * Returns true if the main memory is allocated as huge page (LINUX only)
     */
    bool isHuge();


    /* Function:  getDeviceIndex
     * ---------------------
     * Returns the index of the GPU device where the GPU memory is allocated
     * 
     * return:  index of the device or negative if at that time selected 
     *          device was used
     */
    int getDeviceIndex();


    /* Function: isIBV
     * -------------
     * Returns true memory is registered with IBV
     * 
     * return:  true if IBV memory
     */
    bool isIBV();

    /* Function: getIBPort
     * ---------------
     * Returns the used IB port
     *
     * return:  IB port
     */
    int getIBPort();

    /* Function: ib_pd
     * ---------------
     * Returns the used IB protection domain handle
     *
     * return:  IB protection domain handle
     */
    ibv_pd* ib_pd();

    /* Function: ib_mr
     * ---------------
     * Returns the used IB memory registration handle
     *
     * return:  IB memory registration handle
     */
    ibv_mr* ib_mr();

    /* Function: ib_port_attributes
     * ---------------
     * Returns the used IB port attributes handle
     *
     * return:  IB port attributes handle
     */
    ibv_port_attr ib_port_attributes();

    /* Function: ib_context
     * ---------------
     * Returns the used IB context handle
     *
     * return:  IB context handle
     */
    ibv_context* ib_context();


    /* Function: getFreeMemList
     * ---------------
     * Returns the internal tracking list of available memory regions
     * on top of the physical buffer.
     * Not relevant, just for debugging
     */
    const std::list<rdma_mem_t> getFreeMemList() const { return m_rdmaMem; }


    /* Function: internalAlloc
     * ---------------
     * Allocates a memory space on top of the physical buffer.
     * Used when using malloc() method of this class.
     * 
     * Use malloc() instead!
     */
    rdma_mem_t internalAlloc(size_t size);

    /* Function: printBuffer
     * ---------------
     * Prints information about to internal memory
     * management structures that work on top of
     * the physical memory.
     * Not relevant, just for debugging
     */
    void printBuffer();

    /* Function: internalFree
     * ---------------
     * Releases an allocated memory part based on the pointer
     * 
     * ptr:  pointer to the allocated memory
     * 
     */
    void internalFree(const void* ptr);

    /* Function: internalFree
     * ---------------
     * Releases an allocated memory part based on the offset
     * 
     * offset:  offset to the allocated memory part
     * 
     */
    void internalFree(const size_t &offset);


    /* Function: alloc
     * ---------------
     * Returns a new memory object (child) that does not allocate new memory space
     * but instead handles a reserved portion of this memory's array (parent).
     * Reserved portion can be released again by destructing the returned memory object.
     *
     * return:  New child memory instance or nullptr if not enough space left
     */
    Memory* malloc(size_t memSize);


    /* Function: toString
     * ----------------------
     * Returns the byte values of the whole buffer
     * 
     * return:  byte values as string
     */
    std::string toString();

     /* Function: toString
     * ----------------------
     * Returns the byte values of a certain section 
     * of this memory as string
     * 
     * offset:  offset from where to start reading bytes
     * count:  how many bytes should be returned
     * 
     * return:  byte values as string
     */
    std::string toString(size_t offset, size_t count);

    /* Function: print
     * ----------------------
     * Prints all bytes of this memory
     * 
     */
    void print();

    /* Function: print
     * ----------------------
     * Prints the byte values of a certain section of this memory
     * 
     * offset:  offset from where to start printing bytes
     * count:  how many bytes should be printed
     * 
     */
    void print(size_t offset, size_t count);

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
    void openContext();

    /* Function: closeContext
     * ---------------------
     * Closes the opened context again such that 
     * another memory operating on the same device 
     * can use it again
     */
    void closeContext();

    /* Function:  setRandom
     * ---------------------
     * Randomly initializes each byte of the buffer
     * 
     */
    void setRandom();

    /* Function:  setRandom
     * ---------------------
     * Randomly initializes byte-wise a part of the buffer
     * offset:  Offset where to start randomizing
     * count:  How many bytes should be randomized
     */
    void setRandom(size_t offset, size_t count);

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the handled memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     *
     */
    void setMemory(int value);

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the handled memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     * count:    how many bytes should be set
     *
     */
    void setMemory(int value, size_t count) ;

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the handled memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     * offset: where to start writing the bytes
     * count:    how many bytes should be set
     *
     */
    void setMemory(int value, size_t offset, size_t count) ;


    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * memType:      memory type of the destination (default to main memory, which GPU doesn't matter)
     */
    void copyTo(void *destination, MEMORY_TYPE memType=MEMORY_TYPE::MAIN) ;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * count:          how many bytes should be copied
     * memType:      memory type of the destination (default to main memory, which GPU doesn't matter)
     */
    void copyTo(void *destination, size_t count, MEMORY_TYPE memType=MEMORY_TYPE::MAIN) ;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the handled memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the handled memory should be copied to
     * destOffset:   offset where to start writing bytes at destination
     * srcOffset:    offset where to start reading bytes at this buffer
     * count:          how many bytes should be copied
     * memType:      memory type of the destination (default to main memory, which GPU doesn't matter)
     */
    void copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t count, MEMORY_TYPE memType=MEMORY_TYPE::MAIN) ;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * memType: memory type of the source (default from main memory, which GPU doesn't matter)
     */
    void copyFrom(const void *source, MEMORY_TYPE memType=MEMORY_TYPE::MAIN) ;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * count:     how many bytes should be copied
     * memType: memory type of the source (default from main memory, which GPU doesn't matter)
     */
    void copyFrom(const void *source, size_t count, MEMORY_TYPE memType=MEMORY_TYPE::MAIN) ;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the handled memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * srcOffset:    offset where to start reading bytes at source
     * destOffset:   offset where to start writing bytes at this buffer
     * count:     how many bytes should be copied
     * memType: memory type of the source (default from main memory, which GPU doesn't matter)
     */
    void copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t count, MEMORY_TYPE memType=MEMORY_TYPE::MAIN) ;


    /* Function:  getChar
     * ---------------------
     * Reads a char
     * 
     * offset: where to start reading
     * return: read char
     */
    char getChar(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a char
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(char value, size_t offset) ;

    /* Function:  getInt8
     * ---------------------
     * Reads a 8 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    int8_t getInt8(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 8 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(int8_t value, size_t offset) ;

    /* Function:  getUInt8
     * ---------------------
     * Reads a 8 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    uint8_t getUInt8(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 8 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(uint8_t value, size_t offset) ;

    /* Function:  getInt16
     * ---------------------
     * Reads a 16 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    int16_t getInt16(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 16 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(int16_t value, size_t offset) ;

    /* Function:  getUInt16
     * ---------------------
     * Reads a 16 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    uint16_t getUInt16(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 16 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(uint16_t value, size_t offset) ;

    /* Function:  getInt32
     * ---------------------
     * Reads a 32 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    int32_t getInt32(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 32 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(int32_t value, size_t offset) ;

    /* Function:  getUInt32
     * ---------------------
     * Reads a 32 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    uint32_t getUInt32(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 32 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(uint32_t value, size_t offset) ;

    /* Function:  getInt64
     * ---------------------
     * Reads a 64 bit int
     * 
     * offset: where to start reading
     * return: read int
     */
    int64_t getInt64(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 64 bit int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(int64_t value, size_t offset) ;

    /* Function:  getUInt64
     * ---------------------
     * Reads a 64 bit unsigned int
     * 
     * offset: where to start reading
     * return: read int
     */
    uint64_t getUInt64(size_t offset) ;

    /* Function:  set
     * ---------------------
     * Writes a 64 bit unsigned int
     * 
     * value:  value that should be written
     * offset: where to start writing
     * 
     */
    void set(uint64_t value, size_t offset) ;
};

}