#ifndef BaseMemory_H_
#define BaseMemory_H_

#include "../utils/Config.h"

#include <infiniband/verbs.h>
#include <stdio.h>

namespace rdma {

class BaseMemory {

protected:
    size_t mem_size;
    void *buffer = NULL;
    struct ibv_pd *pd; // ProtectionDomain handle
    struct ibv_mr *mr; // MemoryRegistration handle for buffer

    // Device attributes
    int ib_port;
    struct ibv_port_attr port_attr; // IB port attributes
    struct ibv_context *ib_ctx;     // device handle

    void init();

public:

    /* Constructor
     * --------------
     * Base class to allocate specific memory
     *
     * mem_size:  size how much memory should be allocated
     *
     */
    BaseMemory(size_t mem_size, int ib_port=Config::Config::RDMA_IBPORT);

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

    /* Function:  pointer
     * ---------------------
     * Returns the pointer of the allocated memory.
     * What type of memory and how to handle the pointer is
     * implementation dependent 
     *
     * return:  pointer of the allocated memory
     */
    void* pointer();

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

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the allocated memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     *
     */
    virtual void setMemory(int value) = 0;

    /* Function:  setMemory
     * ---------------------
     * Sets each byte of the allocated memory to a given value.
     * Same behavior as memset()
     *
     * value:  value that should be set for each byte
     * num:    how many bytes should be set
     *
     */
    virtual void setMemory(int value, size_t num) = 0;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the allocated memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the allocated memory should be copied to
     *
     */
    virtual void copyTo(void *destination) = 0;

    /* Function:  copyTo
     * ---------------------
     * Copies the data from the allocated memory to a given destination.
     * Same behavior as memcpy()
     *
     * destination:  the data in the allocated memory should be copied to
     * num:          how many bytes should be copied
     *
     */
    virtual void copyTo(void *destination, size_t num) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the allocated memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     *
     */
    virtual void copyFrom(const void *source) = 0;

    /* Function:  copyFrom
     * ---------------------
     * Copies the data from a given source to the allocated memory.
     * Same behavior as memcpy()
     *
     * source:  the data that should be copied to
     * num:     how many bytes should be copied
     *
     */
    virtual void copyFrom(const void *source, size_t num) = 0;

};

} // namespace rdma

#endif /* BaseMemory_H_ */