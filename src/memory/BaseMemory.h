#ifndef BaseMemory_H_
#define BaseMemory_H_

#include "AbstractBaseMemory.h"
#include "LocalBaseMemoryStub.h"
#include "../utils/Config.h"

#include <infiniband/verbs.h>
#include <stdio.h>

namespace rdma {

class BaseMemory : virtual public AbstractBaseMemory {

protected:
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

    /* Function createLocalMemoryStub
     * ---------------
     * Creates a memory stub object for handling 
     * a given memory sub-section of this memory
     *
     * pointer:  pointer to memory sub-section
     * mem_size:  size of memory sub-section
     * freeFunc:  function to release the memory sub-section
     * 
     * return: object for handling memory sub-section 
     */
    virtual LocalBaseMemoryStub *createLocalMemoryStub(void* pointer, size_t mem_size, std::function<void(const void* buffer)> freeFunc) = 0;
};

} // namespace rdma

#endif /* BaseMemory_H_ */