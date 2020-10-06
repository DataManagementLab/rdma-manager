#ifndef BaseMemory_H_
#define BaseMemory_H_

#include "AbstractBaseMemory.h"
#include "LocalBaseMemoryStub.h"
#include "../utils/Config.h"

#include <infiniband/verbs.h>
#include <stdio.h>
#include <list>
#include <mutex>
#include <condition_variable>

namespace rdma {

struct rdma_mem_t {
  size_t size; // size of memory region
  bool free;
  size_t offset;
  bool isnull;

  rdma_mem_t(size_t initSize, bool initFree, size_t initOffset)
      : size(initSize), free(initFree), offset(initOffset), isnull(false) {}

  rdma_mem_t() : size(0), free(false), offset(0), isnull(true) {}
};


class BaseMemory : virtual public AbstractBaseMemory {

protected:
    int numa_node;

    struct ibv_pd *pd; // ProtectionDomain handle
    struct ibv_mr *mr; // MemoryRegistration handle for buffer

    // Device attributes
    int ib_port;
    struct ibv_port_attr port_attr; // IB port attributes
    struct ibv_context *ib_ctx;     // device handle

    // Memory management
    list<rdma_mem_t> m_rdmaMem;
    unordered_map<size_t, rdma_mem_t> m_usedRdmaMem; // <offset, memory-segment>
    static rdma_mem_t s_nillmem;

    // Thread safe alloc/free
    std::recursive_mutex m_lockMem;

    void preInit();
    void postInit();

public:

    /* Constructor
     * --------------
     * Base class to allocate specific memory
     *
     * mem_size:  size how much memory should be allocated
     *
     */
    BaseMemory(size_t mem_size, int numa_node=Config::RDMA_NUMAREGION, int ib_port=Config::RDMA_IBPORT);

    /* Destructor
     * -------------
     * Releases the allocated memory
     */
    virtual ~BaseMemory();

    /* Function: getNumaNode
     * -------------
     * Returns on which NUMA node the memory is allocated
     * 
     * return numa node
     */
    int getNumaNode();

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

    const list<rdma_mem_t> getFreeMemList() const { return m_rdmaMem; }
    
    void mergeFreeMem(list<rdma_mem_t>::iterator &iter);

    rdma_mem_t internalAlloc(size_t size);

    void printBuffer();

    /* Function: alloc
     * ---------------
     * Allocates a part of this memory
     *
     * size:   how many bytes should be allocated
     * return: pointer to the allocated memory
     */
    void* alloc(size_t size);

    /* Function: free
     * ---------------
     * Releases an allocated memory part based on the pointer
     * 
     * ptr:  pointer to the allocated memory
     * 
     */
    void free(const void* ptr);

    /* Function: free
     * ---------------
     * Releases an allocated memory part based on the offset
     * 
     * offset:  offset to the allocated memory part
     * 
     */
    void free(const size_t &offset);

    /* Function: malloc
     * ---------------
     * Allocates a part of this memory
     * 
     * size:   how many bytes should be allocated
     * return: memory handler 
     */
    virtual LocalBaseMemoryStub *malloc(size_t size) = 0;

};

} // namespace rdma

#endif /* BaseMemory_H_ */