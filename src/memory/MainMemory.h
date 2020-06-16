#ifndef MainMemory_H_
#define MainMemory_H_

#include "AbstractMainMemory.h"
#include "LocalMainMemoryStub.h"
#include "BaseMemory.h"

namespace rdma {
    
class MainMemory : virtual public AbstractMainMemory, virtual public BaseMemory {

protected:
    bool huge;
    int numa_node;

public:

    /* Constructor
     * --------------
     * Allocates main memory.
     * Uses therefore malloc()
     *
     * mem_size:  size how much memory should be allocated
     *
     */
    MainMemory(size_t mem_size);

    /* Constructor
     * --------------
     * Allocates main memory.
     * Uses therefore malloc() or mmap()
     *
     * mem_size:  size how much memory should be allocated
     * huge:      If true then a huge memory block can be 
     *            allocated (uses therefore mmap())
     *
     */
    MainMemory(size_t mem_size, bool huge);

    /* Constructor
     * --------------
     * Allocates main memory.
     * Uses therefore malloc() or mmap()
     *
     * mem_size:  size how much memory should be allocated
     * numa_node: Index of the NUMA node where the memory
     *            should be allocated on (LINUX only)
     *
     */
    MainMemory(size_t mem_size, int numa_node);

    /* Constructor
     * --------------
     * Allocates main memory.
     * Uses therefore malloc() or mmap()
     *
     * mem_size:  size how much memory should be allocated
     * huge:      If true then a huge memory block can be 
     *            allocated (uses therefore mmap())
     * numa_node: Index of the NUMA node where the memory
     *            should be allocated on (LINUX only)
     *
     */
    MainMemory(size_t mem_size, bool huge, int numa_node);

    // destructor
    virtual ~MainMemory();

    virtual bool isHuge();

    virtual int getNumaNode();

    LocalBaseMemoryStub *createLocalMemoryStub(void* pointer, size_t mem_size, std::function<void(const void* buffer)> freeFunc) override;
};

} // namespace rdma

#endif /* MainMemory_H_ */