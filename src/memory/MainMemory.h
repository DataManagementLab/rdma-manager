#ifndef MainMemory_H_
#define MainMemory_H_

#include "BaseMemory.h"

namespace rdma {
    
class MainMemory : public BaseMemory {

protected:
    bool huge;

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

    // destructor
    ~MainMemory();

    virtual void setMemory(int value) override;

    virtual void setMemory(int value, size_t num) override;

    virtual void copyTo(void *destination) override;

    virtual void copyTo(void *destination, size_t num) override;

    virtual void copyFrom(void *source) override;

    virtual void copyFrom(void *source, size_t num) override;

};

} // namespace rdma

#endif /* MainMemory_H_ */