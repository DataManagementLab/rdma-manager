#ifndef AbstractMainMemory_H_
#define AbstractMainMemory_H_

#include "AbstractBaseMemory.h"

namespace rdma {
    
class AbstractMainMemory : virtual public AbstractBaseMemory {

protected:

    /* Constructor
     * --------------
     * Handles main memory.
     *
     * mem_size:  size how much memory should be handled
     *
     */
    AbstractMainMemory(size_t mem_size);

public:

    /* Constructor
     * --------------
     * Handles main memory.
     * buffer:  pointer to main memory that should be handled
     * mem_size:  size how much memory should be handled
     *
     */
    AbstractMainMemory(void* buffer, size_t mem_size);

    virtual void setMemory(int value) override;

    virtual void setMemory(int value, size_t num) override;

    virtual void copyTo(void *destination) override;

    virtual void copyTo(void *destination, size_t num) override;

    virtual void copyFrom(const void *source) override;

    virtual void copyFrom(const void *source, size_t num) override;
};

} // namespace rdma

#endif /* AbstractMainMemory_H_ */