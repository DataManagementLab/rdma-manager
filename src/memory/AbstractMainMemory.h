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

    virtual void openContext() override;

    virtual void closeContext() override;

    virtual void setMemory(int value) override;

    virtual void setMemory(int value, size_t num) override;

    virtual void setMemory(int value, size_t offset, size_t num) override;

    virtual void copyTo(void *destination) override;

    virtual void copyTo(void *destination, size_t num) override;

    virtual void copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num) override;

    virtual void copyFrom(const void *source) override;

    virtual void copyFrom(const void *source, size_t num) override;

    virtual void copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num) override;

    virtual char getChar(size_t offset) override;

    virtual void set(char value, size_t offset) override;

    virtual int16_t getInt16(size_t offset) override;

    virtual void set(int16_t value, size_t offset) override;

    virtual uint16_t getUInt16(size_t offset) override;

    virtual void set(uint16_t value, size_t offset) override;

    virtual int32_t getInt32(size_t offset) override;

    virtual void set(int32_t value, size_t offset) override;

    virtual uint32_t getUInt32(size_t offset) override;

    virtual void set(uint32_t value, size_t offset) override;

    virtual int64_t getInt64(size_t offset) override;

    virtual void set(int64_t value, size_t offset) override;

    virtual uint64_t getUInt64(size_t offset) override;

    virtual void set(uint64_t value, size_t offset) override;
};

} // namespace rdma

#endif /* AbstractMainMemory_H_ */