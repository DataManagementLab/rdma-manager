#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef AbstractCudaMemory_H_
#define AbstractCudaMemory_H_

#include "AbstractBaseMemory.h"
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace rdma {
    
class AbstractCudaMemory : virtual public AbstractBaseMemory {

protected:
    int device_index, previous_device_index, open_context_counter = 0;

    /* Function:  checkCudaError
     * ---------------------
     * Handles CUDA errors
     *
     * code:  CUDA error code that should be handled
     * msg:   Message that should be printed if code is an error code
     *
     */
    void checkCudaError(cudaError_t code, const char* msg){
        if(code != cudaSuccess){
            fprintf(stderr, "CUDA-Error(%i): %s", code, msg);
            throw std::runtime_error("CUDA error has occurred!");
        }
    }

    /* Constructor
     * --------------
     * Handles CUDA memory.
     *
     * mem_size:  size how much memory should be handled
     *
     */
    AbstractCudaMemory(size_t mem_size, int device_index);

public:

    /* Constructor
     * --------------
     * Handles CUDA memory.
     * buffer:  pointer to CUDA memory that should be handled
     * mem_size:  size how much memory should be handled
     *
     */
    AbstractCudaMemory(void* buffer, size_t mem_size);

    /* Constructor
     * --------------
     * Handles CUDA memory.
     * buffer:  pointer to CUDA memory that should be handled
     * mem_size:  size how much memory should be handled
     * device_index:  index of the GPU device that should be used to
     *                handle the memory. If negative the currently 
     *                selected device will be used
     *
     */
    AbstractCudaMemory(void* buffer, size_t mem_size, int device_index);

    ~AbstractCudaMemory();

    /* Function:  getDeviceIndex
     * ---------------------
     * Returns the index of the GPU device where the memory is allocated
     * 
     * return:  index of the device or negative if at that time selected 
     *          device was used
     */
    int getDeviceIndex(){
        return this->device_index;
    }

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

#endif /* AbstractCudaMemory_H_ */
#endif /* CUDA support */