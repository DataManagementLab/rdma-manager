#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef AbstractCudaMemory_H_
#define AbstractCudaMemory_H_

#include "AbstractBaseMemory.h"
#include <cuda_runtime_api.h>

namespace rdma {
    
class AbstractCudaMemory : virtual public AbstractBaseMemory {

protected:
    int device_index;

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
        }
    }

     /* Function:  selectDevice
     * ---------------------
     * Selects the device where the memory is/should be allocated
     *
     * return:  index of the previously selected GPU device index
     */
    int selectDevice(){
        if(this->device_index < 0) return -1;
        int previous_device_index = -1;
        checkCudaError(cudaGetDevice(&previous_device_index), "AbstractCudaMemory::selectDevice could not get selected device\n");
        if(previous_device_index == this->device_index) return previous_device_index;
        checkCudaError(cudaSetDevice(this->device_index), "AbstractCudaMemory::selectDevice could not set selected device\n");
        return previous_device_index;
    }

    /* Function:  resetDevice
     * ---------------------
     * Sets the selected GPU device to the given device index
     *
     * previous_device_index:  index of the GPU device that should be selected
     *
     */
    void resetDevice(int previous_device_index){
        if(this->device_index < 0 || this->device_index == previous_device_index) return;
        checkCudaError(cudaSetDevice(previous_device_index), "AbstractCudaMemory::resetDevice could not reset selected device\n");
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

    virtual void setMemory(int value) override;

    virtual void setMemory(int value, size_t num) override;

    virtual void copyTo(void *destination) override;

    virtual void copyTo(void *destination, size_t num) override;

    virtual void copyFrom(const void *source) override;

    virtual void copyFrom(const void *source, size_t num) override;
};

} // namespace rdma

#endif /* AbstractCudaMemory_H_ */
#endif /* CUDA support */