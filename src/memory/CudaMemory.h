#ifndef CudaMemory_H_
#define CudaMemory_H_

#include "BaseMemory.h"
#include <cuda_runtime_api.h>

namespace rdma {
    
class CudaMemory : public BaseMemory {

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
        checkCudaError(cudaGetDevice(&previous_device_index), "CudaMemory::selectDevice could not get selected device\n");
        if(previous_device_index == this->device_index) return previous_device_index;
        checkCudaError(cudaSetDevice(this->device_index), "CudaMemory::selectDevice could not set selected device\n");
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
        checkCudaError(cudaSetDevice(previous_device_index), "CudaMemory::resetDevice could not reset selected device\n");
    }

public:
    
    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * mem_size:      size how much memory should be allocated
     * device_index:  index of the GPU device that should be used to
     *                allocate the memory. If negative the currently 
     *                selected device will be used
     *
     */
    CudaMemory(size_t mem_size, int device_index);

    // destructor
    ~CudaMemory();

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

    virtual void copyFrom(void *source) override;

    virtual void copyFrom(void *source, size_t num) override;

};

} // namespace rdma

#endif /* CudaMemory_H_ */