#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "AbstractCudaMemory.h"

using namespace rdma;

// constructors
AbstractCudaMemory::AbstractCudaMemory(size_t mem_size, int device_index) : AbstractBaseMemory(mem_size){
    this->device_index = device_index;
}
AbstractCudaMemory::AbstractCudaMemory(void* buffer, size_t mem_size) : AbstractBaseMemory(buffer, mem_size){}
AbstractCudaMemory::AbstractCudaMemory(void* buffer, size_t mem_size, int device_index) : AbstractBaseMemory(buffer, mem_size){
    this->device_index = device_index;
}

void AbstractCudaMemory::setMemory(int value){
    setMemory(value, this->mem_size);
}

void AbstractCudaMemory::setMemory(int value, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemset(this->buffer, value, num), "AbstractCudaMemory::setMemory could not set memory to value\n");
    resetDevice(previous_device_index);
}

void AbstractCudaMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void AbstractCudaMemory::copyTo(void *destination, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemcpy(destination, this->buffer, num, cudaMemcpyDeviceToHost), "AbstractCudaMemory::copyTo could not copy data to given destination\n");
    resetDevice(previous_device_index);
}

void AbstractCudaMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void AbstractCudaMemory::copyFrom(const void *source, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemcpy(this->buffer, source, num, cudaMemcpyHostToDevice), "AbstractCudaMemory::copyFrom could not copy data from given source\n");
    resetDevice(previous_device_index);
}

#endif /* CUDA support */