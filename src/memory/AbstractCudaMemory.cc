#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "AbstractCudaMemory.h"

using namespace rdma;

// constructors
AbstractCudaMemory::AbstractCudaMemory(size_t mem_size, int device_index) : AbstractBaseMemory(mem_size){
    this->device_index = device_index;
}
AbstractCudaMemory::AbstractCudaMemory(void* buffer, size_t mem_size) : AbstractBaseMemory(buffer, mem_size){
    this->device_index = -1;
}
AbstractCudaMemory::AbstractCudaMemory(void* buffer, size_t mem_size, int device_index) : AbstractBaseMemory(buffer, mem_size){
    this->device_index = device_index;
}

void AbstractCudaMemory::setMemory(int value){
    setMemory(value, 0, this->mem_size);
}

void AbstractCudaMemory::setMemory(int value, size_t num){
    setMemory(value, 0, num);
}

void AbstractCudaMemory::setMemory(int value, size_t offset, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemset((void*)((size_t)this->buffer + offset), value, num), "AbstractCudaMemory::setMemory could not set memory to value\n");
    resetDevice(previous_device_index);
}

void AbstractCudaMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void AbstractCudaMemory::copyTo(void *destination, size_t num){
    copyTo(destination, 0, 0, num);
}

void AbstractCudaMemory::copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemcpy((void*)((size_t)destination + destOffset), (void*)((size_t)this->buffer + srcOffset), num, cudaMemcpyDeviceToHost), 
                                "AbstractCudaMemory::copyTo could not copy data to given destination\n");
    resetDevice(previous_device_index);
}

void AbstractCudaMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void AbstractCudaMemory::copyFrom(const void *source, size_t num){
    copyFrom(source, 0, 0, num);
}

void AbstractCudaMemory::copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemcpy((void*)((size_t)this->buffer + destOffset), (void*)((size_t)source + srcOffset), num, cudaMemcpyHostToDevice), 
                                "AbstractCudaMemory::copyFrom could not copy data from given source\n");
    resetDevice(previous_device_index);
}

char AbstractCudaMemory::getChar(size_t offset){
    char tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(char value, size_t offset){
    copyFrom((void*)value, 0, offset, sizeof(value));
}

int16_t AbstractCudaMemory::getInt16(size_t offset){
    int16_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(int16_t value, size_t offset){
    copyFrom((void*)value, 0, offset, sizeof(value));
}

int32_t AbstractCudaMemory::getInt32(size_t offset){
    int32_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(int32_t value, size_t offset){
    copyFrom((void*)value, 0, offset, sizeof(value));
}

int64_t AbstractCudaMemory::getInt64(size_t offset){
    int64_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(int64_t value, size_t offset){
    copyFrom((void*)value, 0, offset, sizeof(value));
}

#endif /* CUDA support */