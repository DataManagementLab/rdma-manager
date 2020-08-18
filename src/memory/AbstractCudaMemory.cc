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

AbstractCudaMemory::~AbstractCudaMemory(){
    if(this->open_context_counter == 0) return;
    this->open_context_counter = 1;
    closeContext();
}

void AbstractCudaMemory::openContext(){
    if(this->device_index < 0) return;
    this->open_context_counter++;
    if(this->open_context_counter > 1) return;
    this->previous_device_index = -1;
    checkCudaError(cudaGetDevice(&(this->previous_device_index)), "AbstractCudaMemory::openContext could not get selected device\n");
    if(this->previous_device_index == this->device_index) return;
    checkCudaError(cudaSetDevice(this->device_index), "AbstractCudaMemory::openContext could not set selected device\n");
}

void AbstractCudaMemory::closeContext(){
    if(this->device_index < 0 || this->open_context_counter==0) return;
    this->open_context_counter--;
    if(this->open_context_counter > 0) return;
    if(this->device_index != this->previous_device_index)
        checkCudaError(cudaSetDevice(this->previous_device_index), "AbstractCudaMemory::closeContext could not reset selected device\n");
    this->previous_device_index = -1;
}

void AbstractCudaMemory::setMemory(int value){
    setMemory(value, 0, this->mem_size);
}

void AbstractCudaMemory::setMemory(int value, size_t num){
    setMemory(value, 0, num);
}

void AbstractCudaMemory::setMemory(int value, size_t offset, size_t num){
    openContext();
    checkCudaError(cudaMemset((void*)((size_t)this->buffer + offset), value, num), "AbstractCudaMemory::setMemory could not set memory to value\n");
    closeContext();
}

void AbstractCudaMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void AbstractCudaMemory::copyTo(void *destination, size_t num){
    copyTo(destination, 0, 0, num);
}

void AbstractCudaMemory::copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num){
    openContext();
    checkCudaError(cudaMemcpy((void*)((size_t)destination + destOffset), (void*)((size_t)this->buffer + srcOffset), num, cudaMemcpyDeviceToHost), 
                                "AbstractCudaMemory::copyTo could not copy data to given destination\n");
    closeContext();
}

void AbstractCudaMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void AbstractCudaMemory::copyFrom(const void *source, size_t num){
    copyFrom(source, 0, 0, num);
}

void AbstractCudaMemory::copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num){
    openContext();
    checkCudaError(cudaMemcpy((void*)((size_t)this->buffer + destOffset), (void*)((size_t)source + srcOffset), num, cudaMemcpyHostToDevice), 
                                "AbstractCudaMemory::copyFrom could not copy data from given source\n");
    closeContext();
}

char AbstractCudaMemory::getChar(size_t offset){
    char tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(char value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

int16_t AbstractCudaMemory::getInt16(size_t offset){
    int16_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(int16_t value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

uint16_t AbstractCudaMemory::getUInt16(size_t offset){
    uint16_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(uint16_t value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

int32_t AbstractCudaMemory::getInt32(size_t offset){
    int32_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(int32_t value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

uint32_t AbstractCudaMemory::getUInt32(size_t offset){
    uint32_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(uint32_t value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

int64_t AbstractCudaMemory::getInt64(size_t offset){
    int64_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(int64_t value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

uint64_t AbstractCudaMemory::getUInt64(size_t offset){
    uint64_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp));
    return tmp[0];
}

void AbstractCudaMemory::set(uint64_t value, size_t offset){
    copyFrom((void*)&value, 0, offset, sizeof(value));
}

#endif /* CUDA support */