#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "CudaMemory.h"

using namespace rdma;

// constructors
CudaMemory::CudaMemory(size_t mem_size) : CudaMemory(mem_size, -1){}
CudaMemory::CudaMemory(size_t mem_size, int device_index) : BaseMemory(mem_size){
    this->device_index = device_index;
    
    // allocate CUDA memory
    int previous_device_index = selectDevice();
    checkCudaError(cudaMalloc(&(this->buffer), mem_size), "CudaMemory::CudaMemory could not allocate memory\n");
    checkCudaError(cudaMemset(this->buffer, 0, mem_size), "CudaMemory::CudaMemory could not set allocated memory to zero\n");

    this->init();

    resetDevice(previous_device_index);
}

// destructor
CudaMemory::~CudaMemory(){
    // release CUDA memory
    int previous_device_index = selectDevice();
    checkCudaError(cudaFree(this->buffer), "CudaMemory::~CudaMemory could not free memory\n");
    resetDevice(previous_device_index);
}

void CudaMemory::setMemory(int value){
    setMemory(value, this->mem_size);
}

void CudaMemory::setMemory(int value, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemset(this->buffer, value, num), "CudaMemory::setMemory could not set memory to value\n");
    resetDevice(previous_device_index);
}

void CudaMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void CudaMemory::copyTo(void *destination, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemcpy(destination, this->buffer, num, cudaMemcpyDeviceToHost), "CudaMemory::copyTo could not copy data to given destination\n");
    resetDevice(previous_device_index);
}

void CudaMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void CudaMemory::copyFrom(const void *source, size_t num){
    int previous_device_index = selectDevice();
    checkCudaError(cudaMemcpy(this->buffer, source, num, cudaMemcpyHostToDevice), "CudaMemory::copyFrom could not copy data from given source\n");
    resetDevice(previous_device_index);
}

#endif /* CUDA support */