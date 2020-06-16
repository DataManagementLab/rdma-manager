#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "CudaMemory.h"

using namespace rdma;

// constructors
CudaMemory::CudaMemory(size_t mem_size) : CudaMemory(mem_size, -1){}
CudaMemory::CudaMemory(size_t mem_size, int device_index) : AbstractBaseMemory(mem_size), AbstractCudaMemory(mem_size, device_index), BaseMemory(mem_size){
    
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

LocalBaseMemoryStub *CudaMemory::createLocalMemoryStub(void* pointer, size_t mem_size, std::function<void(const void* buffer)> freeFunc){
    return (LocalBaseMemoryStub*) new LocalCudaMemoryStub(pointer, mem_size, freeFunc);
}

#endif /* CUDA support */