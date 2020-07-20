#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "CudaMemory.h"
#include "../utils/GpuNumaUtils.h"

using namespace rdma;

// constructors
CudaMemory::CudaMemory(size_t mem_size) : CudaMemory(mem_size, -2){}
CudaMemory::CudaMemory(size_t mem_size, int device_index) : AbstractBaseMemory(mem_size), AbstractCudaMemory(mem_size, device_index), BaseMemory(mem_size){
    if(this->device_index < -1) this->device_index = GpuNumaUtils::get_cuda_device_index_for_numa_node();

    // allocate CUDA memory
    openContext();
    checkCudaError(cudaMalloc(&(this->buffer), mem_size), "CudaMemory::CudaMemory could not allocate memory\n");
    checkCudaError(cudaMemset(this->buffer, 0, mem_size), "CudaMemory::CudaMemory could not set allocated memory to zero\n");

    this->init();

    closeContext();
}

// destructor
CudaMemory::~CudaMemory(){
    // release CUDA memory
    openContext();
    checkCudaError(cudaFree(this->buffer), "CudaMemory::~CudaMemory could not free memory\n");
    closeContext();
}

LocalBaseMemoryStub *CudaMemory::malloc(size_t size){
    return (LocalBaseMemoryStub*) new LocalCudaMemoryStub(alloc(size), size, this->device_index, [this](const void* ptr){
        free(ptr);
    });
}

#endif /* CUDA support */