#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "CudaMemory.h"
#include "../utils/GpuNumaUtils.h"

using namespace rdma;

// constructors
CudaMemory::CudaMemory(size_t mem_size) : CudaMemory(mem_size, -2){}
CudaMemory::CudaMemory(size_t mem_size, MEMORY_TYPE memory_type) : CudaMemory(mem_size, (int)memory_type, -1){}
CudaMemory::CudaMemory(size_t mem_size, int device_index) : CudaMemory(mem_size, device_index, -1){}
CudaMemory::CudaMemory(size_t mem_size, MEMORY_TYPE memory_type, int ib_numa) : CudaMemory(mem_size, (int)memory_type, ib_numa){}
CudaMemory::CudaMemory(size_t mem_size, int device_index, int ib_numa) : AbstractBaseMemory(mem_size), AbstractCudaMemory(mem_size, device_index), BaseMemory(mem_size, ib_numa){
    if(this->device_index < -2) throw std::invalid_argument("Device index cannot be smaller than -2. See documentation");
    if(this->device_index == -2) this->device_index = GpuNumaUtils::get_cuda_device_index_by_numa();

    if(ib_numa < 0){
        this->numa_node = GpuNumaUtils::get_numa_node_by_cuda_device_index(this->device_index);
        if(this->numa_node < 0) this->numa_node = rdma::Config::RDMA_NUMAREGION;
    }

    this->preInit();

    // allocate CUDA memory
    openContext();
    checkCudaError(cudaMalloc(&(this->buffer), mem_size), "CudaMemory::CudaMemory could not allocate memory\n");
    checkCudaError(cudaMemset(this->buffer, 0, mem_size), "CudaMemory::CudaMemory could not set allocated memory to zero\n");

    this->postInit();

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
    size_t rootOffset = (size_t)alloc(size) - (size_t)this->buffer;
    return (LocalBaseMemoryStub*) new LocalCudaMemoryStub(this->buffer, rootOffset, size, this->device_index, [this](const void* ptr){
        free(ptr);
    });
}

#endif /* CUDA support */