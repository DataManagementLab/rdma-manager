#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef CudaMemory_H_
#define CudaMemory_H_

#include "AbstractCudaMemory.h"
#include "LocalCudaMemoryStub.h"
#include "BaseMemory.h"

namespace rdma {
    
class CudaMemory : virtual public AbstractCudaMemory, virtual public BaseMemory {

public:
    
    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory based of the preferred NUMA region
     *
     * mem_size:      size how much memory should be allocated
     *
     */
    CudaMemory(size_t mem_size);

    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * mem_size:      size how much memory should be allocated
     * memory_type:   enum instead of device index. Type MAIN is not allowed!
     *
     */
    CudaMemory(size_t mem_size, MEMORY_TYPE memory_type);

    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * mem_size:      size how much memory should be allocated
     * device_index:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected based on the preferred NUMA region
     *
     */
    CudaMemory(size_t mem_size, int device_index);

    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * mem_size:      size how much memory should be allocated
     * memory_type:   enum instead of device index. Type MAIN is not allowed!
     * ib_numa:       Defines on which NUMA region the memory should be 
     *                registered for IBV. (-1 to automatically detect NUMA region)
     *
     */
    CudaMemory(size_t mem_size, MEMORY_TYPE memory_type, int ib_numa);

    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * mem_size:      size how much memory should be allocated
     * device_index:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected based on the preferred NUMA region
     * ib_numa:       Defines on which NUMA region the memory should be 
     *                registered for IBV. (-1 to automatically detect NUMA region)
     *
     */
    CudaMemory(size_t mem_size, int device_index, int ib_numa);

    /* Constructor
     * --------------
     * Allocates CUDA (GPU) memory
     *
     * register_ibv:  If memory should be registered with IBV
     * mem_size:      size how much memory should be allocated
     * device_index:  index of the GPU device that should be used to
     *                allocate the memory. If -1 the currently 
     *                selected device will be used. If -2 a device 
     *                will be selected based on the preferred NUMA region
     * ib_numa:       Defines on which NUMA region the memory should be 
     *                registered for IBV. (-1 to automatically detect NUMA region)
     *
     */
    CudaMemory(bool register_ibv, size_t mem_size, int device_index, int ib_numa);

    // destructor
    virtual ~CudaMemory();

    LocalBaseMemoryStub *malloc(size_t size) override;

    LocalBaseMemoryStub *createStub(void* rootBuffer, size_t rootOffset, size_t mem_size, std::function<void(const void* buffer)> freeFunc=nullptr) override;
};

} // namespace rdma

#endif /* CudaMemory_H_ */
#endif /* CUDA support */