set(RDMA_MEMORY_SRC
  AbstractBaseMemory.h
  AbstractBaseMemory.cc
  AbstractMainMemory.h
  AbstractMainMemory.cc
  AbstractCudaMemory.h
  AbstractCudaMemory.cc
  BaseMemory.h
  BaseMemory.cc
  MainMemory.h
  MainMemory.cc
  CudaMemory.h
  CudaMemory.cc
  LocalBaseMemoryStub.h
  LocalBaseMemoryStub.cc
  LocalMainMemoryStub.h
  LocalMainMemoryStub.cc
  LocalCudaMemoryStub.h
  LocalCudaMemoryStub.cc
) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/
add_library(rdma_memory_lib ${RDMA_MEMORY_SRC})
target_include_directories(rdma_memory_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (CUDA_ENABLED)
target_link_libraries(rdma_memory_lib ${CUDA_LIBRARIES})
endif()

find_package(IBVerbs)
target_link_libraries(rdma_memory_lib ${IBVERBS_LIBRARY})
find_package(Numa)
target_link_libraries(rdma_memory_lib ${NUMA_LIBRARY})

# target_link_libraries(rdma_memory_lib ${RDMA_LINKER_FLAGS})