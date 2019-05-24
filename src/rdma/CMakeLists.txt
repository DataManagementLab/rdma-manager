set(NET_RDMA_SRC
  RDMAManager.h
  RDMAManager.cc
  RDMAManagerRC.h
  RDMAManagerRC.cc
  RDMAManagerUD.h
  RDMAManagerUD.cc
  RDMAServer.h
  RDMAServer.cc
  RDMAClient.h
  RDMAClient.cc
) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/
add_library(net_rdma ${NET_RDMA_SRC})
target_include_directories(net_rdma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(IBVerbs)
target_link_libraries(net_rdma ${IBVERBS_LIBRARY})
find_package(RdmaCm)
target_link_libraries(net_rdma ${RDMACM_LIBRARY})
find_package(Numa)
target_link_libraries(net_rdma ${NUMA_LIBRARY})

target_link_libraries(net_rdma
  ${RDMA_LINKER_FLAGS}
  net_message
  net_proto
)