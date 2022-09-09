set(PERFTEST_SRC
  PerfTest.h
  PerfTests.h
  RemoteMemoryPerf.h
  RemoteMemoryPerf.cc
  #RemoteScanPerf.h
  #RemoteScanPerf.cc
  MulticastPerf.h
  MulticastPerf.cc
  MulticastPerfLat.h
  MulticastPerfLat.cc
  #SwMulticastPerf.h
  #SwMulticastPerf.cc
  FetchAndAddPerf.h
  FetchAndAddPerf.cc
    RPCPerf.cc
    RPCPerf.h
    ) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/

add_library(perftest ${PERFTEST_SRC})

target_include_directories(perftest PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  
target_link_libraries(
  perftest rdma_lib
)

add_executable(perf_test MainPerf.cc)
target_link_libraries(perf_test perftest)
	
add_executable(multicast_order_test MulticastOrderTest.cc)
target_link_libraries(multicast_order_test rdma_lib)