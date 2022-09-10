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

add_library(perftest_old ${PERFTEST_SRC})

target_include_directories(perftest_old PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  
target_link_libraries(
  perftest_old rdma_lib
)

add_executable(perf_test_old MainPerf.cc)
target_link_libraries(perf_test_old perftest_old)
	
add_executable(multicast_order_test MulticastOrderTest.cc)
target_link_libraries(multicast_order_test rdma_lib)