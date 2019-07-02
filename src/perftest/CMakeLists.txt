set(PERFTEST_SRC
  PerfTest.h
  PerfTests.h
  RemoteMemoryPerf.h
  RemoteMemoryPerf.cc
  RemoteScanPerf.h
  RemoteScanPerf.cc
  MulticastPerf.h
  MulticastPerf.cc
  SwMulticastPerf.h
  SwMulticastPerf.cc
) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/

add_library(perftest ${PERFTEST_SRC})

target_include_directories(perftest PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  
target_link_libraries(
  perftest rdma_lib db_utils
)
