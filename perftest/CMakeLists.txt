set(PERFTEST_SRC
  PerfTest.h
  BandwidthPerfTest.h
  BandwidthPerfTest.cc
  AtomicsBandwidthPerfTest.h
  AtomicsBandwidthPerfTest.cc
  LatencyPerfTest.h
  LatencyPerfTest.cc
  AtomicsLatencyPerfTest.h
  AtomicsLatencyPerfTest.cc
  OperationsCountPerfTest.h
  OperationsCountPerfTest.cc
  AtomicsOperationsCountPerfTest.h
  AtomicsOperationsCountPerfTest.cc
) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/

add_library(perftest ${PERFTEST_SRC})

target_include_directories(perftest PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  
target_link_libraries(
  perftest rdma_lib
)

add_executable(perf_test MainPerfTest.cc)
target_link_libraries(perf_test perftest)

find_package(gflags REQUIRED)
target_link_libraries(perf_test gflags)

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/PlotResults.py
    DESTINATION ${CMAKE_BINARY_DIR}/bin/
)