set(PERFTEST_SRC
  PerfTest.h
  BandwidthPerfTest.h
  BandwidthPerfTest.cc
  XRC_BandwidthPerfTest.h
  XRC_BandwidthPerfTest.cc
) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/

add_library(perftest_tmp ${PERFTEST_SRC})

target_include_directories(perftest_tmp PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  
target_link_libraries(
  perftest_tmp rdma_lib
)

add_executable(perf_test_tmp MainPerfTest.cc)
target_link_libraries(perf_test_tmp perftest_tmp)

find_package(gflags REQUIRED)
target_link_libraries(perf_test_tmp gflags)

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/PlotResults.py
    DESTINATION ${CMAKE_BINARY_DIR}/bin/
)