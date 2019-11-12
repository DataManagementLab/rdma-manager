FIND_PACKAGE(CPPUNIT REQUIRED)
find_package(GTest REQUIRED)
enable_testing()

# add_subdirectory(proto)
add_subdirectory(rdma)
# add_subdirectory(thread)
# add_subdirectory(utils)
# add_subdirectory(RPC)

set(TEST_SRC
	Tests.h
		)

add_library(test_lib ${TEST_SRC})

add_executable(rdma_tests TestRunner.cc)
 
add_test(NAME rdma_tests COMMAND ${EXECUTABLE_OUTPUT_PATH}/rdma_tests)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure 
DEPENDS rdma_tests)
  
SET_TARGET_PROPERTIES(rdma_tests PROPERTIES
   PREFIX ""
   LINK_FLAGS "-Wl,--no-as-needed"
)

target_link_libraries(rdma_tests 
	${CPPUNIT_LIBRARY}
	# test_rdma_proto
	test_rdma
	# test_thread
	# test_utils
	# test_rpc
)
	
target_link_libraries(test_lib 
	${CPPUNIT_LIBRARY}
	# test_rdma_proto
	test_rdma
	# test_thread
	# test_utils
	# test_rpc
)


add_executable(rdma_test Main.cc)
target_link_libraries(rdma_test
	test_lib
	GTest::GTest GTest::Main
)

add_executable(perf_test MainPerf.cc)
target_link_libraries(perf_test perftest)
	
