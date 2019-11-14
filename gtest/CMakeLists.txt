find_package(GTest REQUIRED)
enable_testing()

# set_source_files_properties(utils/TestConfig.h PROPERTIES LANGUAGE CXX)
#    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#        add_definitions("-x c++")
#    endif()


# add_subdirectory(proto)
# add_subdirectory(rdma)
# add_subdirectory(thread)
# add_subdirectory(utils)
# add_subdirectory(RPC) 

# add_library(test_lib ${TEST_SRC})
  # Unit Tests
  ##############\


  MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


SUBDIRLIST(SUBDIRS ".")



  include_directories(${GTEST_INCLUDE_DIRS})
  foreach(subdir ${SUBDIRS})
  include_directories(${subdir})
	endforeach()
  
  file(GLOB test_SRC ./*/*.cc)
  message(${test_SRC})

  add_executable(runUnitTests ${test_SRC})
  # Standard linking to gtest stuff.
  target_link_libraries(runUnitTests GTest::GTest  pthread
	GTest::Main
	rdma_lib)

# )


# add_executable(rdma_test gtest_main)
# target_link_libraries(rdma_test
# 	test_lib
# 	GTest::GTest GTest::Main
# )

# add_executable(perf_test MainPerf.cc)
# target_link_libraries(perf_test perftest)
	
