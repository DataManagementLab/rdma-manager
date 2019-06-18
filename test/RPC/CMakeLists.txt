set(TEST_RPC_SRC
        TestRPC.cc
        TestRPC.h

  ) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/
  
add_library(test_rpc ${TEST_RPC_SRC})

target_include_directories(test_rpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rpc net_rpc)
