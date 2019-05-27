set(TEST_PROTO_SRC
  TestProtoServer.cc
  TestProtoServer.h 
  ) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/
  
add_library(test_rdma_proto ${TEST_PROTO_SRC})

target_include_directories(test_rdma_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rdma_proto rdma_proto)
