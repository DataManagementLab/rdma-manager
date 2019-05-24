set(TEST_NET_SRC
  TestRDMAServerMCast.cc
  TestRDMAServerMCast.h
  TestRDMAServer.cc
  TestRDMAServer.h
  TestRDMAServerMultClients.cc
  TestRDMAServerMultClients.h
  TestProtoServer.cc
  TestProtoServer.h 
  TestSimpleUD.h 
  TestSimpleUD.cc
  ) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/
  
add_library(test_rdma_proto ${TEST_NET_SRC})

target_include_directories(test_rdma_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_rdma_proto rdma_proto)
