enable_testing()
include(GNUInstallDirs)

add_subdirectory(message)
add_subdirectory(proto)
add_subdirectory(rdma)
add_subdirectory(thread)
add_subdirectory(utils)
add_subdirectory(RPC)


add_library(rdma_lib)

target_link_libraries(rdma_lib 
    net_rdma
)

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/RDMA.conf
    DESTINATION ${CMAKE_BINARY_DIR}/bin/conf/
)
