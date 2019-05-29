enable_testing()
include(GNUInstallDirs)

add_subdirectory(message)
add_subdirectory(proto)
add_subdirectory(rdma)
add_subdirectory(thread)
add_subdirectory(utils)


add_library(rdma_lib)

target_link_libraries(rdma_lib 
    net_rdma
)