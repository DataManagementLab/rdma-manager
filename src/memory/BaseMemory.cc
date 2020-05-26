#include "BaseMemory.h"
#include "MainMemory.h"
#include "CudaMemory.h"
#include "../utils/Logging.h"
#include "../utils/Filehelper.h"

#include <infiniband/verbs.h>
// #include <infiniband/verbs_exp.h>
#include <stdio.h>

#ifdef LINUX
#include <numa.h>
#include <numaif.h>
#endif

using namespace rdma;

// constructor
BaseMemory::BaseMemory(size_t mem_size, int ib_port){
    this->mem_size = mem_size;
    this->ib_port = ib_port;
}

BaseMemory::~BaseMemory(){
    // de-register memory region
    if (this->mr != nullptr){
        ibv_dereg_mr(this->mr);
        this->mr = nullptr;
    }

    // de-allocate protection domain
    if (this->pd != nullptr){
        ibv_dealloc_pd(this->pd);
        this->pd = nullptr;
    }

    // close device
    if (this->ib_ctx != nullptr){
        ibv_close_device(this->ib_ctx);
        this->ib_ctx = nullptr;
    }
}

size_t BaseMemory::getSize(){
    return this->mem_size;
}

void* BaseMemory::pointer(){
    return this->buffer;
}

int BaseMemory::getIBPort(){
    return this->ib_port;
}

ibv_pd* BaseMemory::ib_pd(){
    return this->pd;
}

ibv_mr* BaseMemory::ib_mr(){
    return this->mr;
}

ibv_port_attr BaseMemory::ib_port_attributes(){
    return this->port_attr;
}

ibv_context* BaseMemory::ib_context(){
    return this->ib_ctx;
}

void BaseMemory::init(){
    // Logging::debug(__FILE__, __LINE__, "Create memory region");

    struct ibv_device **dev_list = nullptr;
    struct ibv_device *ib_dev = nullptr;
    int num_devices = 0;

    // get devices
    if ((dev_list = ibv_get_device_list(&num_devices)) == nullptr) {
        throw runtime_error("Get device list failed!");
    }

    bool found = false;
    //Choose rdma device on the correct numa node
    for (int i = 0; i < num_devices; i++) {
        ifstream numa_node_file;
        numa_node_file.open(std::string(dev_list[i]->ibdev_path)+"/device/numa_node");
        int numa_node = -1;
        numa_node_file >> numa_node;
        if (numa_node == (int)Config::RDMA_NUMAREGION) {
            ib_dev = dev_list[i];
            found = true;
            break;
        }
    }
    Config::RDMA_DEVICE_FILE_PATH = ib_dev->ibdev_path;
    ibv_free_device_list(dev_list);

    if (!found){
        throw runtime_error("Did not find a device connected to specified numa node (Config::RDMA_NUMAREGION)");
    }

    if (!Filehelper::isDirectory(Config::RDMA_DEVICE_FILE_PATH + "/device/net/" + Config::RDMA_INTERFACE)){
        Logging::error(__FILE__, __LINE__, "rdma::Config::RDMA_INTERFACE (" + Config::RDMA_INTERFACE + ") does not match chosen RDMA device! I.e. interface not found under: " + Config::RDMA_DEVICE_FILE_PATH + "/device/net/");
    }
    // std::cout << ib_dev->ibdev_path << std::endl;
    // std::cout << ib_dev->dev_name << std::endl;
    // std::cout << ib_dev->name << std::endl;
    // std::cout << ib_dev->dev_path << std::endl;

    // open device
    if (!(this->ib_ctx = ibv_open_device(ib_dev))) {
        throw runtime_error("Open device failed!");
    }

    // get port properties
    if ((errno = ibv_query_port(this->ib_ctx, this->ib_port, &this->port_attr)) != 0) {
        throw runtime_error("Query port failed");
    }

    // create protected domain
    this->pd = ibv_alloc_pd(this->ib_ctx);
    if (this->pd == 0) {
        throw runtime_error("Cannot create protected domain!");
    }

    // register memory
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                   IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    this->mr = ibv_reg_mr(this->pd, this->buffer, this->mem_size, mr_flags);
    if (this->mr == 0) {
        throw runtime_error("Cannot register memory!");
    }
}