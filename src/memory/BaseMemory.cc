#include "BaseMemory.h"
#include "MainMemory.h"
#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
#include "CudaMemory.h"
#endif /* CUDA support */

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

rdma_mem_t BaseMemory::s_nillmem;

// constructor
BaseMemory::BaseMemory(size_t mem_size, int ib_port) : AbstractBaseMemory(mem_size){
    this->ib_port = ib_port;
    this->m_rdmaMem.push_back(rdma_mem_t(mem_size, true, 0));

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
        // choose rdma device on the correct numa node
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
    ibv_free_device_list(dev_list);
    if (!found){
        throw runtime_error("Did not find a device connected to specified numa node (Config::RDMA_NUMAREGION)");
    }
    Config::RDMA_DEVICE_FILE_PATH = ib_dev->ibdev_path;

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
}


void BaseMemory::init(){
    // create protected domain
    this->pd = ibv_alloc_pd(this->ib_ctx);
    if (this->pd == 0) {
        throw runtime_error("Cannot create protected domain with InfiniBand");
    }

    // register memory
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                   IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    this->mr = ibv_reg_mr(this->pd, this->buffer, this->mem_size, mr_flags);
    if (this->mr == 0) {
        fprintf(stderr, "Cannot register memory(%p) for InfiniBand because error(%i): %s\n", this->buffer, errno, strerror(errno));
        throw runtime_error("Cannot register memory for InfiniBand");
    }
}

BaseMemory::~BaseMemory(){
    // de-register memory region
    if (this->mr != nullptr){
        if(ibv_dereg_mr(this->mr))
            fprintf(stderr, "Could not deregister memory from InfiniBand\n");
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

void BaseMemory::mergeFreeMem(list<rdma_mem_t>::iterator &listIter) {
    std::unique_lock<std::recursive_mutex> lock(m_lockMem);
    size_t freeSpace = (*listIter).size;
    size_t offset = (*listIter).offset;
    size_t size = (*listIter).size;

    // start with the prev
    if (listIter != m_rdmaMem.begin()) {
        --listIter;
        if(listIter->offset + listIter->size == offset) {
            // increase mem of prev
            freeSpace += listIter->size;
            (*listIter).size = freeSpace;

            // delete hand-in el
            listIter++;
            listIter = m_rdmaMem.erase(listIter);
            listIter--;
        } else {
            // adjust iter to point to hand-in el
            listIter++;
        }
    }
    // now check following
    ++listIter;
    if (listIter != m_rdmaMem.end()) {
        if(offset + size == listIter->offset) {
            freeSpace += listIter->size;

            // delete following
            listIter = m_rdmaMem.erase(listIter);

            // go to previous and extend
            --listIter;
            (*listIter).size = freeSpace;
        }
    }
    Logging::debug(
        __FILE__, __LINE__,
        "Merged consecutive free RDMA memory regions, total free space: " +
            to_string(freeSpace));
    lock.unlock();
}

rdma_mem_t BaseMemory::internalAlloc(size_t size){
    std::unique_lock<std::recursive_mutex> lock(m_lockMem);
    auto listIter = m_rdmaMem.begin();
    for (; listIter != m_rdmaMem.end(); ++listIter) {
        rdma_mem_t memRes = *listIter;
        if (memRes.free && memRes.size >= size) {
            rdma_mem_t memResUsed(size, false, memRes.offset);
            m_usedRdmaMem[memRes.offset] = memResUsed;

            if (memRes.size > size) {
                rdma_mem_t memResFree(memRes.size - size, true, memRes.offset + size);
                m_rdmaMem.insert(listIter, memResFree);
            }
            m_rdmaMem.erase(listIter);
            lock.unlock();
            return memResUsed;
        }
    }
    lock.unlock();
    Logging::warn("BaseMemory out of local memory");
    return rdma_mem_t();  // nullptr
}

void BaseMemory::printBuffer() {
    std::unique_lock<std::recursive_mutex> lock(m_lockMem);
    std::cout << "Free Buffer(" << m_rdmaMem.size() << ")=[";
    bool next = false;
    auto listIter = m_rdmaMem.begin();
    for (; listIter != m_rdmaMem.end(); ++listIter) {
        if(next){ std::cout << ", "; } else { next=true; }
        std::cout << "(offset=" << to_string((*listIter).offset) << "; size=" << to_string((*listIter).size) << "; free=" << to_string((*listIter).free) << ")";
        Logging::debug(__FILE__, __LINE__,
                    "offset=" + to_string((*listIter).offset) + "," +
                        "size=" + to_string((*listIter).size) + "," +
                        "free=" + to_string((*listIter).free));
    }
    std::cout << "]" << std::endl;
    Logging::debug(__FILE__, __LINE__, "---------");

    next = false;
    std::cout << "Used Buffer(" << m_usedRdmaMem.size() << ")=[";
    for(auto &mapIter : m_usedRdmaMem){
        if(next){ std::cout << ", "; } else { next=true; }
        auto info = mapIter.second;
        std::cout << "(offset=" << to_string(info.offset) << "; size=" << to_string(info.size) << "; free=" << to_string(info.free) << ")";
    }
    std::cout << "]" << std::endl;

    lock.unlock();
}

void* BaseMemory::alloc(size_t size){
    rdma_mem_t memRes = internalAlloc(size);
    if (!memRes.isnull) {
        return (void *)((char *)buffer + memRes.offset);
    }
    throw runtime_error("Could not allocate local rdma memory");
}

void BaseMemory::free(const void* ptr){
    char *begin = (char *)buffer;
    char *end = (char *)ptr;
    size_t offset = end - begin;
    free(offset);
}

void BaseMemory::free(const size_t &offset){
    std::unique_lock<std::recursive_mutex> lock(m_lockMem);
    size_t lastOffset = 0;
    rdma_mem_t memResFree = m_usedRdmaMem[offset];

    m_usedRdmaMem.erase(offset);
    // std::cout << "offset: " << offset << " m_rdmaMem.size() " << m_rdmaMem.size() << std::endl;
    // lookup the memory region that was assigned to this pointer
    auto listIter = m_rdmaMem.begin();
    if (listIter != m_rdmaMem.end()) {
        for (; listIter != m_rdmaMem.end(); listIter++) {
            rdma_mem_t &memRes = *(listIter);
            if (lastOffset <= offset && offset < memRes.offset) {
                memResFree.free = true;
                m_rdmaMem.insert(listIter, memResFree);
                listIter--;
                Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");

                mergeFreeMem(listIter);
                lock.unlock();
                return;
            }
            lastOffset += memRes.size;
        }

        // added because otherwise not able to append largest offset at end
        if(lastOffset <= offset){
            memResFree.free = true;
            m_rdmaMem.insert(listIter, memResFree);
            listIter--;
            mergeFreeMem(listIter);
            lock.unlock();
            return;
        }
        
    } else {
        memResFree.free = true;
        m_rdmaMem.insert(listIter, memResFree);
        Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");
        lock.unlock();
        return;
    }
    lock.unlock();
    throw runtime_error("Did not free any internal memory!");
}