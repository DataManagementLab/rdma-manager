#include "./Memory.h"
#include "../utils/Filehelper.h"
#include "../utils/GpuNumaUtils.h"
#include "../utils/Logging.h"
#include "../utils/RandomHelper.h"

#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <vector>

#ifndef HUGEPAGE
#define HUGEPAGE false
#endif


using namespace rdma;


// CHILD memory
Memory::Memory(Memory* parent, size_t offset, size_t memSize) : mainMem(parent->mainMem), parent(parent), huge(parent->huge){
    this->buffer = (void*)((size_t)parent->buffer + offset);
    this->memSize = memSize;
    this->numaNode = parent->numaNode;
    this->deviceIndex = parent->deviceIndex;
    this->m_ibv = parent->m_ibv;
    this->port_attr = parent->port_attr;
    this->ib_ctx = parent->ib_ctx;
}


// MAIN memory
Memory::Memory(size_t memSize) : Memory(memSize, (bool)HUGEPAGE){}
Memory::Memory(size_t memSize, bool huge) : Memory(memSize, huge, -1){}
Memory::Memory(size_t memSize, bool huge, int numaNode) : Memory(true, memSize, huge, numaNode, Config::RDMA_IBPORT){}
Memory::Memory(bool registerIbv, size_t memSize, bool huge, int numaNode, int ibPort) : memSize(memSize), mainMem(true), numaNode(numaNode), parent(nullptr), huge(huge), m_ibv(registerIbv), ib_port(ibPort) {
    this->m_rdmaMem.push_back(rdma_mem_t(memSize, true, 0));
    
    if(registerIbv) this->preInit();

    // allocate memory (same as in MemoryFactory)
    #ifdef LINUX
        if(huge){
            this->buffer = mmap(NULL, this->memSize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            madvise(this->buffer, this->memSize, MADV_HUGEPAGE);
            numa_tonode_memory(this->buffer, this->memSize, this->numaNode);
        } else {
            this->buffer = numa_alloc_onnode(this->memSize, this->numaNode);
        }
    #else
        this->buffer = std::malloc(this->memSize);
    #endif

    if (this->buffer == 0) {
        throw runtime_error("Cannot allocate memory! Requested size: " + to_string(this->memSize));
    }

    if(registerIbv){
        memset(this->buffer, 0, this->memSize);
        this->postInit();
    }
}


// GPU memory
Memory::Memory(size_t memSize, MEMORY_TYPE memoryType) : Memory(memSize, (int) memoryType, -1){}
Memory::Memory(size_t memSize, int deviceIndex) : Memory(memSize, deviceIndex, -1){}
Memory::Memory(size_t memSize, MEMORY_TYPE memoryType, int ibNuma) : Memory(true, memSize, (int)memoryType, ibNuma){}
Memory::Memory(size_t memSize, int deviceIndex, int ibNuma) : Memory(true, memSize, deviceIndex, ibNuma){}
Memory::Memory(bool registerIbv, size_t memSize, int deviceIndex, int ibNuma) : mainMem(false), parent(nullptr), huge(false), deviceIndex(deviceIndex) {
    if(this->deviceIndex < -2) throw std::invalid_argument("Memory::Memory GPU device index cannot be smaller than -2. See documentation");
    if(this->deviceIndex == -2) this->deviceIndex = GpuNumaUtils::get_cuda_device_index_by_numa();

    if(ibNuma < 0){
        ibNuma = GpuNumaUtils::get_numa_node_by_cuda_device_index(this->deviceIndex);
        if(ibNuma < 0) ibNuma = rdma::Config::RDMA_NUMAREGION;
    }
    this->numaNode = ibNuma;

    if(registerIbv) this->preInit();

    // allocate CUDA memory
    #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
    openContext();
    checkCudaError(cudaMalloc(&(this->buffer), memSize), "Memory::Memory could not allocate GPU memory\n");

    if(registerIbv){
        checkCudaError(cudaMemset(this->buffer, 0, memSize), "Memory::Memory could not set allocated GPU memory to zero\n");
        this->postInit();
    }
    closeContext();
    #else
    throw new std::runtime_error("Tried creating CUDA memory of size "+std::to_string(memSize)+" without being compiled with CUDA support enabled");
    #endif
}


// destructor
Memory::~Memory(){
    // if child memory simply free reserved space on parent
    if(this->parent != nullptr){
        this->parent->internalFree(this->buffer);
        this->parent = nullptr;
        this->buffer = nullptr;
        return;
    }

    // close IB
    if(this->m_ibv){

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

    // release memory (same as in MemoryFactory)
    if(mainMem){

        // main memory
        #ifdef LINUX
            if(this->huge){
                munmap(this->buffer, this->memSize);
            } // TODO else {
            numa_free(this->buffer, this->memSize);
            // TODO }
        #else
            std::free(this->buffer);
        #endif

    } else {

        // gpu memory
        #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
        checkCudaError(cudaFree(this->buffer), "Memory::~Memory could not free GPU memory\n");
        #endif

    }
    this->buffer = nullptr;
}



void Memory::preInit(){
    // Logging::debug(__FILE__, __LINE__, "Create memory region");

    if(!m_ibv) return; // skip if memory should not be registered with IBV

    struct ibv_device **dev_list = nullptr;
    struct ibv_device *ib_dev = nullptr;
    int num_devices = 0;

    // get devices
    if ((dev_list = ibv_get_device_list(&num_devices)) == nullptr) {
        throw runtime_error("Get device list failed!");
    }

    bool found = false;
    if(Config::RDMA_DEV_NAME.size() > 0) for (int i = 0; i < num_devices; i++) {
        // Choose rdma device based on the correct name
        if(Config::RDMA_DEV_NAME == std::string(dev_list[i]->name)) {
            ifstream numa_node_file;
            numa_node_file.open(std::string(dev_list[i]->ibdev_path) + "/device/numa_node");
            int numa_node = -1;
            if (numa_node_file) {
                numa_node_file >> numa_node;
            }
            if (numa_node != -1 && numa_node != numaNode) {
                Logging::warn("Device was selected even though numa_node is not the right one (device has numa_node " + std::to_string(numa_node) + ", you selected " + std::to_string(numaNode) + ")");
            }
            ib_dev = dev_list[i];
            found = true;
            break;
        }
    }
    if(!found) for (int i = 0; i < num_devices; i++) {
        // Choose rdma device based on the correct numa node
        ifstream numa_node_file;
        numa_node_file.open(std::string(dev_list[i]->ibdev_path) + "/device/numa_node");
        int numa_node = -1;
        if (numa_node_file) {
            numa_node_file >> numa_node;
        }
        if (numa_node != -1 && numa_node == numaNode)
        {
            ib_dev = dev_list[i];
            found = true;
            break;
        }
    }

    if(!found)
        Logging::warn("Did not find a device connected to specified numa node or by name: " + std::to_string(numaNode) + " / '" + Config::RDMA_DEV_NAME + "' (Set in Config::RDMA_NUMAREGION/RDMA_DEV_NAME or constructor)");
    

    if(!found && num_devices > 0) {
        // Choose first rdma device
        ib_dev = dev_list[0];
        found = true;
        Logging::info("Selected first RDMA device found" + (std::string)ib_dev->dev_name + " | " + (std::string)ib_dev->ibdev_path);
    }
    
    ibv_free_device_list(dev_list);

    if(!found) throw new std::runtime_error("No RDMA devices found!");

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


void Memory::postInit(){

    if(!m_ibv) return; // skip if memory should not be registered with IBV

    // create protected domain
    this->pd = ibv_alloc_pd(this->ib_ctx);
    if (this->pd == 0) {
        throw runtime_error("Cannot create protected domain with InfiniBand");
    }

    // register memory
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                   IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    this->mr = ibv_reg_mr(this->pd, this->buffer, this->memSize, mr_flags);
    if (this->mr == 0) {
        fprintf(stderr, "Cannot register memory(%p) for InfiniBand because error(%i): %s\n", this->buffer, errno, strerror(errno));
        throw runtime_error("Cannot register memory for InfiniBand");
    }
}


void Memory::mergeFreeMem(list<rdma_mem_t>::iterator &listIter) {
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


rdma_mem_t Memory::internalAlloc(size_t size){
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
    listIter = m_rdmaMem.begin();
    size_t total_free = 0;
    size_t num_fragments_free = 0;
    for (; listIter != m_rdmaMem.end(); ++listIter) {
        rdma_mem_t memRes = *listIter;
        if (memRes.free)
        {
            num_fragments_free++;
            total_free += memRes.size;
        }
    }
    lock.unlock();
    Logging::warn("Memory out of local memory, requested: " + to_string(size) + " total free: " + to_string(total_free)+ " in " + to_string(num_fragments_free) +" fragment(s).");
    return rdma_mem_t();  // nullptr
}


void Memory::printBuffer() {
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


void Memory::internalFree(const void* ptr){
    char *begin = (char *)buffer;
    char *end = (char *)ptr;
    size_t offset = end - begin;
    internalFree(offset);
}


void Memory::internalFree(const size_t &offset){
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










bool Memory::isChild(){
    return this->parent != nullptr;
}

bool Memory::isRoot(){
    return this->parent == nullptr;
}

size_t Memory::getRootOffset(){
    Memory *root = this;
    while(root->parent != nullptr) root = root->getParent();
    return (size_t)this->pointer() - (size_t) root->pointer();
}

Memory* Memory::getParent(){
    return this->parent;
}

size_t Memory::getSize(){
    return this->memSize;
}

void Memory::setSize(size_t memSize){
    this->memSize = memSize;
}

void* Memory::pointer(){
    return this->buffer;
}

void* Memory::pointer(size_t offset){
    return ((char*)buffer + offset);
}

bool Memory::isMainMemory(){
    return this->mainMem;
}

bool Memory::isGPUMemory(){
    return !this->mainMem;
}

int Memory::getNumaNode(){
    return this->numaNode;
}


bool Memory::isHuge(){
    return this->huge;
}


int Memory::getDeviceIndex(){
    return this->deviceIndex;
}


bool Memory::isIBV(){
    return this->m_ibv;
}

int Memory::getIBPort(){
    return this->ib_port;
}

ibv_pd* Memory::ib_pd(){
    return this->pd;
}

ibv_mr* Memory::ib_mr(){
    return this->mr;
}

ibv_port_attr Memory::ib_port_attributes(){
    return this->port_attr;
}

ibv_context* Memory::ib_context(){
    return this->ib_ctx;
}

Memory* Memory::malloc(size_t memSize){
    rdma_mem_t memRes = internalAlloc(memSize);
    if(!memRes.isnull)
        return new Memory(this, memRes.offset, memSize);
    throw runtime_error(std::string("Memory::alloc: Could not allocate child memory, requested size: %lu", memSize));
}

std::string Memory::toString(){
    return this->toString(0, this->memSize);
}

std::string Memory::toString(size_t offset, size_t count){
    std::ostringstream oss;
    oss << "[";
    if(this->mainMem){
        // main memory
        char* arr = (char*) this->buffer;
        bool next = false;
        for(size_t i=0; i < count; i++){
            if(next){ oss << ", "; } else { next = true; }
            oss << ((int)arr[i + offset]);
        }
    } else {
        // gpu memory
        #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
        char* arr = (char*)std::malloc(count * sizeof(char)); // allocate in main memory
        copyTo(arr, 0, offset, count, MEMORY_TYPE::MAIN); // copy GPU memory to main memory

        // print 
        bool next = false;
        for(size_t i=0; i < count; i++){
            if(next){ oss << ", "; } else { next = true; }
            oss << ((int)arr[i + offset]);
        }

        std::free(arr); // release main memory again
        #else
        oss << " not compiled with CUDA, remove NO_CUDA flag ";
        #endif
    }
    oss << "]";
    return oss.str();
}

void Memory::print(){
    print(0, this->memSize);
}

void Memory::print(size_t offset, size_t count){
    std::cout << toString(offset, count) << std::endl;
}

void Memory::openContext(){
    if(this->mainMem) return;

    // gpu memory
    #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
    if(this->deviceIndex < 0) return;
    this->openContextCounter++;
    if(this->openContextCounter > 1) return;
    this->previousDeviceIndex = -1;
    checkCudaError(cudaGetDevice(&(this->previousDeviceIndex)), "Memory::openContext could not get selected GPU\n");
    if(this->previousDeviceIndex == this->deviceIndex) return;
    checkCudaError(cudaSetDevice(this->deviceIndex), "Memory::openContext could not set selected GPU\n");
    #endif
}

void Memory::closeContext(){
    if(this->mainMem) return;

    // gpu memory
    #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
    if(this->deviceIndex < 0 || this->openContextCounter==0) return;
    this->openContextCounter--;
    if(this->openContextCounter > 0) return;
    if(this->deviceIndex != this->previousDeviceIndex)
        checkCudaError(cudaSetDevice(this->previousDeviceIndex), "Memory::closeContext could not reset selected GPU\n");
    this->previousDeviceIndex = -1;
    #endif
}

void Memory::setRandom(){
    setRandom(0, this->memSize);
}

void Memory::setRandom(size_t offset, size_t count){
    if(this->mainMem){
        // main memory
        RandomHelper::randomizeMemory((char*)this->buffer, 0, this->memSize);
    } else {
        // gpu memory
        std::vector<uint8_t> rd = RandomHelper::generateRandomVector(count);
        this->copyFrom((void*)rd.data(), 0, offset, count, rdma::MEMORY_TYPE::MAIN);
        delete rd.data();
    }
}

void Memory::setMemory(int value){
    setMemory(value, 0, this->memSize);
}

void Memory::setMemory(int value, size_t count){
    setMemory(value, 0, count);
}

void Memory::setMemory(int value, size_t offset, size_t count){
    if(this->mainMem){
        // main memory
        memset((void*)((size_t)this->buffer + offset), value, count);
    } else {
        // gpu memory
        #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
        checkCudaError(cudaMemset((void*)((size_t)this->buffer + offset), value, count), "Memory::setMemory could not set GPU memory");
        #endif
    }
}

void Memory::copyTo(void *destination, MEMORY_TYPE memType){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->memSize ? s : this->memSize), memType);
}

void Memory::copyTo(void *destination, size_t count, MEMORY_TYPE memType){
    copyTo(destination, 0, 0, count, memType);
}

void Memory::copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t count, MEMORY_TYPE memType){
    destination = (void*)((size_t)destination + destOffset);
    void* source = (void*)((size_t)this->buffer + srcOffset);
    if(this->mainMem){
        // main memory (this)

        if((int)memType <= (int)MEMORY_TYPE::MAIN){
            // main memory (destination)
            memcpy(destination, source, count);

        } else {
            // gpu memory (destination)
            #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
            checkCudaError(cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice), "Memory::copyTo could not copy from MAIN to GPU");
            #endif
        }

    } else {
        // gpu memory (this)
        #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */

        if((int)memType <= (int)MEMORY_TYPE::MAIN){
            // main memory (destination)
            checkCudaError(cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost), "Memory::copyTo could not copy from GPU to MAIN");

        } else {
            // gpu memory (destination)
            checkCudaError(cudaMemcpy(destination, source, count, cudaMemcpyDeviceToDevice), "Memory::copyTo could not copy from GPU to GPU");

        }

        #endif
    }
}

void Memory::copyFrom(const void *source, MEMORY_TYPE memType){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->memSize ? s : this->memSize), memType);
}

void Memory::copyFrom(const void *source, size_t count, MEMORY_TYPE memType){
    copyFrom(source, 0, 0, count, memType);
}

void Memory::copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t count, MEMORY_TYPE memType){
    source = (void*)((size_t)source + srcOffset);
    void* destination = (void*)((size_t)this->buffer + destOffset);
    if(this->mainMem){
        // main memory (this)

        if((int)memType <= (int)MEMORY_TYPE::MAIN){
            // main memory (source)
            memcpy(destination, source, count);

        } else {
            // gpu memory (source)
            #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */
            checkCudaError(cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost), "Memory::copyFrom could not copy from GPU to MAIN");
            #endif
        }

    } else {
        // gpu memory (this)
        #ifndef NO_CUDA /* defined in CMakeLists.txt to globally enable/disable CUDA support */

        if((int)memType <= (int)MEMORY_TYPE::MAIN){
            // main memory (source)
            checkCudaError(cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice), "Memory::copyFrom could not copy from MAIN to GPU");

        } else {
            // gpu memory (source)
            checkCudaError(cudaMemcpy(destination, source, count, cudaMemcpyDeviceToDevice), "Memory::copyFrom could not copy from GPU to GPU");

        }

        #endif
    }
}


char Memory::getChar(size_t offset){
    // main memory
    if(this->mainMem) return ((char*)this->buffer)[offset];

    // gpu memory
    char tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(char value, size_t offset){
    // main memory
    if(this->mainMem){ ((char*)this->buffer)[offset] = value; return; }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

int8_t Memory::getInt8(size_t offset){
    // main memory
    if(this->mainMem) return *(int8_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    int8_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(int8_t value, size_t offset){
    // main memory
    if(this->mainMem){
        int8_t *tmp = (int8_t*)((size_t)this->buffer + offset);
        *tmp = value; return;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

uint8_t Memory::getUInt8(size_t offset){
    // main memory
    if(this->mainMem) return *(uint8_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    uint8_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(uint8_t value, size_t offset){
    // main memory
    if(this->mainMem){
        uint8_t *tmp = (uint8_t*)((size_t)this->buffer + offset);
        *tmp = value; return;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

int16_t Memory::getInt16(size_t offset){
    // main memory
    if(this->mainMem) return *(int16_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    int16_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(int16_t value, size_t offset){
    // main memory
    if(this->mainMem){
        int16_t *tmp = (int16_t*)((size_t)this->buffer + offset);
        *tmp = value; return;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

uint16_t Memory::getUInt16(size_t offset){
    // main memory
    if(this->mainMem) return *(uint16_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    uint16_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(uint16_t value, size_t offset){
    // main memory
    if(this->mainMem){
        uint16_t *tmp = (uint16_t*)((size_t)this->buffer + offset);
        *tmp = value; return;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

int32_t Memory::getInt32(size_t offset){
    // main memory
    if(this->mainMem) return *(int32_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    int32_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(int32_t value, size_t offset){
    // main memory
    if(this->mainMem){
        int32_t *tmp = (int32_t*)((size_t)this->buffer + offset);
        *tmp = value;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

uint32_t Memory::getUInt32(size_t offset){
    // main memory
    if(this->mainMem) return *(uint32_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    uint32_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(uint32_t value, size_t offset){
    // main memory
    if(this->mainMem){
        uint32_t *tmp = (uint32_t*)((size_t)this->buffer + offset);
        *tmp = value; return;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

int64_t Memory::getInt64(size_t offset){
    // main memory
    if(this->mainMem) return *(int64_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    int64_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(int64_t value, size_t offset){
    // main memory
    if(this->mainMem){
        int64_t *tmp = (int64_t*)((size_t)this->buffer + offset);
        *tmp = value;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}

uint64_t Memory::getUInt64(size_t offset){
    // main memory
    if(this->mainMem) return *(int64_t*)((size_t)this->buffer + offset);
    
    // gpu memory
    uint64_t tmp[1];
    copyTo((void*)tmp, 0, offset, sizeof(tmp), MEMORY_TYPE::MAIN);
    return tmp[0];
}

void Memory::set(uint64_t value, size_t offset){
    // main memory
    if(this->mainMem){
        uint64_t *tmp = (uint64_t*)((size_t)this->buffer + offset);
        *tmp = value;
    }
    
    // gpu memory
    copyFrom((void*)&value, 0, offset, sizeof(value), MEMORY_TYPE::MAIN);
}