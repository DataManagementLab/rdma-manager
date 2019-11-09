

#include "RDMAManager.h"
#include "RDMAManagerRC.h"
#include "RDMAManagerUD.h"
#include "../utils/Logging.h"
#include "../message/MessageTypes.h"

#ifdef LINUX
#include <numa.h>
#endif

using namespace rdma;

rdma_mem_t RDMAManager::s_nillmem;


//------------------------------------------------------------------------------------//

RDMAManager::RDMAManager(size_t mem_size)
{
  m_memSize = mem_size;
  m_numaRegion = Config::RDMA_NUMAREGION;
  m_rdmaDevice = Config::RDMA_DEVICE;
  m_ibPort = Config::RDMA_IBPORT;
  m_gidIdx = -1;
  m_rdmaMem.push_back(rdma_mem_t(m_memSize, true, 0));
  m_lastConnKey = 0;

  if (!createBuffer())
  {
    throw invalid_argument("RDMA buffer could not be created!");
  }
}

//------------------------------------------------------------------------------------//

RDMAManager::~RDMAManager(){
    // destroy QPS
  destroyQPs();
  m_qps.clear();

  //de-register memory region
  if (m_res.mr != nullptr)
  {
    ibv_dereg_mr(m_res.mr);
    m_res.mr = nullptr;
  }

  // free memory
  if (m_res.buffer != nullptr)
  {
#ifdef LINUX
    numa_free(m_res.buffer, m_memSize);
#else
    free(m_res.buffer);
#endif
    m_res.buffer = nullptr;
  }

  // de-allocate protection domain
  if (m_res.pd != nullptr)
  {
    ibv_dealloc_pd(m_res.pd);
    m_res.pd = nullptr;
  }

  // close device
  if (m_res.ib_ctx != nullptr)
  {
    ibv_close_device(m_res.ib_ctx);
    m_res.ib_ctx = nullptr;
  }
}

//------------------------------------------------------------------------------------//
bool RDMAManager::createBuffer()
{
  //Logging::debug(__FILE__, __LINE__, "Create memory region");

  struct ibv_device **dev_list = nullptr;
  struct ibv_device *ib_dev = nullptr;
  int num_devices = 0;

  //get devices
  if ((dev_list = ibv_get_device_list(&num_devices)) == nullptr)
  {
    Logging::error(__FILE__, __LINE__, "Get device list failed!");
    return false;
  }

  if (m_rdmaDevice >= num_devices)
  {
    Logging::error(__FILE__, __LINE__, "Device not present!");
    ibv_free_device_list(dev_list);
    return false;
  }

  ib_dev = dev_list[m_rdmaDevice];
  ibv_free_device_list(dev_list);

  // open device
  if (!(m_res.ib_ctx = ibv_open_device(ib_dev)))
  {
    Logging::error(__FILE__, __LINE__, "Open device failed");
    return false;
  }

  // get port properties
  if ((errno = ibv_query_port(m_res.ib_ctx, m_ibPort, &m_res.port_attr)) != 0)
  {
    Logging::error(__FILE__, __LINE__, "Query port failed");
    return false;
  }

//allocate memory
#ifdef LINUX
  m_res.buffer = numa_alloc_onnode(m_memSize, m_numaRegion);
#else
  m_res.buffer = malloc(m_memSize);
#endif
  memset(m_res.buffer, 0, m_memSize);
  if (m_res.buffer == 0)
  {
    Logging::error(__FILE__, __LINE__, "Cannot allocate memory! Requested size: " + to_string(m_memSize));
    return false;
  }

  //create protected domain
  m_res.pd = ibv_alloc_pd(m_res.ib_ctx);
  if (m_res.pd == 0)
  {
    Logging::error(__FILE__, __LINE__, "Cannot create protected domain!");
    return false;
  }

  //register memory
  int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  m_res.mr = ibv_reg_mr(m_res.pd, m_res.buffer, m_memSize, mr_flags);
  if (m_res.mr == 0)
  {
    Logging::error(__FILE__, __LINE__, "Cannot register memory!");
    return false;
  }

  //Logging:debug(__FILE__, __LINE__, "Created memory region!");

  return true;
}

//------------------------------------------------------------------------------------//

bool RDMAManager::createCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq)
{
  //send queue
  if (!(send_cq = ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1,
                                nullptr, nullptr, 0)))
  {
    Logging::error(__FILE__, __LINE__, "Cannot create send CQ!");
    return false;
  }

  //receive queue
  if (!(rcv_cq = ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1, nullptr,
                               nullptr, 0)))
  {
    Logging::error(__FILE__, __LINE__, "Cannot create receive CQ!");
    return false;
  }

  Logging::debug(__FILE__, __LINE__, "Created send and receive CQs!");
  return true;
}


//------------------------------------------------------------------------------------//

bool RDMAManager::destroyCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq)
{
  auto err = ibv_destroy_cq(send_cq);
  if (err != 0)
  {
    Logging::error(__FILE__, __LINE__, "Cannot delete send CQ. errno: " + to_string(err));
    return false;
  }

  err = ibv_destroy_cq(rcv_cq);
  if (err == EBUSY)
  {
    Logging::info("Could not destroy receive queue in destroyCQ(): One or more Work Queues is still associated with the CQ");
  }
  else if(err != 0)
  {
    Logging::error(__FILE__, __LINE__, "Cannot delete receive CQ. errno: " + to_string(err));
    return false;
  }
  
  return true;
}

//------------------------------------------------------------------------------------//

    void RDMAManager::setQP(ib_addr_t& ibAddr, ib_qp_t& qp) {
        size_t connKey = ibAddr.conn_key;
        if (m_qps.size() < connKey + 1) {
            m_qps.resize(connKey + 1);
        }
        m_qps[connKey] = qp;
        m_qpNum2connKey[qp.qp->qp_num] = connKey;
    }

//------------------------------------------------------------------------------------//

    void RDMAManager::setLocalConnData(ib_addr_t& ibAddr, ib_conn_t& conn) {
        size_t connKey = ibAddr.conn_key;
        if (m_lconns.size() < connKey + 1) {
            m_lconns.resize(connKey + 1);
        }
        m_lconns[connKey] = conn;
    }

//------------------------------------------------------------------------------------//

    bool RDMAManager::internalFree(const size_t& offset) {
        size_t lastOffset = 0;
        rdma_mem_t memResFree = m_usedRdmaMem[offset];
        m_usedRdmaMem.erase(offset);

        // lookup the memory region that was assigned to this pointer
        auto listIter = m_rdmaMem.begin();
        if (listIter != m_rdmaMem.end()) {
            for (; listIter != m_rdmaMem.end(); listIter++) {
                rdma_mem_t& memRes = *(listIter);
                if (lastOffset <= offset && offset < memRes.offset) {
                    memResFree.free = true;
                    m_rdmaMem.insert(listIter, memResFree);
                    listIter--;
                    Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");
                    //printMem();
                    mergeFreeMem(listIter);
                    //printMem();

                    return true;

                }
                lastOffset += memRes.offset;
            }
        } else {
            memResFree.free = true;
            m_rdmaMem.insert(listIter, memResFree);
            Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");
            //printMem();
            return true;

        }
        //printMem();
        return false;
    }

//------------------------------------------------------------------------------------//

    rdma_mem_t RDMAManager::internalAlloc(const size_t& size) {
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
                //printMem();
                return memResUsed;
            }
        }
        //printMem();
        return rdma_mem_t();  //nullptr
    }

//------------------------------------------------------------------------------------//

        bool RDMAManager::mergeFreeMem(list<rdma_mem_t>::iterator& iter) {
        size_t freeSpace = (*iter).size;
        size_t offset = (*iter).offset;
        size_t size = (*iter).size;

        // start with the prev
        if (iter != m_rdmaMem.begin()) {
            --iter;
            if (iter->offset + iter->size == offset) {
                //increase mem of prev
                freeSpace += iter->size;
                (*iter).size = freeSpace;

                //delete hand-in el
                iter++;
                iter = m_rdmaMem.erase(iter);
                iter--;
            } else {
                //adjust iter to point to hand-in el
                iter++;
            }

        }
        // now check following
        ++iter;
        if (iter != m_rdmaMem.end()) {
            if (offset + size == iter->offset) {
                freeSpace += iter->size;

                //delete following
                iter = m_rdmaMem.erase(iter);

                //go to previous and extend
                --iter;
                (*iter).size = freeSpace;
            }
        }
        Logging::debug(
                __FILE__,
                __LINE__,
                "Merged consecutive free RDMA memory regions, total free space: "
                        + to_string(freeSpace));
        return true;
    }

//------------------------------------------------------------------------------------//

    void RDMAManager::printBuffer() {
        auto listIter = m_rdmaMem.begin();
        for (; listIter != m_rdmaMem.end(); ++listIter) {
            Logging::debug(
                    __FILE__,
                    __LINE__,
                    "offset=" + to_string((*listIter).offset) + "," + "size="
                            + to_string((*listIter).size) + "," + "free="
                            + to_string((*listIter).free));
        }
        Logging::debug(__FILE__, __LINE__, "---------");
    }

//------------------------------------------------------------------------------------//

        void RDMAManager::setRemoteConnData(ib_addr_t& ibAddr, ib_conn_t& conn) {
        size_t connKey = ibAddr.conn_key;
        if (m_rconns.size() < connKey + 1) {
            m_rconns.resize(connKey + 1);
        }
        m_rconns[connKey] = conn;
    }