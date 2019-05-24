

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

/********** constructor and destructor **********/

RDMAManager::RDMAManager(size_t mem_size) {
  m_memSize = mem_size;
  m_numaRegion = Config::RDMA_NUMAREGION;
  m_rdmaDevice = Config::RDMA_DEVICE;
  m_ibPort = Config::RDMA_IBPORT;
  m_gidIdx = -1;
  m_rdmaMem.push_back(rdma_mem_t(m_memSize, true, 0));
  m_lastConnKey = 0;

  if (!createBuffer()) {
    throw invalid_argument("RDMA buffer could not be created!");
  }
}

void RDMAManager::destroyManager() {
  // destroy QPS
  destroyQPs();
  m_qps.clear();

  //de-register memory region
  if (m_res.mr != nullptr) {
    ibv_dereg_mr(m_res.mr);
    m_res.mr = nullptr;
  }

  // free memory
  if (m_res.buffer != nullptr) {
#ifdef LINUX
    numa_free(m_res.buffer, m_memSize);
#else
    free(m_res.buffer);
#endif
    m_res.buffer = nullptr;
  }

  // de-allocate protection domain
  if (m_res.pd != nullptr) {
    ibv_dealloc_pd(m_res.pd);
    m_res.pd = nullptr;
  }

  // close device
  if (m_res.ib_ctx != nullptr) {
    ibv_close_device(m_res.ib_ctx);
    m_res.ib_ctx = nullptr;
  }

}

/********** private methods **********/
bool RDMAManager::createBuffer() {
//Logging::debug(__FILE__, __LINE__, "Create memory region");

  struct ibv_device **dev_list = nullptr;
  struct ibv_device *ib_dev = nullptr;
  int num_devices = 0;

//get devices
  if ((dev_list = ibv_get_device_list(&num_devices)) == nullptr) {
    Logging::error(__FILE__, __LINE__, "Get device list failed!");
    return false;
  }

  if (m_rdmaDevice >= num_devices) {
    Logging::error(__FILE__, __LINE__, "Device not present!");
    ibv_free_device_list(dev_list);
    return false;
  }

  ib_dev = dev_list[m_rdmaDevice];
  ibv_free_device_list(dev_list);

// open device
  if (!(m_res.ib_ctx = ibv_open_device(ib_dev))) {
    Logging::error(__FILE__, __LINE__, "Open device failed");
    return false;
  }

// get port properties
  if ((errno = ibv_query_port(m_res.ib_ctx, m_ibPort, &m_res.port_attr))
      != 0) {
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
  if (m_res.buffer == 0) {
    Logging::error(__FILE__, __LINE__, "Cannot allocate memory!");
    return false;
  }

//create protected domain
  m_res.pd = ibv_alloc_pd(m_res.ib_ctx);
  if (m_res.pd == 0) {
    Logging::error(__FILE__, __LINE__, "Cannot create protected domain!");
    return false;
  }

//register memory
  int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ
  | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  m_res.mr = ibv_reg_mr(m_res.pd, m_res.buffer, m_memSize, mr_flags);
  if (m_res.mr == 0) {
    Logging::error(__FILE__, __LINE__, "Cannot register memory!");
    return false;
  }

//Logging:debug(__FILE__, __LINE__, "Created memory region!");

  return true;
}

bool RDMAManager::createCQ(ibv_cq*& send_cq, ibv_cq*& rcv_cq) {
  //send queue
  if (!(send_cq = ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1,
  nullptr, nullptr, 0))) {
    Logging::error(__FILE__, __LINE__, "Cannot create send CQ!");
    return false;
  }

  //receive queue
  if (!(rcv_cq = ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1, nullptr,
  nullptr, 0))) {
    Logging::error(__FILE__, __LINE__, "Cannot create receive CQ!");
    return false;
  }

  Logging::debug(__FILE__, __LINE__, "Created send and receive CQs!");
  return true;
}

bool RDMAManager::destroyCQ(ibv_cq*& send_cq, ibv_cq*& rcv_cq) {
  if (ibv_destroy_cq(send_cq) != 0) {
    Logging::error(__FILE__, __LINE__, "Cannot delete send CQ");
    return false;
  }

  if (ibv_destroy_cq(rcv_cq) != 0) {
    Logging::error(__FILE__, __LINE__, "Cannot delete receive CQ");
    return false;
  }
  return true;
}
