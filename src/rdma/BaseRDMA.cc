

#include "BaseRDMA.h"
#include "../memory/MainMemory.h"
#include "../memory/CudaMemory.h"
#include "../message/ProtoMessageFactory.h"
#include "../utils/Logging.h"
#include "ReliableRDMA.h"
#include "UnreliableRDMA.h"

#ifdef LINUX
#include <numa.h>
#include <numaif.h>
#endif

#ifndef HUGEPAGE
#define HUGEPAGE false
#endif

using namespace rdma;

//------------------------------------------------------------------------------------//

BaseRDMA::BaseRDMA(BaseMemory *buffer) : BaseRDMA(buffer, false){}
BaseRDMA::BaseRDMA(BaseMemory *buffer, bool pass_buffer_ownership) {
  m_buffer = buffer;
  m_buffer_owner = pass_buffer_ownership;
}

BaseRDMA::BaseRDMA(size_t mem_size) : BaseRDMA(mem_size, HUGEPAGE){}
BaseRDMA::BaseRDMA(size_t mem_size, bool huge) : BaseRDMA(mem_size, huge, (int)Config::RDMA_NUMAREGION){}
BaseRDMA::BaseRDMA(size_t mem_size, int numaNode) : BaseRDMA(mem_size, HUGEPAGE, numaNode){}
BaseRDMA::BaseRDMA(size_t mem_size, bool huge, int numaNode) : BaseRDMA(new MainMemory(mem_size, huge, numaNode), true){}
BaseRDMA::BaseRDMA(size_t mem_size, MEMORY_TYPE mem_type) : BaseRDMA(mem_size, (int)mem_type, HUGEPAGE, (int)Config::RDMA_NUMAREGION){}
BaseRDMA::BaseRDMA(size_t mem_size, MEMORY_TYPE mem_type, bool huge, int numaNode) : BaseRDMA(mem_size, (int)mem_type, huge, numaNode){}
BaseRDMA::BaseRDMA(size_t mem_size, int mem_type, bool huge, int numaNode) : BaseRDMA(
  (mem_type <= (int)MEMORY_TYPE::MAIN ? (BaseMemory*)new MainMemory(mem_size, huge, numaNode) : 
#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
  (BaseMemory*)new CudaMemory(mem_size, mem_type)
#else
  (BaseMemory*)new MainMemory(mem_size, huge, numaNode)
#endif
 ), true){}

//------------------------------------------------------------------------------------//

BaseRDMA::~BaseRDMA(){
  if(m_buffer_owner){
    delete m_buffer;
  }
}

//------------------------------------------------------------------------------------//

void BaseRDMA::createCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq) {
  // send queue
  if (!(send_cq = ibv_create_cq(m_buffer->ib_context(), Config::RDMA_MAX_WR + 1, nullptr, nullptr, 0))) {
    throw runtime_error("Cannot create send CQ!");
  }

  // receive queue
  if (!(rcv_cq = ibv_create_cq(m_buffer->ib_context(), Config::RDMA_MAX_WR + 1, nullptr, nullptr, 0))) {
    throw runtime_error("Cannot create receive CQ!");
  }

  Logging::debug(__FILE__, __LINE__, "Created send and receive CQs!");
}

//------------------------------------------------------------------------------------//

void BaseRDMA::destroyCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq) {
  auto err = ibv_destroy_cq(send_cq);
  if (err == EBUSY) {
    Logging::info(
        "Could not destroy send queue in destroyCQ(): One or more Work "
        "Queues is still associated with the CQ");
  } else if (err != 0) {
    throw runtime_error("Cannot delete send CQ. errno: " + to_string(err));
  }

  err = ibv_destroy_cq(rcv_cq);
  if (err == EBUSY) {
    Logging::info(
        "Could not destroy receive queue in destroyCQ(): One or more Work "
        "Queues is still associated with the CQ");
  } else if (err != 0) {
    throw runtime_error("Cannot delete receive CQ. errno: " + to_string(err));
  }
}

//------------------------------------------------------------------------------------//

void BaseRDMA::setQP(const rdmaConnID rdmaConnID, ib_qp_t &qp) {
  if (m_qps.size() < rdmaConnID + 1) {
    m_qps.resize(rdmaConnID + 1);
    m_countWR.resize(rdmaConnID + 1);
  }
  m_qps[rdmaConnID] = qp;
  m_qpNum2connID[qp.qp->qp_num] = rdmaConnID;
}

//------------------------------------------------------------------------------------//

void BaseRDMA::setLocalConnData(const rdmaConnID rdmaConnID, ib_conn_t &conn) {
  if (m_lconns.size() < rdmaConnID + 1) {
    m_lconns.resize(rdmaConnID + 1);
  }
  m_lconns[rdmaConnID] = conn;
}

//------------------------------------------------------------------------------------//

void BaseRDMA::internalFree(const size_t &offset) { m_buffer->free(offset); }

//------------------------------------------------------------------------------------//

rdma_mem_t BaseRDMA::internalAlloc(const size_t &size) { return m_buffer->internalAlloc(size); }

//------------------------------------------------------------------------------------//

void BaseRDMA::printBuffer() { m_buffer->printBuffer(); }

//------------------------------------------------------------------------------------//

void BaseRDMA::setRemoteConnData(const rdmaConnID rdmaConnID, ib_conn_t &conn) {
  if (m_rconns.size() < rdmaConnID + 1) {
    m_rconns.resize(rdmaConnID + 1);
  }
  m_rconns[rdmaConnID] = conn;
}