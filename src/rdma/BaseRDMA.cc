

#include "BaseRDMA.h"
#include "../memory/MainMemory.h"
#include "../message/ProtoMessageFactory.h"
#include "../utils/Logging.h"
#include "ReliableRDMA.h"
#include "UnreliableRDMA.h"

#ifdef LINUX
#include <numa.h>
#include <numaif.h>
#endif

using namespace rdma;

rdma_mem_t BaseRDMA::s_nillmem;

//------------------------------------------------------------------------------------//
BaseRDMA::BaseRDMA(BaseMemory *buffer) : m_buffer(buffer) {
  m_gidIdx = -1;
  m_rdmaMem.push_back(rdma_mem_t(m_buffer->getSize(), true, 0));
}

BaseRDMA::BaseRDMA(size_t mem_size) : BaseRDMA(new MainMemory(mem_size)) {}

BaseRDMA::BaseRDMA(size_t mem_size, bool huge) : BaseRDMA(new MainMemory(mem_size, huge)) {}

//------------------------------------------------------------------------------------//

BaseRDMA::~BaseRDMA(){

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
  if (err != 0) {
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

void BaseRDMA::internalFree(const size_t &offset) {
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
        // printMem();
        mergeFreeMem(listIter);
        // printMem();

        return;
      }
      lastOffset += memRes.offset;
    }
  } else {
    memResFree.free = true;
    m_rdmaMem.insert(listIter, memResFree);
    Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");
    // printMem();
    return;
  }
  // printMem();
  throw runtime_error("Did not free any internal memory!");
}

//------------------------------------------------------------------------------------//

rdma_mem_t BaseRDMA::internalAlloc(const size_t &size) {
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
      // printMem();
      return memResUsed;
    }
  }
  // printMem();
  return rdma_mem_t();  // nullptr
}

//------------------------------------------------------------------------------------//

void BaseRDMA::mergeFreeMem(list<rdma_mem_t>::iterator &iter) {
  size_t freeSpace = (*iter).size;
  size_t offset = (*iter).offset;
  size_t size = (*iter).size;

  // start with the prev
  if (iter != m_rdmaMem.begin()) {
    --iter;
    if (iter->offset + iter->size == offset) {
      // increase mem of prev
      freeSpace += iter->size;
      (*iter).size = freeSpace;

      // delete hand-in el
      iter++;
      iter = m_rdmaMem.erase(iter);
      iter--;
    } else {
      // adjust iter to point to hand-in el
      iter++;
    }
  }
  // now check following
  ++iter;
  if (iter != m_rdmaMem.end()) {
    if (offset + size == iter->offset) {
      freeSpace += iter->size;

      // delete following
      iter = m_rdmaMem.erase(iter);

      // go to previous and extend
      --iter;
      (*iter).size = freeSpace;
    }
  }
  Logging::debug(
      __FILE__, __LINE__,
      "Merged consecutive free RDMA memory regions, total free space: " +
          to_string(freeSpace));
}

//------------------------------------------------------------------------------------//

void BaseRDMA::printBuffer() {
  auto listIter = m_rdmaMem.begin();
  for (; listIter != m_rdmaMem.end(); ++listIter) {
    Logging::debug(__FILE__, __LINE__,
                   "offset=" + to_string((*listIter).offset) + "," +
                       "size=" + to_string((*listIter).size) + "," +
                       "free=" + to_string((*listIter).free));
  }
  Logging::debug(__FILE__, __LINE__, "---------");
}

//------------------------------------------------------------------------------------//

void BaseRDMA::setRemoteConnData(const rdmaConnID rdmaConnID, ib_conn_t &conn) {
  if (m_rconns.size() < rdmaConnID + 1) {
    m_rconns.resize(rdmaConnID + 1);
  }
  m_rconns[rdmaConnID] = conn;
}