

#include "RDMAServer.h"
#include "../utils/Logging.h"
#include "../message/MessageTypes.h"
#include "../utils/Network.h"

using namespace rdma;

RDMAServer::RDMAServer(int port, rdma_transport_t transport)
    : RDMAServer("RDMAServer", port, Config::RDMA_MEMSIZE, transport) {
}
RDMAServer::RDMAServer(string name, int port, bool newFunc)
    : ProtoServer(name, port){
      (void)newFunc;

}
RDMAServer::RDMAServer(string name, int port, uint64_t memsize, rdma_transport_t transport)
    : ProtoServer(name, port) {
  m_countWR = 0;

  switch (transport) {
    case rc:
      m_rdmaManager = new RDMAManagerRC(memsize);
      break;
    case ud:
      m_rdmaManager = new RDMAManagerUD();
      break;
  }

  if (m_rdmaManager == nullptr) {
    throw invalid_argument("RDMAManager could not be created");
  }
}

RDMAServer::~RDMAServer() {
  if (m_rdmaManager != nullptr) {
    delete m_rdmaManager;
    m_rdmaManager = nullptr;
  }
}

bool RDMAServer::startServer() {
  // start data node server
  if (!ProtoServer::startServer()) {
    Logging::error(__FILE__, __LINE__, "RDMAServer: could not be started");
    return false;
  }

  Logging::debug(__FILE__, __LINE__, "RDMAServer: started server!");
  return true;
}

/* public methods */

void RDMAServer::stopServer() {
  ProtoServer::stopServer();
}

void* RDMAServer::localAlloc(const size_t& size) {
  return m_rdmaManager->localAlloc(size);
}

rdma_mem_t RDMAServer::remoteAlloc(const size_t& size) {
  return m_rdmaManager->remoteAlloc(size);
}

bool RDMAServer::localFree(const void* ptr) {
  return m_rdmaManager->localFree(ptr);
}

bool RDMAServer::localFree(const size_t& offset) const {
  return m_rdmaManager->localFree(offset);
}

void* RDMAServer::getBuffer(const size_t offset) {
  return (void*) ((char*) m_rdmaManager->getBuffer() + offset);
}

bool RDMAServer::receive(ib_addr_t& ibAddr, void* localAddr, size_t size) {
  return m_rdmaManager->receive(ibAddr, localAddr, size);
}

bool RDMAServer::send(ib_addr_t& ibAddr, void* localAddr, size_t size,
bool signaled) {
  signaled = checkSignaled(signaled);
  return m_rdmaManager->send(ibAddr, localAddr, size, signaled);
}

bool RDMAServer::pollReceive(ib_addr_t &ibAddr, bool doPoll) {
  return m_rdmaManager->pollReceive(ibAddr, doPoll);
}

bool RDMAServer::pollReceive(ib_addr_t& ibAddr, uint32_t& ret_qp_num) {
    return m_rdmaManager->pollReceive(ibAddr, ret_qp_num);
}
bool RDMAServer::joinMCastGroup(string mCastAddress) {
  struct ib_addr_t retIbAddr;
  return joinMCastGroup(mCastAddress, retIbAddr);
}

bool RDMAServer::joinMCastGroup(string mCastAddress,
                                struct ib_addr_t& retIbAddr) {
  // join group
  bool result = m_rdmaManager->joinMCastGroup(mCastAddress, retIbAddr);
  if (result) {
    m_mcastAddr[mCastAddress] = retIbAddr;
  }
  return result;
}

bool RDMAServer::leaveMCastGroup(string mCastAddress) {
  ib_addr_t ibAddr = m_mcastAddr[mCastAddress];
  return m_rdmaManager->leaveMCastGroup(ibAddr);
}

bool RDMAServer::leaveMCastGroup(struct ib_addr_t ibAddr) {
  return m_rdmaManager->leaveMCastGroup(ibAddr);
}

bool RDMAServer::sendMCast(string mCastAddress, const void* memAddr,
                           size_t size, bool signaled) {
  ib_addr_t ibAddr = m_mcastAddr[mCastAddress];
  return m_rdmaManager->sendMCast(ibAddr, memAddr, size, signaled);
}

bool RDMAServer::sendMCast(struct ib_addr_t ibAddr, const void* memAddr,
                           size_t size, bool signaled) {
  return m_rdmaManager->sendMCast(ibAddr, memAddr, size, signaled);
}

bool RDMAServer::receiveMCast(string mCastAddress, const void* memAddr,
                              size_t size) {
  ib_addr_t ibAddr = m_mcastAddr[mCastAddress];
  return m_rdmaManager->receiveMCast(ibAddr, memAddr, size);
}

bool RDMAServer::receiveMCast(struct ib_addr_t ibAddr, const void* memAddr,
                              size_t size) {
  return m_rdmaManager->receiveMCast(ibAddr, memAddr, size);
}

bool RDMAServer::pollReceiveMCast(string mCastAddress) {
  ib_addr_t ibAddr = m_mcastAddr[mCastAddress];
  return m_rdmaManager->pollReceiveMCast(ibAddr);
}

bool RDMAServer::pollReceiveMCast(struct ib_addr_t ibAddr) {
  return m_rdmaManager->pollReceiveMCast(ibAddr);
}

void RDMAServer::handle(Any* anyReq, Any* anyResp) {
  if (anyReq->Is<RDMAConnRequest>()) {
    RDMAConnResponse connResp;
    RDMAConnRequest connReq;
    anyReq->UnpackTo(&connReq);
    connectQueue(&connReq, &connResp);
    Logging::debug(__FILE__, __LINE__, "RDMAServer::handle: after connectQueue");
    anyResp->PackFrom(connResp);
  } else if (anyReq->Is<RDMAConnRequestMgmt>()) {
    RDMAConnResponseMgmt connResp;
    RDMAConnRequestMgmt connReq;
    anyReq->UnpackTo(&connReq);
    connectMgmtQueue(&connReq, &connResp);
    anyResp->PackFrom(connResp);
  } else if (anyReq->Is<MemoryResourceRequest>()) {
    MemoryResourceResponse respMsg;
    MemoryResourceRequest reqMsg;
    anyReq->UnpackTo(&reqMsg);
    if (reqMsg.type() == MessageTypesEnum::MEMORY_RESOURCE_RELEASE) {
      size_t offset = reqMsg.offset();
      respMsg.set_return_(releaseMemoryResource(offset));
      respMsg.set_offset(offset);
    } else if (reqMsg.type() == MessageTypesEnum::MEMORY_RESOURCE_REQUEST) {
      size_t offset = 0;
      size_t size = reqMsg.size();
      respMsg.set_return_(requestMemoryResource(size, offset));
      respMsg.set_offset(offset);
    }
    anyResp->PackFrom(respMsg);
  } else {
    //Send response with bad return code;
    ErrorMessage errorResp;
    errorResp.set_return_(MessageErrors::INVALID_MESSAGE);
    anyResp->PackFrom(errorResp);
  }
}

/* public methods that interfere with server handlers */
ib_addr_t RDMAServer::getMgmtQueue(const ib_addr_t& serverQueue) {
  unique_lock<mutex> lck(m_connLock);
  ib_addr_t result = m_addr2mgmtAddr.at(serverQueue.conn_key);
  lck.unlock();
  return result;
}

vector<ib_addr_t> RDMAServer::getQueues() {
  unique_lock<mutex> lck(m_connLock);
  vector<ib_addr_t> result = m_addr;
  lck.unlock();
  return result;

}

/* private server handler methods */

bool RDMAServer::connectQueue(RDMAConnRequest* connRequest,
                              RDMAConnResponse* connResponse) {
  unique_lock<mutex> lck(m_connLock);

  //create local QP
  struct ib_addr_t ibAddr;
  if (!m_rdmaManager->initQP(ibAddr)) {
    lck.unlock();
    return false;
  }
  m_addr.push_back(ibAddr);

  // set remote connection data
  struct ib_conn_t remoteConn;
  remoteConn.buffer = connRequest->buffer();
  remoteConn.rc.rkey = connRequest->rkey();
  remoteConn.qp_num = connRequest->qp_num();
  remoteConn.lid = connRequest->lid();
  for (int i = 0; i < 16; ++i) {
    remoteConn.gid[i] = connRequest->gid(i);
  }
  remoteConn.ud.psn = connRequest->psn();
  m_rdmaManager->setRemoteConnData(ibAddr, remoteConn);

  //connect QPs
  if (!m_rdmaManager->connectQP(ibAddr)) {
    lck.unlock();
    return false;
  }

  //create response
  ib_conn_t localConn = m_rdmaManager->getLocalConnData(ibAddr);
  connResponse->set_buffer(localConn.buffer);
  connResponse->set_rkey(localConn.rc.rkey);
  connResponse->set_qp_num(localConn.qp_num);
  connResponse->set_lid(localConn.lid);
  connResponse->set_psn(localConn.ud.psn);
  connResponse->set_server_connkey(ibAddr.conn_key);
  for (int i = 0; i < 16; ++i) {
    connResponse->add_gid(localConn.gid[i]);
  }

  Logging::debug(
      __FILE__, __LINE__,
      "RDMAServer: connected to client!" + to_string(ibAddr.conn_key));

  lck.unlock();
  return true;
}



bool RDMAServer::connectMgmtQueue(RDMAConnRequestMgmt* connRequest,
                                  RDMAConnResponseMgmt* connResponse) {
  unique_lock<mutex> lck(m_connLock);

  uint64_t serverConnKey = connRequest->server_connkey();

  //create local QP
  struct ib_addr_t serverMgmtAddr;
  if (!m_rdmaManager->initQP(serverMgmtAddr, true)) {
    lck.unlock();
    return false;
  }
  ib_conn_t localConn = m_rdmaManager->getLocalConnData(serverMgmtAddr);
  m_addr2mgmtAddr[serverConnKey] = serverMgmtAddr;

  // set remote connection data
  struct ib_conn_t remoteConn;
  remoteConn.buffer = connRequest->buffer();
  remoteConn.rc.rkey = connRequest->rkey();
  remoteConn.qp_num = connRequest->qp_num();
  remoteConn.lid = connRequest->lid();
  for (int i = 0; i < 16; ++i) {
    remoteConn.gid[i] = connRequest->gid(i);
  }
  remoteConn.ud.psn = connRequest->psn();
  m_rdmaManager->setRemoteConnData(serverMgmtAddr, remoteConn);

  //connect QPs
  if (!m_rdmaManager->connectQP(serverMgmtAddr)) {
    lck.unlock();
    return false;
  }

  //create response
  connResponse->set_buffer(localConn.buffer);
  connResponse->set_rkey(localConn.rc.rkey);
  connResponse->set_qp_num(localConn.qp_num);
  connResponse->set_lid(localConn.lid);
  for (int i = 0; i < 16; ++i) {
    connResponse->add_gid(localConn.gid[i]);
  }
  connResponse->set_psn(localConn.ud.psn);

  Logging::debug(__FILE__, __LINE__,
                 "RDMAServer: Management connected to client!");

  lck.unlock();
  return true;
}

MessageErrors RDMAServer::requestMemoryResource(const size_t size,
                                                size_t& offset) {
  unique_lock<mutex> lck(m_memLock);

  rdma_mem_t memRes = m_rdmaManager->remoteAlloc(size);
  offset = memRes.offset;

  if (!memRes.isnull) {
    lck.unlock();
    return MessageErrors::NO_ERROR;
  }

  lck.unlock();
  return MessageErrors::MEMORY_NOT_AVAILABLE;
}

MessageErrors RDMAServer::releaseMemoryResource(size_t& offset) {
  unique_lock<mutex> lck(m_memLock);
  if (!m_rdmaManager->remoteFree(offset)) {
    lck.unlock();
    return MessageErrors::MEMORY_RELEASE_FAILED;
  }
  lck.unlock();
  return MessageErrors::NO_ERROR;
}

