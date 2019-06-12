

#include "RDMAClient.h"
#include "../utils/Logging.h"
#include "../utils/Network.h"
#include "../message/MessageTypes.h"
#include "../message/MessageErrors.h"

#include <iterator>

using namespace rdma;

RDMAClient::RDMAClient(size_t mem_size, rdma_transport_t transport) {

  switch (transport) {
    case rc:
      m_rdmaManager = new RDMAManagerRC(mem_size);
      break;
    case ud:
      m_rdmaManager = new RDMAManagerUD();
      break;
  }

  if (m_rdmaManager == nullptr) {
    throw invalid_argument("RDMAManager could not be created");
  }
}

RDMAClient::~RDMAClient() {
  if (m_rdmaManager != nullptr) {
    delete m_rdmaManager;
    m_rdmaManager = nullptr;
  }

  for (auto kv : m_clients) {
    delete kv.second;
  }
  m_clients.clear();
}

void* RDMAClient::localAlloc(const size_t& size) {
  return m_rdmaManager->localAlloc(size);
}

bool RDMAClient::localFree(const void* ptr) {
  return m_rdmaManager->localFree(ptr);
}

bool RDMAClient::remoteAlloc(const NodeID nodeID, const size_t size, size_t& offset) {
  return remoteAlloc(m_nodeIDsConnection[nodeID], size, offset);
}

bool RDMAClient::remoteAlloc(const string& connection, const size_t size,
                             size_t& offset) {
  if (!connect(connection)) {
    return false;
  }
  ProtoClient* client = m_clients[connection];

  Any sendAny = MessageTypes::createMemoryResourceRequest(size);
  Any rcvAny;
  if (!client->send(&sendAny, &rcvAny)) {
    Logging::error(__FILE__, __LINE__, "cannot send message");
    return false;
  }

  if (rcvAny.Is<MemoryResourceResponse>()) {
    MemoryResourceResponse resResp;
    rcvAny.UnpackTo(&resResp);
    if (resResp.return_() == MessageErrors::NO_ERROR) {
      offset = resResp.offset();
      return true;
    }
    Logging::warn("RDMAClient: Got error code " + to_string(resResp.return_()));
  }
  return false;
}

bool RDMAClient::remoteFree(const NodeID nodeID, const size_t size, const size_t offset) {
  return remoteFree(m_nodeIDsConnection[nodeID], size, offset);
}

bool RDMAClient::remoteFree(const string& connection, const size_t size,
                            const size_t offset) {
  if (!connect(connection)) {
    return false;
  }
  ProtoClient* client = m_clients[connection];
  Any sendAny = MessageTypes::createMemoryResourceRelease(size, offset);
  Any rcvAny;
  if (!client->send(&sendAny, &rcvAny)) {
    return false;
  }

  if (rcvAny.Is<MemoryResourceResponse>()) {
    MemoryResourceResponse resResp;
    rcvAny.UnpackTo(&resResp);
    if (resResp.return_() == MessageErrors::NO_ERROR) {
      return true;
    }
    Logging::debug(
        __FILE__, __LINE__,
        "Release failed! Error was: " + to_string(resResp.return_()));
  }
  return false;
}

void* RDMAClient::getBuffer(const size_t offset) {
  return (void*) ((char*) m_rdmaManager->getBuffer() + offset);
}

bool RDMAClient::connect(const string& connection, bool managementQueue) {
  struct ib_addr_t retIbAddr;
  return connect(connection, retIbAddr, managementQueue);
}

bool RDMAClient::connect(const string& connection, const NodeID nodeID,
                                  bool managementQueue) {
    struct ib_addr_t retIbAddr;
    auto ret = connect(connection, retIbAddr, managementQueue);
    if(ret){
        if(nodeID >= m_nodeIDsIBaddr.size() ){
            m_countWR.resize(nodeID+1);
            m_countWR[nodeID] = 0;
            m_nodeIDsIBaddr.resize(nodeID+1);
            m_nodeIDsIBaddr[nodeID] = retIbAddr;
            m_nodeIDsConnection.resize(nodeID+1);
            m_nodeIDsConnection[nodeID] = connection;
            
        }else{
            m_countWR[nodeID] = 0;
            m_nodeIDsIBaddr[nodeID] = retIbAddr;
            m_nodeIDsConnection[nodeID] = connection;
        }
       
    }
    return ret;
}

bool RDMAClient::connect(const string& connection, struct ib_addr_t& ibAddr, bool managementQueue) {
  //check if client is connected to data node
  if (isConnected(connection)) {
    ibAddr = m_addr[connection];
    return true;
  }

  //create local QP
  if (!m_rdmaManager->initQP(ibAddr)) {
    Logging::debug(__FILE__, __LINE__,
                   "An Error occurred while creating client QP");
    return false;
  }
  m_addr[connection] = ibAddr;

  //exchange QP info
  string ipAddr = Network::getAddressOfConnection(connection);
  size_t ipPort = Network::getPortOfConnection(connection);
  ProtoClient* client = new ProtoClient(ipAddr, ipPort);
  if (!client->connect()) {
    return false;
  }
  m_clients[connection] = client;

  ib_conn_t localConn = m_rdmaManager->getLocalConnData(ibAddr);
  RDMAConnRequest connRequest;
  connRequest.set_buffer(localConn.buffer);
  connRequest.set_rkey(localConn.rc.rkey);
  connRequest.set_qp_num(localConn.qp_num);
  connRequest.set_lid(localConn.lid);
  for (int i = 0; i < 16; ++i) {
    connRequest.add_gid(localConn.gid[i]);
  }
  connRequest.set_psn(localConn.ud.psn);

  Any sendAny;
  sendAny.PackFrom(connRequest);
  Any rcvAny;

  if (!client->send(&sendAny, &rcvAny)) {
    return false;
  }

  uint64_t serverConnKey;
  if (rcvAny.Is<RDMAConnResponse>()) {
    RDMAConnResponse connResponse;
    rcvAny.UnpackTo(&connResponse);

    struct ib_conn_t remoteConn;
    remoteConn.buffer = connResponse.buffer();
    remoteConn.rc.rkey = connResponse.rkey();
    remoteConn.qp_num = connResponse.qp_num();
    remoteConn.lid = connResponse.lid();
    remoteConn.ud.psn = connResponse.psn();
    serverConnKey = connResponse.server_connkey();
    for (int i = 0; i < 16; ++i) {
      remoteConn.gid[i] = connResponse.gid(i);
    }

    m_rdmaManager->setRemoteConnData(ibAddr, remoteConn);
  } else {
    Logging::debug(__FILE__, __LINE__,
                   "An Error occurred while exchanging QP info");
    return false;
  }

  //connect QPs
  if (!m_rdmaManager->connectQP(ibAddr)) {
    return false;
  }

  //create management queue
  if (managementQueue && !createManagementQueue(connection, serverConnKey)) {
    Logging::debug(__FILE__, __LINE__,
                   "An Error occurred while creating management QP");
    return false;
  }

  Logging::debug(__FILE__, __LINE__, "RDMAClient: connected to server!");
  return true;
}

bool RDMAClient::write(const NodeID& nodeid, size_t remoteOffset,
                                void* localData, size_t size, bool signaled) {
  signaled = checkSignaled(signaled, nodeid);
  return m_rdmaManager->remoteWrite(m_nodeIDsIBaddr[nodeid], remoteOffset, localData, size,
                                    signaled);
}


uint64_t RDMAClient::getStartRdmaAddrForNode(NodeID nodeid) {
  auto ib_addr = m_nodeIDsIBaddr[nodeid];

  auto remote_conn = m_rdmaManager->getRemoteConnData(ib_addr);
  return remote_conn.buffer;
}

bool RDMAClient::writeRC(const NodeID& nodeid, size_t remoteOffset,
                                void* localData, size_t size, bool signaled) {
  signaled = checkSignaled(signaled, nodeid);
  return (RDMAManagerRC*)m_rdmaManager->remoteWrite(m_nodeIDsIBaddr[nodeid], remoteOffset, localData, size,
                                    signaled);
}

bool RDMAClient::read(const NodeID& nodeid, size_t remoteOffset,
                               void* localData, size_t size, bool signaled) {
  signaled = checkSignaled(signaled, nodeid);
  return m_rdmaManager->remoteRead(m_nodeIDsIBaddr.at(nodeid), remoteOffset, localData, size,
                                   signaled);
}

bool RDMAClient::fetchAndAdd(const NodeID& nodeid, size_t remoteOffset,
                                      void* localData, size_t size,
                                      bool signaled) {
  signaled = checkSignaled(signaled, nodeid);
  return m_rdmaManager->remoteFetchAndAdd(m_nodeIDsIBaddr.at(nodeid), remoteOffset, localData, size,
                                          signaled);
}

bool RDMAClient::compareAndSwap(const NodeID& nodeid,
                                         size_t remoteOffset, void* localData,
                                         int toCompare, int toSwap, size_t size,
                                         bool signaled) {
  signaled = checkSignaled(signaled, nodeid);


  if (m_rdmaManager->remoteCompareAndSwap(m_nodeIDsIBaddr.at(nodeid), remoteOffset, localData,
                                          toCompare, toSwap, size, signaled)) {
    return true;
  }
  return false;
}

bool RDMAClient::receive(const NodeID& nodeid, void* localData,
                                  size_t size) {
  return m_rdmaManager->receive(m_nodeIDsIBaddr.at(nodeid), localData, size);
}

bool RDMAClient::send(const NodeID& nodeid, void* localData,
                               size_t size, bool signaled) {
  signaled = checkSignaled(signaled, nodeid);
  return m_rdmaManager->send(m_nodeIDsIBaddr.at(nodeid), localData, size, signaled);
}

bool RDMAClient::pollReceive(const NodeID& nodeid) {
  return m_rdmaManager->pollReceive(m_nodeIDsIBaddr.at(nodeid));
}



bool RDMAClient::createManagementQueue(const string& connection,
                                       const uint64_t connKeyServer) {

//create local management QP
  struct ib_addr_t clientAddr;
  if (!m_rdmaManager->initQP(clientAddr, true)) {
    Logging::debug(__FILE__, __LINE__,
                   "An Error ocurred while creating client QP");
    return false;
  }
  m_mgmt_addr[connection] = clientAddr;

  uint64_t mgmtQPNum = m_rdmaManager->getQPNum(clientAddr);

//exchange QP info
  ProtoClient* client = m_clients[connection];

  ib_conn_t localConn = m_rdmaManager->getLocalConnData(clientAddr);
  RDMAConnRequestMgmt connRequest;
  connRequest.set_buffer(localConn.buffer);
  connRequest.set_rkey(localConn.rc.rkey);
  connRequest.set_server_connkey(connKeyServer);
  connRequest.set_qp_num(mgmtQPNum);
  connRequest.set_lid(localConn.lid);
  for (int i = 0; i < 16; ++i) {
    connRequest.add_gid(localConn.gid[i]);
  }
  connRequest.set_psn(localConn.ud.psn);

  Any sendAny;
  sendAny.PackFrom(connRequest);
  Any rcvAny;

  if (!client->send(&sendAny, &rcvAny)) {
    return false;
  }

  if (rcvAny.Is<RDMAConnResponseMgmt>()) {
    RDMAConnResponseMgmt connResponse;
    rcvAny.UnpackTo(&connResponse);

    struct ib_conn_t remoteConn;
    remoteConn.buffer = connResponse.buffer();
    remoteConn.rc.rkey = connResponse.rkey();
    remoteConn.qp_num = connResponse.qp_num();
    remoteConn.lid = connResponse.lid();
    for (int i = 0; i < 16; ++i) {
      remoteConn.gid[i] = connResponse.gid(i);
    }
    remoteConn.ud.psn = connResponse.psn();

    m_rdmaManager->setRemoteConnData(clientAddr, remoteConn);
  } else {
    Logging::debug(__FILE__, __LINE__,
                   "An Error occurred while exchanging QP info");
    return false;
  }

//connect QPs
  if (!m_rdmaManager->connectQP(clientAddr)) {
    return false;
  }

  Logging::debug(__FILE__, __LINE__,
                 "RDMAClient: Management connected to server!");

  return true;
}

// [[deprecated]]
// bool RDMAClient::write(ib_addr_t& ibAddr, size_t remoteOffset, void* localData,
//                        size_t size, bool signaled) {
//   signaled = checkSignaled(signaled);

//   //struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->remoteWrite(ibAddr, remoteOffset, localData, size,
//                                     signaled);
// }

// [[deprecated]]
// bool RDMAClient::write(const string& connection, size_t remoteOffset,
//                        void* localData, size_t size, bool signaled) {
//   signaled = checkSignaled(signaled);

//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->remoteWrite(ibAddr, remoteOffset, localData, size,
//                                     signaled);
// }

// bool RDMAClient::requestRead(ib_addr_t& ibAddr, size_t remoteOffset,
//                              void* localData, size_t size) {
//   return m_rdmaManager->requestRead(ibAddr, remoteOffset, localData, size);
// }

// bool RDMAClient::requestRead(const string& connection, size_t remoteOffset,
//                              void* localData, size_t size) {
//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->requestRead(ibAddr, remoteOffset, localData, size);
// }
// bool RDMAClient::requestRead(const NodeID& nid, size_t remoteOffset,
//                              void* localData, size_t size) {
//   return m_rdmaManager->requestRead(m_nodeIDsIBaddr[nid], remoteOffset,
//                                     localData, size);
// }

// [[deprecated]]
// bool RDMAClient::read(ib_addr_t& ibAddr, size_t remoteOffset, void* localData,
//                       size_t size, bool signaled) {
//   signaled = checkSignaled(signaled);

//   return m_rdmaManager->remoteRead(ibAddr, remoteOffset, localData, size,
//                                    signaled);
// }

// [[deprecated]]
// bool RDMAClient::read(const string& connection, size_t remoteOffset,
//                       void* localData, size_t size, bool signaled) {

//   signaled = checkSignaled(signaled);
//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->remoteRead(ibAddr, remoteOffset, localData, size,
//                                    signaled);
// }

// [[deprecated]]
// bool RDMAClient::fetchAndAdd(ib_addr_t& ibAddr, size_t remoteOffset,
//                              void* localData, size_t size, bool signaled) {
//   signaled = checkSignaled(signaled);

//   return m_rdmaManager->remoteFetchAndAdd(ibAddr, remoteOffset, localData, size,
//                                           signaled);
// }

// [[deprecated]]
// bool RDMAClient::fetchAndAdd(const string& connection, size_t remoteOffset,
//                              void* localData, size_t size, bool signaled) {
//   signaled = checkSignaled(signaled);

//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->remoteFetchAndAdd(ibAddr, remoteOffset, localData, size,
//                                           signaled);
// }

// [[deprecated]]
// bool RDMAClient::compareAndSwap(ib_addr_t& ibAddr, size_t remoteOffset,
//                                 void* localData, int toCompare, int toSwap,
//                                 size_t size,
//                                 bool signaled) {
//   signaled = checkSignaled(signaled);

//   if (m_rdmaManager->remoteCompareAndSwap(ibAddr, remoteOffset, localData,
//                                           toCompare, toSwap, size, signaled)) {
//     return true;
//   }
//   return false;
// }

// [[deprecated]]
// bool RDMAClient::compareAndSwap(const string& connection, size_t remoteOffset,
//                                 void* localData, int toCompare, int toSwap,
//                                 size_t size,
//                                 bool signaled) {
//   signaled = checkSignaled(signaled);

//   struct ib_addr_t ibAddr = m_addr[connection];
//   if (m_rdmaManager->remoteCompareAndSwap(ibAddr, remoteOffset, localData,
//                                           toCompare, toSwap, size, signaled)) {
//     return true;
//   }
//   return false;
// }

// bool RDMAClient::receive(const string& connection, void* localData,
//                          size_t size) {
//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->receive(ibAddr, localData, size);
// }

// [[deprecated]]
// bool RDMAClient::send(const string& connection, void* localData, size_t size,
// bool signaled) {
//   signaled = checkSignaled(signaled);

//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->send(ibAddr, localData, size, signaled);
// }

// bool RDMAClient::pollReceive(const string& connection) {
//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->pollReceive(ibAddr);
// }

// bool RDMAClient::pollSend(const string& connection) {
//   struct ib_addr_t ibAddr = m_addr[connection];
//   return m_rdmaManager->pollSend(ibAddr);
// }
bool RDMAClient::pollSend(const NodeID& nid) {
  return m_rdmaManager->pollSend(m_nodeIDsIBaddr[nid]);
}

// bool RDMAClient::receive(ib_addr_t& ib_addr, void* localAddr, size_t size) {
//   return m_rdmaManager->receive(ib_addr, localAddr, size);
// }

// [[deprecated]]
// bool RDMAClient::send(ib_addr_t& ib_addr, void* localAddr, size_t size,
// bool signaled) {
//   signaled = checkSignaled(signaled);
//   return m_rdmaManager->send(ib_addr, localAddr, size, signaled);
// }

// bool RDMAClient::pollReceive(ib_addr_t& ib_addr) {
//   return m_rdmaManager->pollReceive(ib_addr);
// }

// bool RDMAClient::pollSend(ib_addr_t& ib_addr) {
//   return m_rdmaManager->pollSend(ib_addr);
// }


// ------------------------ MULTICAST ------------------------

bool RDMAClient::joinMCastGroup(string mCastAddress) {
  struct ib_addr_t retIbAddr;
  return joinMCastGroup(mCastAddress, retIbAddr);
}

bool RDMAClient::joinMCastGroup(string mCastAddress,
                                struct ib_addr_t& retIbAddr) {
  if (m_mcast_addr.count(mCastAddress) != 0) {
    return false;
  }

  // join group
  bool result = m_rdmaManager->joinMCastGroup(mCastAddress, retIbAddr);
  if (result) {
    m_mcast_addr[mCastAddress] = retIbAddr;
  }
  return result;
}

bool RDMAClient::leaveMCastGroup(string mCastAddress) {
  if (m_mcast_addr.count(mCastAddress) == 0) {
    return false;
  }
  ib_addr_t ibAddr = m_mcast_addr[mCastAddress];
  return m_rdmaManager->leaveMCastGroup(ibAddr);
}

bool RDMAClient::leaveMCastGroup(struct ib_addr_t ibAddr) {
  return m_rdmaManager->leaveMCastGroup(ibAddr);
}

bool RDMAClient::sendMCast(string mCastAddress, const void* memAddr,
                           size_t size, bool signaled) {
  if (m_mcast_addr.count(mCastAddress) == 0) {
    return false;
  }
  ib_addr_t ibAddr = m_mcast_addr[mCastAddress];
  return m_rdmaManager->sendMCast(ibAddr, memAddr, size, signaled);
}

bool RDMAClient::sendMCast(struct ib_addr_t ibAddr, const void* memAddr,
                           size_t size, bool signaled) {
  return m_rdmaManager->sendMCast(ibAddr, memAddr, size, signaled);
}

bool RDMAClient::receiveMCast(string mCastAddress, const void* memAddr,
                              size_t size) {
  if (m_mcast_addr.count(mCastAddress) == 0) {
    return false;
  }
  ib_addr_t ibAddr = m_mcast_addr[mCastAddress];
  return m_rdmaManager->receiveMCast(ibAddr, memAddr, size);
}

bool RDMAClient::receiveMCast(struct ib_addr_t ibAddr, const void* memAddr,
                              size_t size) {
  return m_rdmaManager->receiveMCast(ibAddr, memAddr, size);
}

bool RDMAClient::pollReceiveMCast(string mCastAddress) {
  if (m_mcast_addr.count(mCastAddress) == 0) {
    return false;
  }
  ib_addr_t ibAddr = m_mcast_addr[mCastAddress];
  return m_rdmaManager->pollReceiveMCast(ibAddr);
}

bool RDMAClient::pollReceiveMCast(struct ib_addr_t ibAddr) {
  return m_rdmaManager->pollReceiveMCast(ibAddr);
}

// [[deprecated]]
// bool rdma::RDMAClient::fetchAndAdd(const string& connection, size_t remoteOffset,
//                                       void* localData, size_t value_to_add, size_t size,
//                                       bool signaled) {
//     signaled = checkSignaled(signaled);

//     struct ib_addr_t ibAddr = m_addr[connection];
//     return m_rdmaManager->remoteFetchAndAdd(ibAddr, remoteOffset, localData,value_to_add, size,
//                                             signaled);

// }

bool rdma::RDMAClient::fetchAndAdd(const NodeID& nodeid, size_t remoteOffset, void* localData,
                                      size_t value_to_add, size_t size, bool signaled) {
    signaled = checkSignaled(signaled, nodeid);
    return m_rdmaManager->remoteFetchAndAdd(m_nodeIDsIBaddr[nodeid], remoteOffset, localData,value_to_add, size,
                                            signaled);

}
