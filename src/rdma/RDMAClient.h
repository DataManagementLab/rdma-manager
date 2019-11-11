/**
 * @file RDMAClient.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef RDMAClient_H_
#define RDMAClient_H_

#include "../proto/ProtoClient.h"
#include "../utils/Config.h"
#include "BaseRDMA.h"
#include "ReliableRDMA.h"
#include "UnreliableRDMA.h"

#include <list>
#include <unordered_map>

namespace rdma {

template <typename RDMA_API_T>
class RDMAClient : public RDMA_API_T, public ProtoClient {
 public:
  RDMAClient() : RDMAClient(Config::RDMA_MEMSIZE) {}
  RDMAClient(size_t mem_size) : RDMA_API_T(mem_size) {}
  RDMAClient(size_t mem_size, NodeID ownNodeID) : RDMA_API_T(mem_size), ownNodeID(ownNodeID) {}

  ~RDMAClient() {}

  // memory management
  bool remoteAlloc(const string& connection, const size_t size,
                   size_t& offset) {
    if (!isConnected(connection)) {
      return false;
    }

    Any sendAny = MessageTypes::createMemoryResourceRequest(size);
    Any rcvAny;
    if (!exchangeProtoMsg(connection, &sendAny, &rcvAny)) {
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
      Logging::warn("RDMAClient: Got error code " +
                    to_string(resResp.return_()));
    }
    return false;
  }
  bool remoteAlloc(const NodeID nodeID, const size_t size, size_t& offset) {
    return remoteAlloc(m_nodeIDsConnection[nodeID], size, offset);
  }

  bool remoteFree(const NodeID nodeID, const size_t size, const size_t offset) {
    return remoteFree(m_nodeIDsConnection[nodeID], size, offset);
  }

  bool remoteFree(const string& connection, const size_t size,
                  const size_t offset) {
    if (!isConnected(connection)) {
      return false;
    }

    Any sendAny = MessageTypes::createMemoryResourceRelease(size, offset);
    Any rcvAny;

    if (!exchangeProtoMsg(connection, &sendAny, &rcvAny)) {
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

  void* getBuffer(const size_t offset) {
    return ((char*)RDMA_API_T::getBuffer() + offset);
  }

  /**
   * @brief connects to an RDMAServer
   * Don't use arbitraty high NodeIDs, since NodeIDs are internally used for
   * array indexing
   */
  bool connect(const string& connection, const NodeID nodeID) {
    // check if client is connected to data node
    if (isConnected(connection)) {
      return true;
    }
    connectProto(connection);

    // create local QP
    if (!RDMA_API_T::initQPWithSuppliedID(nodeID)) {
      Logging::debug(__FILE__, __LINE__,
                     "An Error occurred while creating client QP");
      return false;
    }

    // exchange QP info

    ib_conn_t localConn = RDMA_API_T::getLocalConnData(nodeID);
    RDMAConnRequest connRequest;
    connRequest.set_buffer(localConn.buffer);
    connRequest.set_rkey(localConn.rc.rkey);
    connRequest.set_qp_num(localConn.qp_num);
    connRequest.set_lid(localConn.lid);
    for (int i = 0; i < 16; ++i) {
      connRequest.add_gid(localConn.gid[i]);
    }
    connRequest.set_psn(localConn.ud.psn);
    connRequest.set_nodeid(nodeID);

    Any sendAny;
    sendAny.PackFrom(connRequest);
    Any rcvAny;

    if (!exchangeProtoMsg(connection, &sendAny, &rcvAny)) {
      return false;
    }

    if (rcvAny.Is<RDMAConnResponse>()) {
      RDMAConnResponse connResponse;
      rcvAny.UnpackTo(&connResponse);

      struct ib_conn_t remoteConn;
      remoteConn.buffer = connResponse.buffer();
      remoteConn.rc.rkey = connResponse.rkey();
      remoteConn.qp_num = connResponse.qp_num();
      remoteConn.lid = connResponse.lid();
      remoteConn.ud.psn = connResponse.psn();
      for (int i = 0; i < 16; ++i) {
        remoteConn.gid[i] = connResponse.gid(i);
      }

      RDMA_API_T::setRemoteConnData(nodeID, remoteConn);
    } else {
      Logging::debug(__FILE__, __LINE__,
                     "An Error occurred while exchanging QP info");
      return false;
    }

    // connect QPs
    if (!RDMA_API_T::connectQP(nodeID)) {
      return false;
    }

    Logging::debug(__FILE__, __LINE__, "RDMAClient: connected to server!");

    if (nodeID >= m_nodeIDsConnection.size()) {
      m_nodeIDsConnection.resize(nodeID + 1);
      m_nodeIDsConnection[nodeID] = connection;

    } else {
      m_nodeIDsConnection[nodeID] = connection;
    }
    return true;
  }

 protected:
  unordered_map<string, NodeID> m_mcast_addr;
  NodeID ownNodeID;
  // Mapping from NodeID to IPs
  vector<string> m_nodeIDsConnection;
};
}  // namespace rdma

#endif /* RDMAClient_H_ */
