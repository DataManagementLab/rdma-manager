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
#include "NodeIDSequencer.h"

#include <list>
#include <unordered_map>

namespace rdma {

template <typename RDMA_API_T>
class RDMAClient : public RDMA_API_T, public ProtoClient {
 protected:
  RDMAClient(size_t mem_size, std::string name, std::string ownIpPort, NodeType::Enum nodeType) : RDMA_API_T(mem_size), m_name(name)
  {
    m_ownNodeID = requestNodeID(m_sequencerIpPort, ownIpPort, nodeType);
  }
 public:
  RDMAClient() : RDMAClient(Config::RDMA_MEMSIZE) {}
  RDMAClient(size_t mem_size) : RDMAClient(mem_size, "RDMAClient") {}
  RDMAClient(size_t mem_size, std::string name) : RDMAClient(mem_size, name, Config::getIP(Config::RDMA_INTERFACE), NodeType::Enum::CLIENT)
  {
  }
  
  ~RDMAClient() {}

  // memory management
  bool remoteAlloc(const string& connection, const size_t size,
                   size_t& offset) {
    if (!isConnected(connection)) {
      Logging::error(__FILE__, __LINE__, "RDMAClient: remoteAlloc failed since client is not connected to ProtoServer: " + connection);
      return false;
    }

    Any sendAny = ProtoMessageFactory::createMemoryResourceRequest(size);
    Any rcvAny;
    ProtoClient::exchangeProtoMsg(connection, &sendAny, &rcvAny);

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

    Any sendAny = ProtoMessageFactory::createMemoryResourceRelease(size, offset);
    Any rcvAny;

    ProtoClient::exchangeProtoMsg(connection, &sendAny, &rcvAny);

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
   * @brief Connects to an RDMAServer
   * 
   * @param ipPort Ip : port string
   * @param retServerNodeID nodeId of the server connected to
   * @return true success
   * @return false failzz
   */
  bool connect(const string& ipPort, NodeID &retServerNodeID) {

    if (!ProtoClient::isConnected(m_sequencerIpPort)) {
      throw runtime_error("RDMAClient must be connected to NodeIDSequencer");
    }

    // check if client is connected to data node
    if (!ProtoClient::isConnected(ipPort)) {
      
      ProtoClient::connectProto(ipPort);

      {
        //Request nodeID from Sequencer
        auto getNodeIdReq = ProtoMessageFactory::createGetNodeIDForIpPortRequest(ipPort);

        Any rcvAny;
        ProtoClient::exchangeProtoMsg(m_sequencerIpPort, &getNodeIdReq, &rcvAny);

        if (rcvAny.Is<GetNodeIDForIpPortResponse>()) {

          GetNodeIDForIpPortResponse connResponse;
          rcvAny.UnpackTo(&connResponse);

          if (connResponse.return_() != MessageErrors::NO_ERROR)
          {
            throw runtime_error("GetNodeIDForIpPortResponse returned an error: " + to_string(connResponse.return_()));
          }

          retServerNodeID = connResponse.node_id();


          if (connResponse.ip() != ipPort)
          {
            std::cout << "name: " << m_name << " returned nodeid: " << retServerNodeID << std::endl;
            throw runtime_error("Fetched IP (" + connResponse.ip() + ") from Sequencer did not match requested IP ("+ipPort+")");
          }
        }
        else
        {
          throw runtime_error("An Error occurred while fetching NodeID for ip: " + ipPort);
        }
      }
      // create local QP
      RDMA_API_T::initQPWithSuppliedID(retServerNodeID);

      // exchange QP info
      ib_conn_t localConn = RDMA_API_T::getLocalConnData(retServerNodeID);
      RDMAConnRequest connRequest;
      connRequest.set_buffer(localConn.buffer);
      connRequest.set_rkey(localConn.rc.rkey);
      connRequest.set_qp_num(localConn.qp_num);
      connRequest.set_lid(localConn.lid);
      for (int i = 0; i < 16; ++i) {
        connRequest.add_gid(localConn.gid[i]);
      }
      connRequest.set_psn(localConn.ud.psn);
      connRequest.set_nodeid(m_ownNodeID);

      Any sendAny;
      sendAny.PackFrom(connRequest);
      Any rcvAny;

      ProtoClient::exchangeProtoMsg(ipPort, &sendAny, &rcvAny);

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

        RDMA_API_T::setRemoteConnData(retServerNodeID, remoteConn);
      } else {
        throw runtime_error("An Error occurred while exchanging QP info");
      }

      // connect QPs
      RDMA_API_T::connectQP(retServerNodeID);

      Logging::debug(__FILE__, __LINE__, "RDMAClient: connected to server!");

      if (retServerNodeID >= m_nodeIDsConnection.size()) {
        m_nodeIDsConnection.resize(retServerNodeID + 1);
        m_nodeIDsConnection[retServerNodeID] = ipPort;

      } else {
        m_nodeIDsConnection[retServerNodeID] = ipPort;
      }
      m_connections[ipPort] = retServerNodeID;
      return true;
    }
    else
    {
      retServerNodeID = m_connections[ipPort];
      return true;
    }

  }

  NodeID getOwnNodeID()
  {
    return m_ownNodeID;
  }

 protected:
  unordered_map<string, NodeID> m_mcast_addr;
  NodeID m_ownNodeID;
  // Mapping from NodeID to IPs
  vector<string> m_nodeIDsConnection;

  // Mapping from IPs to NodeIDs
  unordered_map<string, NodeID> m_connections;

  std::string m_name;
  std::string m_sequencerIpPort = Config::SEQUENCER_IP + ":" + to_string(Config::SEQUENCER_PORT);
  //Can be overwritten for special use-cases where NodeIDSequencer is insufficient
  virtual NodeID requestNodeID(std::string sequencerIpPort, std::string ownIpPort, NodeType::Enum nodeType)
  {
    // check if client is connected to sequencer
    if (ProtoClient::isConnected(sequencerIpPort)) {
      return m_ownNodeID;
    }
    ProtoClient::connectProto(sequencerIpPort);

    Any nodeIDRequest = ProtoMessageFactory::createNodeIDRequest(ownIpPort, m_name, nodeType);
    Any rcvAny;

    ProtoClient::exchangeProtoMsg(sequencerIpPort, &nodeIDRequest, &rcvAny);

    if (rcvAny.Is<NodeIDResponse>()) {
      NodeIDResponse connResponse;
      rcvAny.UnpackTo(&connResponse);
      return connResponse.nodeid();
    } else {
      Logging::error(__FILE__, __LINE__,
                     "RDMAClient could not request NodeID from NodeIDSequencer: received wrong response type");
      throw std::runtime_error("RDMAClient could not request NodeID from NodeIDSequencer: received wrong response type");
    }
  }

protected:
  using ProtoClient::connectProto; //Make private
  using ProtoClient::exchangeProtoMsg; //Make private

};
}  // namespace rdma

#endif /* RDMAClient_H_ */
