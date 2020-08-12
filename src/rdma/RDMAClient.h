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
  RDMAClient(size_t mem_size, std::string name, std::string ownIpPort, std::string sequencerIpPort, NodeType::Enum nodeType) : RDMA_API_T(mem_size), m_name(name), m_ownIpPort(ownIpPort), m_sequencerIpPort(sequencerIpPort), m_nodeType(nodeType)
  {
  }
  RDMAClient(BaseMemory *memory, std::string name, std::string ownIpPort, std::string sequencerIpPort, NodeType::Enum nodeType) : RDMA_API_T(memory), m_name(name), m_ownIpPort(ownIpPort), m_sequencerIpPort(sequencerIpPort), m_nodeType(nodeType)
  {
  }
 public:
  RDMAClient() : RDMAClient(Config::RDMA_MEMSIZE) {}
  RDMAClient(size_t mem_size) : RDMAClient(mem_size, "RDMAClient") {}
  RDMAClient(size_t mem_size, string name) : RDMAClient(mem_size, name, Config::SEQUENCER_IP, Config::SEQUENCER_PORT){}
  RDMAClient(size_t mem_size, string name, string sequencerAddr, int sequencerPort) : RDMAClient(mem_size, name, Config::getIP(Config::RDMA_INTERFACE), sequencerAddr, sequencerPort){}
  RDMAClient(size_t mem_size, string name, string sequencerIpPort) : RDMAClient(mem_size, name, Config::getIP(Config::RDMA_INTERFACE), sequencerIpPort){}
  RDMAClient(size_t mem_size, string name, string ownIpPort, string sequencerAddr, int sequencerPort) : RDMAClient(mem_size, name, ownIpPort, sequencerAddr+":"+to_string(sequencerPort)){}
  RDMAClient(size_t mem_size, string name, string ownIpPort, string sequencerIpPort) : RDMAClient(mem_size, name, ownIpPort, sequencerIpPort, NodeType::Enum::CLIENT){

  }

  RDMAClient(BaseMemory *memory) : RDMAClient(memory, "RDMAClient") {}
  RDMAClient(BaseMemory *memory, string name) : RDMAClient(memory, name, Config::SEQUENCER_IP, Config::SEQUENCER_PORT){}
  RDMAClient(BaseMemory *memory, string name, string sequencerAddr, int sequencerPort) : RDMAClient(memory, name, Config::getIP(Config::RDMA_INTERFACE), sequencerAddr, sequencerPort){}
  RDMAClient(BaseMemory *memory, string name, string sequencerIpPort) : RDMAClient(memory, name, Config::getIP(Config::RDMA_INTERFACE), sequencerIpPort){}
  RDMAClient(BaseMemory *memory, string name, string ownIpPort, string sequencerAddr, int sequencerPort) : RDMAClient(memory, name, ownIpPort, sequencerAddr+":"+to_string(sequencerPort)){}
  RDMAClient(BaseMemory *memory, string name, string ownIpPort, string sequencerIpPort) : RDMAClient(memory, name, ownIpPort, sequencerIpPort, NodeType::Enum::CLIENT){

  }
  
  ~RDMAClient() {
    RDMAConnDisconnect disconnMsg;
    disconnMsg.set_nodeid(m_ownNodeID);
    for(std::pair<std::string, NodeID> entry : m_connections){
      if(entry.second == m_ownNodeID) continue;
      Any sendAny;
      sendAny.PackFrom(disconnMsg);
      Any rcvAny;
      ProtoClient::exchangeProtoMsg(entry.first, &sendAny, &rcvAny);
    }
  }

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
   * @return false fail
   */
  bool connect(const string& ipPort, NodeID &retServerNodeID) {

    // TODO REMOVE
    std::cout << std::endl << "RDMAClient.connect( " << ipPort << ", " << retServerNodeID << ") with OwnIp=" << m_ownIpPort << " and NodeIDSequencer=" << m_sequencerIpPort << std::endl; // TODO REMOVE

    if (!ProtoClient::isConnected(m_sequencerIpPort)) {
      m_ownNodeID = requestNodeID(m_sequencerIpPort, m_ownIpPort, m_nodeType);
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

          size_t retries = 50;
          size_t i = 0;
          while (i < retries && connResponse.return_() != MessageErrors::NO_ERROR)
          {
            ProtoClient::exchangeProtoMsg(m_sequencerIpPort, &getNodeIdReq, &rcvAny);
            rcvAny.UnpackTo(&connResponse);
            Logging::debug(__FILE__, __LINE__, "GetNodeIDForIpPortResponse returned an error: " + to_string(connResponse.return_()) + " retry " + to_string(i) + "/" + to_string(retries));
            usleep(Config::RDMA_SLEEP_INTERVAL * i);
            ++i;
          }

          if (connResponse.return_() != MessageErrors::NO_ERROR)
          {
            Logging::error(__FILE__, __LINE__, m_name + " could not fetch node id of server on connect! Address: " + ipPort);
            return false;
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

        if (retServerNodeID >= m_nodeIDsConnection.size()) {
            m_nodeIDsConnection.resize(retServerNodeID + 1);
            m_nodeIDsConnection[retServerNodeID] = ipPort;

        } else {
            m_nodeIDsConnection[retServerNodeID] = ipPort;
        }
        m_connections[ipPort] = retServerNodeID;




        // check if other Server tried to connect
        //only relevant if this is a Server too
        unique_lock<mutex> lck(m_connLock);
            if (retServerNodeID >= m_NodeIDsQPs.size()) {
                m_NodeIDsQPs.resize(retServerNodeID + 1);
            }

            if(m_NodeIDsQPs.at(retServerNodeID) == false){

                m_NodeIDsQPs[retServerNodeID] = true;
            }else{
                //other Server already called connect exit
                lck.unlock();
                return true;
            }
            lck.unlock();

    struct ib_qp_t qp;
    struct ib_conn_t localConn;

    // need to pass pointer of pointers because of UnreliableRDMA
    // UnreliableRDMA returns a pointer to the member of qp and locaCon
    auto qpPt = &qp;
    auto localConnPt = &localConn;

        //srq Server to Server is not yet working
        //init QP but dont add it to the members yet
      RDMA_API_T::initQPWithSuppliedID(&qpPt,&localConnPt);



      // exchange QP info

      RDMAConnRequest connRequest;
      connRequest.set_buffer(localConnPt->buffer);
      connRequest.set_rkey(localConnPt->rc.rkey);
      connRequest.set_qp_num(localConnPt->qp_num);
      connRequest.set_lid(localConnPt->lid);
      for (int i = 0; i < 16; ++i) {
        connRequest.add_gid(localConnPt->gid[i]);
      }
      connRequest.set_psn(localConnPt->ud.psn);
      connRequest.set_nodeid(m_ownNodeID);

      Any sendAny;
      sendAny.PackFrom(connRequest);
      Any rcvAny;

      ProtoClient::exchangeProtoMsg(ipPort, &sendAny, &rcvAny);


      if (rcvAny.Is<RDMAConnResponse>()) {
          // connect request was successful
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
          //set qp to members
          RDMA_API_T::setQP(retServerNodeID, *qpPt);
          RDMA_API_T::setLocalConnData(retServerNodeID, *localConnPt);

        RDMA_API_T::setRemoteConnData(retServerNodeID, remoteConn);
      } else {
          // connect request failed because other Server already connected
          //cleanup
          if (ibv_destroy_qp(qp.qp) != 0) {
              throw runtime_error("Error, ibv_destroy_qp() failed after invalid connection build up");
          }
          qp.qp = nullptr;
          RDMA_API_T::destroyCQ(qp.send_cq, qp.recv_cq);

         return true;
      }



      // connect QPs
      RDMA_API_T::connectQP(retServerNodeID);

      Logging::debug(__FILE__, __LINE__, "RDMAClient: connected to server!");


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

  //lock for connection
  mutex m_connLock;
  vector<bool> m_NodeIDsQPs;

  // Mapping from IPs to NodeIDs
  unordered_map<string, NodeID> m_connections;
  
  std::string m_name;
  std::string m_ownIpPort;
  std::string m_sequencerIpPort;
  NodeType::Enum m_nodeType;

  //Can be overwritten for special use-cases where NodeIDSequencer is insufficient
  virtual NodeID requestNodeID(std::string sequencerIpPort, std::string ownIpPort, NodeType::Enum nodeType)
  {
    // TODO REMOVE ownIpPort = Network::getAddressOfConnection(ownIpPort); // just ip is needed
    
    // std::cout << "Requesting IP. sequencerIpPort" << sequencerIpPort << " ownIpPort " << ownIpPort << std::endl;
    // check if client is connected to sequencer

    std::cout << "RDMAClient.requestNodeID(" << sequencerIpPort << ", " << ownIpPort << ", " << nodeType << ")" << std::endl; // TODO REMOVE

    if (ProtoClient::isConnected(sequencerIpPort)) {
      return m_ownNodeID;
    }
    std::cout << "RDMAClient.requestNodeID():  CONNECTING to NodeIDSequencer at " << sequencerIpPort << std::endl; // TODO REMOVE
    ProtoClient::connectProto(sequencerIpPort);
    std::cout << "RDMAClient.requestNodeID():  CONNECTED to NodeIDSequencer" << std::endl; // TODO REMOVE

    Any nodeIDRequest = ProtoMessageFactory::createNodeIDRequest(ownIpPort, m_name, nodeType);
    std::cout << "CHECK DONE" << std::endl; // TODO REMOVE
    Any rcvAny;
    // std::cout << "Sending nodeid request to NodeIDSequencer" << std::endl;
    std::cout << "RDMAClient.requestNodeID():  SENDING REQUEST " << std::endl; // TODO REMOVE
    ProtoClient::exchangeProtoMsg(sequencerIpPort, &nodeIDRequest, &rcvAny);

    std::cout << "RDMAClient.requestNodeID():  REQUEST SENT" << std::endl; // TODO REMOVE

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
