#pragma once

#include "../message/ProtoMessageFactory.h"
#include "../utils/Config.h"
#include "ProtoSendSocket.h"

namespace rdma {

class ProtoClient {
 public:
  ProtoClient() = default;
  ~ProtoClient() {
    for (auto kv : m_connections) {
      delete kv.second;
    }
    m_connections.clear();
  }

  void sendProtoMsg(std::string ipAndPortString, Any* sendMsg);
  void exchangeProtoMsg(std::string ipAndPortString, Any* sendMsg, Any* recMsg);
  bool connectProto(const string& connection);

  /**
   * Sets a timeout for sending packets. If timeout gets 
   * exceeded the send method will return with an error.
   * 
   * @param milliseconds how long the timeout should be or -1 to infinite
   * @param ipAndPortString if not empty the timeout will only be set for specific connection
   */
  void setSendTimeout(int milliseconds = -1, std::string ipAndPortString = "");

  /**
   * Sets a timeout for receiving packets. If timeout gets 
   * exceeded the receive method will return with an error.
   * 
   * @param milliseconds how long the timeout should be or -1 to infinite
   * @param ipAndPortString if not empty the timeout will only be set for specific connection
   */
  void setRecvTimeout(int milliseconds = -1, std::string ipAndPortString = "");

  bool isConnected(std::string ipAndPortString) {
    return m_connections.find(ipAndPortString) != m_connections.end();
  }

 protected:
  unordered_map<string, ProtoSendSocket*> m_connections;
};

}  // namespace rdma