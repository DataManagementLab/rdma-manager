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
  void exchangeProtoMsg(std::string ipAndPortString, Any* sendMsg, Any* recMsg);
  bool connectProto(const string& connection);

  bool isConnected(std::string ipAndPortString) {
    return m_connections.find(ipAndPortString) != m_connections.end();
  }

  /**
   * Returns the send timeout for a specific ip:port 
   * in milliseconds. -1 means no timeout set
   * 
   * @return send timeout in millseconds
   */
  int64_t getSendTimeout(std::string ipAndPortString);

  /**
   * Sets the send timeout for a specific connection or 
   * for all connections if ip:port is empty.
   * 
   * @param milliseconds how many milliseconds the send timeout 
   *                     should be long or -1 to disable
   * @param ipAndPortString ip:port for a specific connection or 
   *                        empty to apply to all connections
   * @return true if successfully applied
   */
  bool setSendTimeout(int64_t milliseconds = -1, std::string ipAndPortString = "");

  /**
   * Returns the receive timeout for a specific ip:port 
   * in milliseconds. -1 means no timeout set
   * 
   * @return receive timeout in millseconds
   */
  int64_t getRecvTimeout(std::string ipAndPortString);

  /**
   * Sets the receive timeout for a specific connection or 
   * for all connections if ip:port is empty.
   * 
   * @param milliseconds how many milliseconds the receive timeout 
   *                     should be long or -1 to disable
   * @param ipAndPortString ip:port for a specific connection or 
   *                        empty to apply to all connections
   * @return true if successfully applied
   */
  bool setRecvTimeout(int64_t milliseconds = -1, std::string ipAndPortString = "");

 protected:
  unordered_map<string, ProtoSendSocket*> m_connections;
};

}  // namespace rdma