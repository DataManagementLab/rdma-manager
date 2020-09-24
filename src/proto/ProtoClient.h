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
   * Returns true if a nodeID known for the given ip and port.
   * Can even be true if actual connection is lost. 
   * Use hasConnection() to check if currently a connection exists
   * 
   * @param ipAndPortString ip:port of the connection 
   * @return true if a nodeID is known for the given ip and port
   */
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

  /**
   * WARNING: It could be that this no longer works!
   * Returns if currently a valid TCP connection is open 
   * for the given ip and port. If ip and port is empty 
   * all connections will be checked and only returns true 
   * if all have a valid connection.
   * Does not mean that a valid nodeID has been received
   * 
   * @param ipAndPortString ip:port of the connection or empty for all
   * @return true if currently a valid TCP connection is open
   */
  bool hasConnection(std::string ipAndPortString = "");

 protected:
  unordered_map<string, ProtoSendSocket*> m_connections;
};

}  // namespace rdma