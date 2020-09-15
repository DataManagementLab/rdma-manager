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
   * @return true if worked (for all)
   */
  bool setSendTimeout(int64_t milliseconds = -1, std::string ipAndPortString = "");

  /**
   * Sets a timeout for receiving packets. If timeout gets 
   * exceeded the receive method will return with an error.
   * 
   * @param milliseconds how long the timeout should be or -1 to infinite
   * @param ipAndPortString if not empty the timeout will only be set for specific connection
   * @return true if worked (for all)
   */
  bool setRecvTimeout(int64_t milliseconds = -1, std::string ipAndPortString = "");

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