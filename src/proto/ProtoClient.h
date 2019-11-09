#pragma once

#include "../message/MessageTypes.h"
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
  bool connectProto(const string& connection);
  bool exchangeProtoMsg(std::string ipAndPortString, Any* sendMsg, Any* recMsg);

 protected:
  bool isConnected(std::string ipAndPortString) {
    return m_connections.find(ipAndPortString) != m_connections.end();
  }

  unordered_map<string, ProtoSendSocket*> m_connections;
};

}  // namespace rdma