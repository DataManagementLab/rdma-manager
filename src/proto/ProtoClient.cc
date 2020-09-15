#include "ProtoClient.h"
#include "../utils/Network.h"
#include "ProtoSendSocket.h"

using namespace rdma;

bool ProtoClient::connectProto(const std::string& connection) {
  if (isConnected(connection)) {
    return true;
  }

  // exchange QP info
  string ipAddr = Network::getAddressOfConnection(connection);
  size_t ipPort = Network::getPortOfConnection(connection);
  ProtoSendSocket* sendSocket = new ProtoSendSocket(ipAddr, ipPort);
  sendSocket->connect();
  m_connections[connection] = sendSocket;
  return true;
}

//------------------------------------------------------------------------------------//

void ProtoClient::sendProtoMsg(std::string ipAndPortString, Any* sendMsg) {
  auto* sendSocket = m_connections[ipAndPortString];
  sendSocket->send(sendMsg);
}

//------------------------------------------------------------------------------------//

void ProtoClient::exchangeProtoMsg(std::string ipAndPortString, Any* sendMsg,
                                   Any* recMsg) {
  auto* sendSocket = m_connections[ipAndPortString];
  sendSocket->send(sendMsg, recMsg);
}

//------------------------------------------------------------------------------------//

bool ProtoClient::setSendTimeout(int64_t milliseconds, std::string ipAndPortString){
  if(ipAndPortString.empty()){
    bool v = true;
    for(auto entry : m_connections){
      v &= entry.second->setOption(ZMQ_SNDTIMEO, &milliseconds, sizeof(int64_t));
    }
    return v;
  } else {
    auto* sendSocket = m_connections[ipAndPortString];
    return sendSocket->setOption(ZMQ_SNDTIMEO, &milliseconds);
  }
}

bool ProtoClient::setRecvTimeout(int64_t milliseconds, std::string ipAndPortString){
  if(ipAndPortString.empty()){
    bool v = true;
    for(auto entry : m_connections){
      v &= entry.second->setOption(ZMQ_RCVTIMEO, &milliseconds, sizeof(int64_t));
    }
    return v;
  } else {
    auto* sendSocket = m_connections[ipAndPortString];
    return sendSocket->setOption(ZMQ_RCVTIMEO, &milliseconds);
  }
}


bool ProtoClient::hasConnection(std::string ipAndPortString){
  if(ipAndPortString.empty()){
    bool v = true;
    for(auto entry : m_connections){
      v &= entry.second->hasConnection();
    }
    return v;
  } else {
    auto* sendSocket = m_connections[ipAndPortString];
    return sendSocket->hasConnection();
  }
}
