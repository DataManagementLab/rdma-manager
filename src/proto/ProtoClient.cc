

#include "ProtoClient.h"
#include "../message/MessageErrors.h"

using namespace rdma;

ProtoClient::ProtoClient(string address, int port) {
  m_address = address;
  m_port = port;
  m_pSocket = NULL;
  m_isConnected = false;
}

ProtoClient::~ProtoClient() {
  if (m_pSocket != nullptr) {
    delete m_pSocket;
    m_pSocket = nullptr;
  }
}

bool ProtoClient::connect() {
  m_pSocket = new ProtoSocket(m_address, m_port, ZMQ_REQ);
  if (!m_pSocket->connect()) {
    Logging::fatal(__FILE__, __LINE__, "Cannot connect to server");
    return false;
  }
  m_isConnected = true;
  return true;
}

bool ProtoClient::send(Any* sendMsg, Any* recMsg) {
  if (!m_isConnected) {
    Logging::fatal(__FILE__, __LINE__, "Not connected to server");
    return false;
  }
  if (sendMsg == NULL || !m_pSocket->send(sendMsg)) {
    Logging::fatal(__FILE__, __LINE__, "Cannot send message");
    return false;
  }
  if (recMsg == NULL || !m_pSocket->receive(recMsg)) {
    Logging::fatal(__FILE__, __LINE__, "Cannot receive message");
    return false;
  }
  if (recMsg->Is<ErrorMessage>()) {
    ErrorMessage errMsg;
    recMsg->UnpackTo(&errMsg);
    if (errMsg.return_() != MessageErrors::NO_ERROR) {
      Logging::error(
          __FILE__, __LINE__,
          "error " + to_string(errMsg.return_()) + " returned from server");
      return false;
    }
  }
  return true;
}

