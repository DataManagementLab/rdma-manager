

#include "ProtoSendSocket.h"
#include "../message/MessageErrors.h"

using namespace rdma;

ProtoSendSocket::ProtoSendSocket(string address, int port) {
  m_address = address;
  m_port = port;
  m_pSocket = NULL;
  m_isConnected = false;
}

ProtoSendSocket::~ProtoSendSocket() {
  if (m_pSocket != nullptr) {
    delete m_pSocket;
    m_pSocket = nullptr;
  }
}

void ProtoSendSocket::connect() {
  m_pSocket = new ProtoSocket(m_address, m_port, ZMQ_REQ);
  if (!m_pSocket->connect()) {
    throw runtime_error("Cannot connect to server");
  }
  m_isConnected = true;
}

void ProtoSendSocket::send(Any* sendMsg){
  if (!m_isConnected) {
    throw runtime_error("Not connected to server");
  }
  if (sendMsg == NULL || !m_pSocket->send(sendMsg)) {
    throw runtime_error("Cannot send message");
  }
}

void ProtoSendSocket::send(Any* sendMsg, Any* recMsg) {
  if (!m_isConnected) {
    throw runtime_error("Not connected to server");
  }
  if (sendMsg == NULL || !m_pSocket->send(sendMsg)) {
    throw runtime_error("Cannot send message");
  }
  if (recMsg == NULL || !m_pSocket->receive(recMsg)) {
    throw runtime_error("Cannot receive message");
  }
  if (recMsg->Is<ErrorMessage>()) {
    ErrorMessage errMsg;
    recMsg->UnpackTo(&errMsg);
    if (errMsg.return_() != MessageErrors::NO_ERROR) {
      throw runtime_error("Error " + to_string(errMsg.return_()) + " returned from server");
    }
  }
}

bool ProtoSendSocket::setOption(int option_name, const void *option_value, size_t option_len){
  return m_pSocket->setOption(option_name, option_value, option_len);
}

int64_t ProtoSendSocket::getSendTimeout(){
  return m_pSocket->getSendTimeout();
}

bool ProtoSendSocket::setSendTimeout(int64_t milliseconds){
  return m_pSocket->setSendTimeout(milliseconds);
}

int64_t ProtoSendSocket::getRecvTimeout(){
  return m_pSocket->getRecvTimeout();
}

bool ProtoSendSocket::setRecvTimeout(int64_t milliseconds){
  return m_pSocket->setRecvTimeout(milliseconds);
}

bool ProtoSendSocket::hasConnection(){
  return m_pSocket->hasConnection();
}

