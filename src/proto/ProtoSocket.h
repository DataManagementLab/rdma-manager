/**
 * @file ProtoSocket.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef ZMQ_ProtoSocket_H
#define ZMQ_ProtoSocket_H

#include "../utils/Config.h"

#include <unistd.h>
#include <cstring>
#include <string>

#include "../message/ProtoMessageFactory.h"
#include "../utils/Logging.h"
#include "zmq.hpp"

using google::protobuf::Any;

namespace rdma {

class ProtoSocket {
 public:
  ProtoSocket(string addr, int port, int sockType);

  ~ProtoSocket();

  bool bind();

  bool connect();

  bool isOpen();

  bool sendMore(Any* msg);

  bool send(Any* msg);

  bool receive(Any* msg);

  bool close();

  bool closeContext();

  bool setOption(int option_name, const void *option_value, size_t option_len = sizeof(int));

  int64_t getSendTimeout();

  bool setSendTimeout(int64_t milliseconds = -1);

  int64_t getRecvTimeout();

  bool setRecvTimeout(int64_t milliseconds = -1);

  bool hasConnection();

 private:
  zmq::context_t* m_pCtx;

  string m_conn;
  int64_t m_send_timeout = -1; // needed to detect valid connection
  int64_t m_recv_timeout = -1;
  int m_sockType;
  bool m_isOpen;
  zmq::socket_t* m_pSock;
};

}  // end namespace rdma

#endif
