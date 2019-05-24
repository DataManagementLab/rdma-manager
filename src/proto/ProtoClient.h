/**
 * @file ProtoClient.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef NET_PROTOCLIENT_H
#define NET_PROTOCLIENT_H

#include "../utils/Config.h"
#include "../utils/Logging.h"
#include "../message/MessageTypes.h"
#include "ProtoSocket.h"

using google::protobuf::Any;
namespace rdma {

class ProtoClient {
 public:
  ProtoClient(string address, int port);
  virtual ~ProtoClient();
  bool connect();
  bool send(Any* sendMsg, Any* recMsg);

  int getPort() {
    return m_port;
  }

 private:
  string m_address;
  int m_port;
  ProtoSocket* m_pSocket;bool m_isConnected;
};

}  // end namespace dpi

#endif /* CLIENT_H_ */
