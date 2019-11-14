/**
 * @file TestProtoServer.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestProtoServer_H_
#define SRC_TEST_NET_TestProtoServer_H_

#include "../../src/utils/Config.h"
#include "../../src/proto/ProtoServer.h"
#include "../../src//proto/ProtoClient.h"
#include <gtest/gtest.h>
#include <string>

using std::string;

using namespace rdma;

class TestProtoServer : public testing::Test {
protected:

  class HelloServer : public ProtoServer {
   protected:
    string m_msgReceived;
   public:
    HelloServer()
        : ProtoServer("HelloServer", Config::HELLO_PORT) {
      m_msgReceived = "";
    }

    void handle(Any* msg, Any* anyResponse) {
      HelloMessage helloMsg;
      if (msg->Is<HelloMessage>()) {
        msg->UnpackTo(&helloMsg);
      }

      m_msgReceived = helloMsg.name();
      Logging::debug(__FILE__, __LINE__, "Server received: " + helloMsg.name());

      HelloMessage responseMsg;
      responseMsg.set_name(m_msgReceived);
      anyResponse->PackFrom(responseMsg);
    }

    string msgReceived() {
      return m_msgReceived;
    }

  };

  class HelloClient : public ProtoClient {
   protected:
    string m_msgReceived;
   public:
    HelloClient()
        : ProtoClient() {
      m_msgReceived = "";
    }

    void hello(string name) {
      string connectionIpPort = "127.0.0.1:" + to_string(Config::HELLO_PORT);
      connectProto(connectionIpPort);
      HelloMessage sendMsg;
      sendMsg.set_name(name);
      Any sendAny;
      sendAny.PackFrom(sendMsg);

      Any rcvAny;

      ASSERT_NO_THROW(exchangeProtoMsg( connectionIpPort , &sendAny, &rcvAny));
      HelloMessage rcvMsg;
      if (rcvAny.Is<HelloMessage>()) {
        rcvAny.UnpackTo(&rcvMsg);
      }

      m_msgReceived = rcvMsg.name();
      Logging::debug(__FILE__, __LINE__, "Client received: " + rcvMsg.name());
    }

    string msgReceived() {
      return m_msgReceived;
    }

  };
  std::unique_ptr<HelloServer> m_testServer;
  std::unique_ptr<HelloClient> m_testClient;

  void SetUp() override;
};

#endif /* SRC_TEST_NET_TestProtoServer_H_ */
