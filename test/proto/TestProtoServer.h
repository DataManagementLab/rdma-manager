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

#include <string>

using std::string;

using namespace rdma;

class TestProtoServer : public CppUnit::TestFixture {
RDMA_UNIT_TEST_SUITE(TestProtoServer);
  RDMA_UNIT_TEST(testMsgExchange);RDMA_UNIT_TEST_SUITE_END()
  ;
 private:
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
        : ProtoClient("127.0.0.1", Config::HELLO_PORT) {
      m_msgReceived = "";
    }

    void hello(string name) {
      HelloMessage sendMsg;
      sendMsg.set_name(name);
      Any sendAny;
      sendAny.PackFrom(sendMsg);

      Any rcvAny;

      if (!send(&sendAny, &rcvAny)) {
        Logging::fatal(__FILE__, __LINE__, "cannot send message");
        return;
      }
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
  HelloServer* m_testServer;
  HelloClient* m_testClient;

 public:
  void setUp();
  void tearDown();

  void testMsgExchange();
};

#endif /* SRC_TEST_NET_TestProtoServer_H_ */
