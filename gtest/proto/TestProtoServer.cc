#include "TestProtoServer.h"

void TestProtoServer::SetUp() {
  m_testServer = std::make_unique<HelloServer> ();
  // ASSERT_TRUE(m_testServer->startServer());

  //connect client
  m_testClient = std::make_unique<HelloClient>();
  // ASSERT_TRUE(m_testClient->connect());
}


TEST_F(TestProtoServer,testMsgExchange) {
  string msg = "";
  ASSERT_EQ(m_testServer->msgReceived(), msg);
  ASSERT_EQ(m_testClient->msgReceived(), msg);
  msg = "NewMessage";
  m_testClient->hello(msg);
  ASSERT_EQ(m_testServer->msgReceived(), msg);
  ASSERT_EQ(m_testClient->msgReceived(), msg);
}
