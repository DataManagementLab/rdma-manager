#include "TestProtoServer.h"

void TestProtoServer::SetUp() {
  m_testServer = new HelloServer();
  ASSERT_TRUE(m_testServer->startServer());

  //connect client
  m_testClient = new HelloClient();
  // ASSERT_TRUE(m_testClient->connect());
}

void TestProtoServer::TearDown() {
  m_testServer->stopServer();
  delete m_testServer;
  delete m_testClient;
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
