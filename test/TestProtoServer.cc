#include "TestProtoServer.h"

void TestProtoServer::setUp() {
  m_testServer = new HelloServer();
  CPPUNIT_ASSERT(m_testServer->startServer());

  //connect client
  m_testClient = new HelloClient();
  CPPUNIT_ASSERT(m_testClient->connect());
}

void TestProtoServer::tearDown() {
  m_testServer->stopServer();
  delete m_testServer;
  delete m_testClient;
}

void TestProtoServer::testMsgExchange() {
  string msg = "";
  CPPUNIT_ASSERT_EQUAL(m_testServer->msgReceived(), msg);
  CPPUNIT_ASSERT_EQUAL(m_testClient->msgReceived(), msg);
  msg = "NewMessage";
  m_testClient->hello(msg);
  CPPUNIT_ASSERT_EQUAL(m_testServer->msgReceived(), msg);
  CPPUNIT_ASSERT_EQUAL(m_testClient->msgReceived(), msg);
}
