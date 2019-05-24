
#include "TestRDMAServerMCast.h"

void TestRDMAServerMCast::setUp() {
  m_mCastAddr = "192.168.1.1";

  m_rdmaServer = new RDMAServer();
  //CPPUNIT_ASSERT(rdmaServer->startServer());
  CPPUNIT_ASSERT(m_rdmaServer->joinMCastGroup(m_mCastAddr, m_addrServer));

  m_rdmaClient = new RDMAClient();
  CPPUNIT_ASSERT(m_rdmaClient->joinMCastGroup(m_mCastAddr, m_addrClient));
}

void TestRDMAServerMCast::tearDown() {
  if (m_rdmaServer != nullptr) {
    m_rdmaServer->leaveMCastGroup(m_addrServer);
    //rdmaServer->stopServer();
    delete m_rdmaServer;
    m_rdmaServer = nullptr;
  }
  if (m_rdmaClient != nullptr) {
    m_rdmaClient->leaveMCastGroup(m_addrClient);
    delete m_rdmaClient;
    m_rdmaClient = nullptr;
  }
}

void TestRDMAServerMCast::testSendReceive() {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  CPPUNIT_ASSERT(
      m_rdmaServer->receiveMCast(m_mCastAddr, (void* )remotestruct,
                                 sizeof(testMsg)));
  CPPUNIT_ASSERT(
      m_rdmaClient->sendMCast(m_mCastAddr,(void*)localstruct,sizeof(testMsg),true));
  CPPUNIT_ASSERT(m_rdmaServer->pollReceiveMCast(m_mCastAddr));

  CPPUNIT_ASSERT_EQUAL(localstruct->id, remotestruct->id);
  CPPUNIT_ASSERT_EQUAL(localstruct->a, remotestruct->a);
}

void TestRDMAServerMCast::testSendReceiveWithIbAdress() {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  CPPUNIT_ASSERT(
      m_rdmaServer->receiveMCast(m_addrServer, (void* )remotestruct,
                                 sizeof(testMsg)));
  CPPUNIT_ASSERT(
      m_rdmaClient->sendMCast(m_addrClient,(void*)localstruct,sizeof(testMsg),true));
  CPPUNIT_ASSERT(m_rdmaServer->pollReceiveMCast(m_addrServer));

  CPPUNIT_ASSERT_EQUAL(localstruct->id, remotestruct->id);
  CPPUNIT_ASSERT_EQUAL(localstruct->a, remotestruct->a);
}

