
#include "TestRDMAServerSRQ.h"

void TestRDMAServerSRQ::setUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  m_nodeId = 0;
  m_rdmaServer = new RDMAServer();
  CPPUNIT_ASSERT(m_rdmaServer->startServer());


  //Create SRQ
  CPPUNIT_ASSERT(m_rdmaServer->createSRQ(m_srq_id));

  m_rdmaServer->activateSRQ(m_srq_id);

  m_connection = "127.0.0.1:" + to_string(Config::RDMA_PORT);
  m_rdmaClient_0 = new RDMAClient();
  CPPUNIT_ASSERT(m_rdmaClient_0->connect(m_connection, m_nodeId));
  m_rdmaClient_1 = new RDMAClient();
  CPPUNIT_ASSERT(m_rdmaClient_1->connect(m_connection, m_nodeId));
}

void TestRDMAServerSRQ::tearDown() {
  if (m_rdmaServer != nullptr) {
    m_rdmaServer->stopServer();
    delete m_rdmaServer;
    m_rdmaServer = nullptr;
  }
  if (m_rdmaClient_0 != nullptr) {
    delete m_rdmaClient_0;
    m_rdmaClient_0 = nullptr;
  }

  if (m_rdmaClient_1 != nullptr) {
    delete m_rdmaClient_1;
    m_rdmaClient_1 = nullptr;
  }
}

void TestRDMAServerSRQ::testSendRecieve() {

  Logging::debug("TestRDMAServerSRQ started", __LINE__, __FILE__);

  testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
      sizeof(testMsg));
  localstruct1->a = 'a';
  localstruct1->id = 1;

  testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
      sizeof(testMsg));
  localstruct2->a = 'a';
  localstruct2->id = 1;

  vector<ib_addr_t> connKeys = m_rdmaServer->getQueues();

  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  CPPUNIT_ASSERT(remotestruct != nullptr);
  CPPUNIT_ASSERT(remotestruct2 != nullptr);

  CPPUNIT_ASSERT(m_rdmaServer->receive(m_srq_id, (void* ) remotestruct, sizeof(testMsg)));
  CPPUNIT_ASSERT(m_rdmaServer->receive(m_srq_id, (void* ) remotestruct2, sizeof(testMsg)));
  
  CPPUNIT_ASSERT(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), false));
  CPPUNIT_ASSERT(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), false));

  ib_addr_t ret_ib_addr;
  CPPUNIT_ASSERT(m_rdmaServer->pollReceive(m_srq_id, ret_ib_addr));

  CPPUNIT_ASSERT_EQUAL(localstruct1->id, remotestruct->id);
  CPPUNIT_ASSERT_EQUAL(localstruct1->a, remotestruct->a);
}
