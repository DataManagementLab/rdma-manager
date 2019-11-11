
#include "TestRDMAServerMultClients.h"

void TestRDMAServerMultClients::setUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  Config::SEQUENCER_IP = "localhost";
  m_nodeId = 0;

  m_nodeIDSequencer = new NodeIDSequencer();
  CPPUNIT_ASSERT(m_nodeIDSequencer->startServer());

  m_rdmaServer = new RDMAServer<ReliableRDMA>();
  CPPUNIT_ASSERT(m_rdmaServer->startServer());

  m_connection = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_rdmaClient_0 = new RDMAClient<ReliableRDMA>();
  CPPUNIT_ASSERT(m_rdmaClient_0->connect(m_connection, m_nodeId));
  m_rdmaClient_1 = new RDMAClient<ReliableRDMA>();
  CPPUNIT_ASSERT(m_rdmaClient_1->connect(m_connection, m_nodeId));
}

void TestRDMAServerMultClients::tearDown() {
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

void TestRDMAServerMultClients::testSendRecieve() {

  Logging::debug("TestRDMASerevrMultClients started", __LINE__, __FILE__);

  testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
      sizeof(testMsg));
  localstruct1->a = 'a';
  localstruct1->id = 1;

  testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
      sizeof(testMsg));
  localstruct2->a = 'a';
  localstruct2->id = 1;

  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  CPPUNIT_ASSERT(remotestruct != nullptr);
  CPPUNIT_ASSERT(remotestruct2 != nullptr);

  std::cout << "m_rdmaClient_0->getOwnNodeID() " << m_rdmaClient_0->getOwnNodeID() << std::endl;
  std::cout << "m_rdmaClient_1->getOwnNodeID() " << m_rdmaClient_1->getOwnNodeID() << std::endl;
  CPPUNIT_ASSERT_NO_THROW(
      m_rdmaServer->receive(m_rdmaClient_0->getOwnNodeID(), (void* ) remotestruct, sizeof(testMsg)));
  CPPUNIT_ASSERT_NO_THROW(
      m_rdmaServer->receive(m_rdmaClient_1->getOwnNodeID(), (void* ) remotestruct2, sizeof(testMsg)));

  std::cout << "Sending to nodeid: " << m_nodeId << std::endl;
  CPPUNIT_ASSERT_NO_THROW(
      m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), true));
  CPPUNIT_ASSERT_NO_THROW(
      m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), true));


  CPPUNIT_ASSERT_NO_THROW(m_rdmaServer->pollReceive(m_rdmaClient_0->getOwnNodeID(), true));
  CPPUNIT_ASSERT_NO_THROW(m_rdmaServer->pollReceive(m_rdmaClient_1->getOwnNodeID(), true));

  CPPUNIT_ASSERT_EQUAL(localstruct1->id, remotestruct->id);
  CPPUNIT_ASSERT_EQUAL(localstruct1->a, remotestruct->a);
}
