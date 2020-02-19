
#include "TestRDMAServerMultClients.h"

void TestRDMAServerMultClients::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  Config::SEQUENCER_IP = "localhost";
  m_nodeId = 0;

  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();
  // ASSERT_TRUE(m_nodeIDSequencer->startServer());

  m_rdmaServer = std::make_unique<RDMAServer<ReliableRDMA>>();
  ASSERT_TRUE(m_rdmaServer->startServer());

  m_connection = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_rdmaClient_0 = std::make_unique<RDMAClient<ReliableRDMA>>();
  ASSERT_TRUE(m_rdmaClient_0->connect(m_connection, m_nodeId));
  m_rdmaClient_1 = std::make_unique<RDMAClient<ReliableRDMA>>();
  ASSERT_TRUE(m_rdmaClient_1->connect(m_connection, m_nodeId));
}


TEST_F(TestRDMAServerMultClients, testSendRecieve) {

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

  ASSERT_TRUE(remotestruct != nullptr);
  ASSERT_TRUE(remotestruct2 != nullptr);

  ASSERT_NO_THROW(
      m_rdmaServer->receive(m_rdmaClient_0->getOwnNodeID(), (void* ) remotestruct, sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaServer->receive(m_rdmaClient_1->getOwnNodeID(), (void* ) remotestruct2, sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), true));
  ASSERT_NO_THROW(
      m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), true));


  ASSERT_NO_THROW(m_rdmaServer->pollReceive(m_rdmaClient_0->getOwnNodeID(), true));
  ASSERT_NO_THROW(m_rdmaServer->pollReceive(m_rdmaClient_1->getOwnNodeID(), true));

  ASSERT_EQ(localstruct1->id, remotestruct->id);
  ASSERT_EQ(localstruct1->a, remotestruct->a);
}
