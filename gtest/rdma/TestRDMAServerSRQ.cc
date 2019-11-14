
#include "TestRDMAServerSRQ.h"

void TestRDMAServerSRQ::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  m_nodeId = 0;
  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();
  // ASSERT_TRUE(m_nodeIDSequencer->startServer());
  m_rdmaServer = std::make_unique<RDMAServer<ReliableRDMA>>();

// creating SRQ and hack it
  m_rdmaServer->createSharedReceiveQueue(m_srq_id);
  m_rdmaServer->activateSRQ(m_srq_id);

  // ASSERT_TRUE(m_rdmaServer->startServer());
  m_connection = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_rdmaClient_0 = std::make_unique<RDMAClient<ReliableRDMA>>();
  ASSERT_TRUE(m_rdmaClient_0->connect(m_connection, m_nodeId));
  m_rdmaClient_1 = std::make_unique<RDMAClient<ReliableRDMA>>();
  ASSERT_TRUE(m_rdmaClient_1->connect(m_connection, m_nodeId));

}



TEST_F(TestRDMAServerSRQ, testSendReceive) {

  Logging::debug("TestRDMAServerSRQ started", __LINE__, __FILE__);

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

  ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, (void* ) remotestruct, sizeof(testMsg)));
  ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, (void* ) remotestruct2, sizeof(testMsg)));
  

  ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), false));
  ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), false));

  NodeID retNodeID = 0;
  bool poll= true;
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID, poll));

  ASSERT_EQ(localstruct1->id, remotestruct->id);
  ASSERT_EQ(localstruct1->a, remotestruct->a);
}


TEST_F(TestRDMAServerSRQ, DISABLED_testPollReceiveBatch) {

  Logging::debug("TestRDMAServerSRQ started", __LINE__, __FILE__);

  testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
      sizeof(testMsg));
  localstruct1->a = 'a';
  localstruct1->id = 1;

  testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
      sizeof(testMsg));
  localstruct2->a = 'a';
  localstruct2->id = 1;

  const size_t receives = 4;

  testMsg *remotestructs[receives];
  for (size_t i = 0; i < receives; i++)
  {
    remotestructs[i] = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
    ASSERT_TRUE(remotestructs[i] != nullptr);
  }

  for (size_t i = 0; i < receives; i++)
  {
    ASSERT_NO_THROW(m_rdmaServer->receive(m_srq_id, (void* ) remotestructs[i], sizeof(testMsg)));
  }

  size_t num_received_1 = 0;
  size_t num_received_2 = 0;

  ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), true));
  ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), true));
  bool poll = true;
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveBatch(m_srq_id, num_received_1, poll));

  ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), true));
  ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), true));
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveBatch(m_srq_id, num_received_2, poll));

  ASSERT_EQ(num_received_1 + num_received_2, receives);

  for (size_t i = 0; i < receives; i++) {
    ASSERT_EQ(localstruct1->id, remotestructs[i]->id);
    ASSERT_EQ(localstruct1->a, remotestructs[i]->a);
  }
}
