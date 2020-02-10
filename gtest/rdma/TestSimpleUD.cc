
#include "TestSimpleUD.h"

void TestSimpleUD::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  Config::SEQUENCER_IP = rdma::Config::getIP(rdma::Config::RDMA_INTERFACE);

  m_s1_nodeId = 0;
  m_s2_nodeId = 1;

  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();


  // run two servers
  m_rdmaServer1 = std::make_unique<RDMAServer<UnreliableRDMA>>("Server1", Config::RDMA_PORT);
  m_rdmaServer2 = std::make_unique<RDMAServer<UnreliableRDMA>>("Server2", Config::RDMA_PORT + 1);
  ASSERT_TRUE(m_rdmaServer1->startServer());
  ASSERT_TRUE(m_rdmaServer2->startServer());

  m_rdmaClient1 = std::make_unique<RDMAClient<UnreliableRDMA>>();
  m_rdmaClient2 = std::make_unique<RDMAClient<UnreliableRDMA>>();

  // connect each client to both servers
  m_connection1 = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_connection2 = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT + 1);
  ASSERT_TRUE(m_rdmaClient1->connect(m_connection1, m_s1_nodeId));  // with management queue
  ASSERT_TRUE(m_rdmaClient1->connect(m_connection2, m_s2_nodeId));  // with management queue
  ASSERT_TRUE(m_rdmaClient2->connect(m_connection1, m_s1_nodeId));  // with management queue
  ASSERT_TRUE(m_rdmaClient2->connect(m_connection2, m_s2_nodeId));  // with management queue
}


TEST_F(TestSimpleUD, testSendRecieve) {
  int* message11 = (int*) m_rdmaClient1->localAlloc(sizeof(int));
  int* message21 = (int*) m_rdmaClient1->localAlloc(sizeof(int));
  *message11 = 0;
  *message21 = 0;

  int* message12 = (int*) m_rdmaClient2->localAlloc(sizeof(int));
  int* message22 = (int*) m_rdmaClient2->localAlloc(sizeof(int));
  *message12 = 0;
  *message22 = 0;

  int* remoteMsg11 = (int*) m_rdmaServer1->localAlloc(sizeof(int));
  int* remoteMsg21 = (int*) m_rdmaServer1->localAlloc(sizeof(int));
  *remoteMsg11 = 1;
  *remoteMsg21 = 1;

  int* remoteMsg12 = (int*) m_rdmaServer2->localAlloc(sizeof(int));
  int* remoteMsg22 = (int*) m_rdmaServer2->localAlloc(sizeof(int));
  *remoteMsg12 = 1;
  *remoteMsg22 = 1;

  // client post receive requests (first client)
  ASSERT_NO_THROW(
      m_rdmaClient1->receive(m_s1_nodeId, (void* ) message11, sizeof(int)));
  ASSERT_NO_THROW(
      m_rdmaClient1->receive(m_s2_nodeId, (void* ) message21, sizeof(int)));

  // client post receive requests (second client)
  ASSERT_NO_THROW(
      m_rdmaClient2->receive(m_s1_nodeId, (void* ) message12, sizeof(int)));
  ASSERT_NO_THROW(
      m_rdmaClient2->receive(m_s2_nodeId, (void* ) message22, sizeof(int)));

  // each server send management messages
  ASSERT_NO_THROW(
      m_rdmaServer1->send(m_rdmaClient1->getOwnNodeID(), (void*)remoteMsg11, sizeof(int), true));
  ASSERT_NO_THROW(
      m_rdmaServer1->send(m_rdmaClient2->getOwnNodeID(), (void*)remoteMsg21, sizeof(int), true));

  ASSERT_NO_THROW(
      m_rdmaServer2->send(m_rdmaClient1->getOwnNodeID(), (void*)remoteMsg12, sizeof(int), true));
  ASSERT_NO_THROW(
      m_rdmaServer2->send(m_rdmaClient2->getOwnNodeID(), (void*)remoteMsg22, sizeof(int), true));

  bool poll = true;
  // client 1 pulls
  ASSERT_NO_THROW(m_rdmaClient1->pollReceive(m_s1_nodeId, poll));
  ASSERT_NO_THROW(m_rdmaClient1->pollReceive(m_s2_nodeId, poll));

  // client 2 pulls
  ASSERT_NO_THROW(m_rdmaClient2->pollReceive(m_s1_nodeId, poll));
  ASSERT_NO_THROW(m_rdmaClient2->pollReceive(m_s2_nodeId, poll));

  //see if sent was successful
  ASSERT_EQ(*message11, 1);
  ASSERT_EQ(*message21, 1);
  ASSERT_EQ(*message12, 1);
  //CPPUNIT_ASSERT_EQUAL(*message22,1);
}

