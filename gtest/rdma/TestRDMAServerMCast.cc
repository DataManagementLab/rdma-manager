
#include "TestRDMAServerMCast.h"

void TestRDMAServerMCast::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  m_mCastAddr = Config::getIP(Config::RDMA_INTERFACE); //Multicast address must be a valid IP of one of the RNICs

  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();

  m_rdmaServer = std::make_unique<RDMAServer<UnreliableRDMA>>();

  ASSERT_NO_THROW(m_rdmaServer->joinMCastGroup(m_mCastAddr, m_serverMCastID));

  m_rdmaClient = std::make_unique<RDMAClient<UnreliableRDMA>>();
  ASSERT_NO_THROW(m_rdmaClient->joinMCastGroup(m_mCastAddr, m_clientMCastID));
}


TEST_F(TestRDMAServerMCast,testSendReceive) {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  ASSERT_NO_THROW(
      m_rdmaServer->receiveMCast(m_serverMCastID, (void* )remotestruct,
                                 sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaClient->sendMCast(m_clientMCastID,(void*)localstruct,sizeof(testMsg),true));
  
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveMCast(m_clientMCastID));

  ASSERT_EQ(localstruct->id, remotestruct->id);
  ASSERT_EQ(localstruct->a, remotestruct->a);
}

TEST_F(TestRDMAServerMCast, testMultipleConcurrentMulticast) {
  string interface = "ib1";
  string mCastAddr2 = Config::getIP(interface);
  NodeID serverMCastID2;
  NodeID clientMCastID2;

  ASSERT_NO_THROW(m_rdmaServer->joinMCastGroup(mCastAddr2, serverMCastID2));

  ASSERT_NO_THROW(m_rdmaClient->joinMCastGroup(mCastAddr2, clientMCastID2));
  

  testMsg* localstruct1 = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct1->a = 'a';
  localstruct1->id = 1;
  testMsg* remotestruct1 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  testMsg* localstruct2 = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct2->a = 'b';
  localstruct2->id = 2;
  testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  ASSERT_NO_THROW(m_rdmaServer->receiveMCast(m_serverMCastID, (void* )remotestruct1, sizeof(testMsg))); //multicast addr 1
  ASSERT_NO_THROW(m_rdmaServer->receiveMCast(serverMCastID2, (void* )remotestruct2, sizeof(testMsg))); //multicast addr 2

  ASSERT_NO_THROW(m_rdmaClient->sendMCast(m_clientMCastID,(void*)localstruct1,sizeof(testMsg),true));
  ASSERT_NO_THROW(m_rdmaClient->sendMCast(clientMCastID2,(void*)localstruct2,sizeof(testMsg),true));

  ASSERT_NO_THROW(m_rdmaServer->pollReceiveMCast(m_serverMCastID));
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveMCast(serverMCastID2));

  ASSERT_EQ(localstruct1->id, remotestruct1->id);
  ASSERT_EQ(localstruct1->a, remotestruct1->a);
  ASSERT_EQ(localstruct2->id, remotestruct2->id);
  ASSERT_EQ(localstruct2->a, remotestruct2->a);

  m_rdmaServer->leaveMCastGroup(m_serverMCastID);
}

