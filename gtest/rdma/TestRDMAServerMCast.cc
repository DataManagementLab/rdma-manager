
#include "TestRDMAServerMCast.h"

void TestRDMAServerMCast::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  m_mCastAddr = Config::getIP(Config::RDMA_INTERFACE); //Multicast address must be a valid IP of one of the RNICs

  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();

  m_rdmaServer = std::make_unique<RDMAServer<UnreliableRDMA>>();

  ASSERT_NO_THROW(m_rdmaServer->joinMCastGroup(m_mCastAddr, m_addrServer));

  m_rdmaClient = std::make_unique<RDMAClient<UnreliableRDMA>>();
  ASSERT_NO_THROW(m_rdmaClient->joinMCastGroup(m_mCastAddr, m_addrClient));
}

void TestRDMAServerMCast::TearDown() {
    m_rdmaServer->leaveMCastGroup(m_addrServer);
    m_rdmaClient->leaveMCastGroup(m_addrClient);
}

TEST_F(TestRDMAServerMCast,testSendReceive) {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  ASSERT_NO_THROW(
      m_rdmaServer->receiveMCast(m_addrServer, (void* )remotestruct,
                                 sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaClient->sendMCast(m_addrClient,(void*)localstruct,sizeof(testMsg),true));
  
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveMCast(m_addrClient));

  ASSERT_EQ(localstruct->id, remotestruct->id);
  ASSERT_EQ(localstruct->a, remotestruct->a);
}

TEST_F(TestRDMAServerMCast, testSendReceiveWithIbAdress) {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  ASSERT_NO_THROW(
      m_rdmaServer->receiveMCast(m_addrServer, (void* )remotestruct,
                                 sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaClient->sendMCast(m_addrClient,(void*)localstruct,sizeof(testMsg),true));
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveMCast(m_addrServer));

  ASSERT_EQ(localstruct->id, remotestruct->id);
  ASSERT_EQ(localstruct->a, remotestruct->a);
}

