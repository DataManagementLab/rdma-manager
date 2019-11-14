
#include "TestRDMAServer.h"

void TestRDMAServer::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  Config::SEQUENCER_IP = "localhost";

  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();
  // ASSERT_TRUE(m_nodeIDSequencer->startServer());

  m_rdmaServer = std::make_unique<RDMAServer<ReliableRDMA>>();
  // ASSERT_TRUE(m_rdmaServer->startServer());

  m_connection = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_rdmaClient = std::make_unique<RDMAClient<ReliableRDMA>>();

  ASSERT_TRUE(m_rdmaClient->connect(m_connection, m_nodeId));
}


TEST_F(TestRDMAServer, testWrite) {
  size_t remoteOffset = 0;
  size_t memSize = sizeof(int) * 2;

  //allocate local array
  int* localValues = (int*) m_rdmaClient->localAlloc(memSize);
  ASSERT_TRUE(localValues!=nullptr);

  //remote allocate array
  ASSERT_TRUE(
      m_rdmaClient->remoteAlloc(m_connection, memSize, remoteOffset));

  //write to remote machine
  localValues[0] = 1;
  localValues[1] = 2;
  m_rdmaClient->write(m_nodeId, remoteOffset, localValues, memSize, true);

  //read from remote machine
  int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);
  ASSERT_EQ(remoteVals[0], localValues[0]);
  ASSERT_EQ(remoteVals[1], localValues[1]);

  //remote free
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, remoteOffset));
}

TEST_F(TestRDMAServer, restRemoteAlloc) {
  size_t memSize = 10;
  size_t offset = 0;  // arbitrary value greater memSize
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize / 2, offset));
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize / 2, offset));
}

TEST_F(TestRDMAServer, testRemoteFree) {
  size_t memSize = 15;
  size_t offset = 0;  // arbitrary value greater memSize
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize, offset));
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, offset));
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize, offset));
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, offset));
}

TEST_F(TestRDMAServer, testSendRecieve) {
  // testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  // CPPUNIT_ASSERT(localstruct!=nullptr);
  // localstruct->a = 'a';
  // localstruct->id = 1;
  // testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  // CPPUNIT_ASSERT(remotestruct!=nullptr);


  // CPPUNIT_ASSERT(
  //     m_rdmaServer->receive(ibAddr, (void* )remotestruct, sizeof(testMsg)));
  // CPPUNIT_ASSERT(
  //     m_rdmaClient->send(m_nodeId,(void*)localstruct,sizeof(testMsg),true));
  // CPPUNIT_ASSERT(m_rdmaServer->pollReceive(ibAddr));

  // CPPUNIT_ASSERT_EQUAL(localstruct->id, remotestruct->id);
  // CPPUNIT_ASSERT_EQUAL(localstruct->a, remotestruct->a);
}

TEST_F(TestRDMAServer, testAtomics) {
  size_t remoteOffset = 0;
  size_t memSize = sizeof(int64_t);

  //allocate local array
  int64_t* localValues = (int64_t*) m_rdmaClient->localAlloc(memSize);
  ASSERT_TRUE(localValues!=nullptr);

  //remote allocate array
  ASSERT_TRUE(
      m_rdmaClient->remoteAlloc(m_connection, memSize, remoteOffset));

  //write to remote machine
  localValues[0] = 1;
  ASSERT_NO_THROW(m_rdmaClient->fetchAndAdd(m_nodeId,remoteOffset,localValues, sizeof(uint64_t), true));
  ASSERT_NO_THROW(m_rdmaClient->fetchAndAdd(m_nodeId,remoteOffset,localValues, sizeof(uint64_t), true));


  int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);

  ASSERT_EQ(remoteVals[0], 2);
  
  // Compare and swap to zero
  ASSERT_NO_THROW(m_rdmaClient->compareAndSwap(m_nodeId,remoteOffset,localValues,2,0, sizeof(uint64_t), true));
  ASSERT_EQ(remoteVals[0], 0);

  ASSERT_NO_THROW(m_rdmaClient->fetchAndAdd(m_nodeId,remoteOffset,localValues,10,sizeof(uint64_t), true));
  ASSERT_EQ(remoteVals[0], 10);

  //remote free
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, remoteOffset));

} 