
#include "TestRDMAServer.h"

void TestRDMAServer::setUp() {
  m_rdmaServer = new RDMAServer();
  CPPUNIT_ASSERT(m_rdmaServer->startServer());

  m_connection = "127.0.0.1:" + to_string(Config::RDMA_PORT);
  m_rdmaClient = new RDMAClient();
  CPPUNIT_ASSERT(m_rdmaClient->connect(m_connection));
}

void TestRDMAServer::tearDown() {
  if (m_rdmaServer != nullptr) {
    m_rdmaServer->stopServer();
    delete m_rdmaServer;
    m_rdmaServer = nullptr;
  }
  if (m_rdmaClient != nullptr) {
    delete m_rdmaClient;
    m_rdmaClient = nullptr;
  }
}

void TestRDMAServer::testWrite() {
  size_t remoteOffset = 0;
  size_t memSize = sizeof(int) * 2;

  //allocate local array
  int* localValues = (int*) m_rdmaClient->localAlloc(memSize);
  CPPUNIT_ASSERT(localValues!=nullptr);

  //remote allocate array
  CPPUNIT_ASSERT(
      m_rdmaClient->remoteAlloc(m_connection, memSize, remoteOffset));

  //write to remote machine
  localValues[0] = 1;
  localValues[1] = 2;
  m_rdmaClient->write(m_connection, remoteOffset, localValues, memSize, true);

  //read from remote machine
  int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);
  CPPUNIT_ASSERT_EQUAL(remoteVals[0], localValues[0]);
  CPPUNIT_ASSERT_EQUAL(remoteVals[1], localValues[1]);

  //remote free
  CPPUNIT_ASSERT(m_rdmaClient->remoteFree(m_connection, memSize, remoteOffset));
}

void TestRDMAServer::testRemoteAlloc() {
  size_t memSize = 10;
  size_t offset = 0;  // arbitrary value greater memSize
  CPPUNIT_ASSERT(m_rdmaClient->remoteAlloc(m_connection, memSize / 2, offset));
  CPPUNIT_ASSERT(m_rdmaClient->remoteAlloc(m_connection, memSize / 2, offset));
}

void TestRDMAServer::testRemoteFree() {
  size_t memSize = 15;
  size_t offset = 0;  // arbitrary value greater memSize
  CPPUNIT_ASSERT(m_rdmaClient->remoteAlloc(m_connection, memSize, offset));
  CPPUNIT_ASSERT(m_rdmaClient->remoteFree(m_connection, memSize, offset));
  CPPUNIT_ASSERT(m_rdmaClient->remoteAlloc(m_connection, memSize, offset));
  CPPUNIT_ASSERT(m_rdmaClient->remoteFree(m_connection, memSize, offset));
}

void TestRDMAServer::testSendRecieve() {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  CPPUNIT_ASSERT(localstruct!=nullptr);
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  CPPUNIT_ASSERT(remotestruct!=nullptr);

  ib_addr_t ibAddr = m_rdmaServer->getQueues()[0];
  CPPUNIT_ASSERT(
      m_rdmaServer->receive(ibAddr, (void* )remotestruct, sizeof(testMsg)));
  CPPUNIT_ASSERT(
      m_rdmaClient->send(m_connection,(void*)localstruct,sizeof(testMsg),true));
  CPPUNIT_ASSERT(m_rdmaServer->pollReceive(ibAddr));

  CPPUNIT_ASSERT_EQUAL(localstruct->id, remotestruct->id);
  CPPUNIT_ASSERT_EQUAL(localstruct->a, remotestruct->a);
}

