
#include "TestSimpleUD.h"

void TestSimpleUD::setUp() {
  // run two servers
  m_rdmaServer1 = new RDMAServer(Config::RDMA_PORT);
  m_rdmaServer2 = new RDMAServer(Config::RDMA_PORT + 1);

  CPPUNIT_ASSERT(m_rdmaServer1->startServer());
  CPPUNIT_ASSERT(m_rdmaServer2->startServer());

  m_rdmaClient1 = new RDMAClient();
  m_rdmaClient2 = new RDMAClient();

  // connect each client to both servers
  m_connection1 = "127.0.0.1:" + to_string(Config::RDMA_PORT);
  m_connection2 = "127.0.0.1:" + to_string(Config::RDMA_PORT + 1);
  CPPUNIT_ASSERT(m_rdmaClient1->connect(m_connection1, true));  // with management queue
  CPPUNIT_ASSERT(m_rdmaClient1->connect(m_connection2, true));  // with management queue
  CPPUNIT_ASSERT(m_rdmaClient2->connect(m_connection1, true));  // with management queue
  CPPUNIT_ASSERT(m_rdmaClient2->connect(m_connection2, true));  // with management queue
}

void TestSimpleUD::tearDown() {
  if (m_rdmaServer1 != nullptr) {
    m_rdmaServer1->stopServer();
    delete m_rdmaServer1;
    m_rdmaServer1 = nullptr;
  }
  if (m_rdmaClient1 != nullptr) {
    delete m_rdmaClient1;
    m_rdmaClient1 = nullptr;
  }
  if (m_rdmaServer2 != nullptr) {
    m_rdmaServer2->stopServer();
    delete m_rdmaServer2;
    m_rdmaServer2 = nullptr;
  }
  if (m_rdmaClient2 != nullptr) {
    delete m_rdmaClient2;
    m_rdmaClient2 = nullptr;
  }
}

void TestSimpleUD::testSendRecieve() {
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
  CPPUNIT_ASSERT(
      m_rdmaClient1->receive(m_connection1, (void* ) message11, sizeof(int)));
  CPPUNIT_ASSERT(
      m_rdmaClient1->receive(m_connection2, (void* ) message21, sizeof(int)));

  // client post receive requests (second client)
  CPPUNIT_ASSERT(
      m_rdmaClient2->receive(m_connection1, (void* ) message12, sizeof(int)));
  CPPUNIT_ASSERT(
      m_rdmaClient2->receive(m_connection2, (void* ) message22, sizeof(int)));

  // each server send management messages
  vector<ib_addr_t> serverConnKeys1 = m_rdmaServer1->getQueues();
  CPPUNIT_ASSERT(
      m_rdmaServer1->send(serverConnKeys1[0], (void*)remoteMsg11, sizeof(int), true));
  CPPUNIT_ASSERT(
      m_rdmaServer1->send(serverConnKeys1[1], (void*)remoteMsg21, sizeof(int), true));

  vector<ib_addr_t> serverConnKeys2 = m_rdmaServer2->getQueues();
  CPPUNIT_ASSERT(
      m_rdmaServer2->send(serverConnKeys2[0], (void*)remoteMsg12, sizeof(int), true));
  CPPUNIT_ASSERT(
      m_rdmaServer2->send(serverConnKeys2[1], (void*)remoteMsg22, sizeof(int), true));

  // client 1 pulls
  CPPUNIT_ASSERT(m_rdmaClient1->pollReceive(m_connection1));
  CPPUNIT_ASSERT(m_rdmaClient1->pollReceive(m_connection2));

  // client 2 pulls
  CPPUNIT_ASSERT(m_rdmaClient2->pollReceive(m_connection1));
  CPPUNIT_ASSERT(m_rdmaClient2->pollReceive(m_connection2));

  //see if sent was successful
  CPPUNIT_ASSERT_EQUAL(*message11, 1);
  CPPUNIT_ASSERT_EQUAL(*message21, 1);
  CPPUNIT_ASSERT_EQUAL(*message12, 1);
  //CPPUNIT_ASSERT_EQUAL(*message22,1);
}

void TestSimpleUD::testSendRecieveMgmt() {
  int* mgmMessage11 = (int*) m_rdmaClient1->localAlloc(sizeof(int));
  int* mgmMessage21 = (int*) m_rdmaClient1->localAlloc(sizeof(int));
  *mgmMessage11 = 0;
  *mgmMessage21 = 0;

  int* mgmMessage12 = (int*) m_rdmaClient2->localAlloc(sizeof(int));
  int* mgmMessage22 = (int*) m_rdmaClient2->localAlloc(sizeof(int));
  *mgmMessage12 = 0;
  *mgmMessage22 = 0;

  int* remoteMsg11 = (int*) m_rdmaServer1->localAlloc(sizeof(int));
  int* remoteMsg21 = (int*) m_rdmaServer1->localAlloc(sizeof(int));
  *remoteMsg11 = 1;
  *remoteMsg21 = 1;

  int* remoteMsg12 = (int*) m_rdmaServer2->localAlloc(sizeof(int));
  int* remoteMsg22 = (int*) m_rdmaServer2->localAlloc(sizeof(int));
  *remoteMsg12 = 1;
  *remoteMsg22 = 1;

  // client post receive requests (first client)
  ib_addr_t clientMgmtQP1 = m_rdmaClient1->getMgmtQueue(m_connection1);  // for the first server
  CPPUNIT_ASSERT(
      m_rdmaClient1->receive(clientMgmtQP1, (void* ) mgmMessage11,
                             sizeof(int)));

  ib_addr_t clientMgmtQP2 = m_rdmaClient1->getMgmtQueue(m_connection2);  // for the second server
  CPPUNIT_ASSERT(
      m_rdmaClient1->receive(clientMgmtQP2, (void* ) mgmMessage21,
                             sizeof(int)));

  // client post receive requests (second client)
  ib_addr_t clientMgmtQP3 = m_rdmaClient2->getMgmtQueue(m_connection1);  // for the first server
  CPPUNIT_ASSERT(
      m_rdmaClient2->receive(clientMgmtQP3, (void* ) mgmMessage12,
                             sizeof(int)));

  ib_addr_t clientMgmtQP4 = m_rdmaClient2->getMgmtQueue(m_connection2);  // for the second server
  CPPUNIT_ASSERT(
      m_rdmaClient2->receive(clientMgmtQP4, (void* ) mgmMessage22,
                             sizeof(int)));

  // each server send management messages
  vector<ib_addr_t> serverConnKeys1 = m_rdmaServer1->getQueues();
  ib_addr_t servMgmtQP1 = m_rdmaServer1->getMgmtQueue(serverConnKeys1[0]);
  CPPUNIT_ASSERT(
      m_rdmaServer1->send(servMgmtQP1, (void*)remoteMsg11, sizeof(int), true));

  ib_addr_t servMgmtQP2 = m_rdmaServer1->getMgmtQueue(serverConnKeys1[1]);
  CPPUNIT_ASSERT(
      m_rdmaServer1->send(servMgmtQP2, (void*)remoteMsg21, sizeof(int), true));

  vector<ib_addr_t> serverConnKeys2 = m_rdmaServer2->getQueues();
  ib_addr_t servMgmtQP3 = m_rdmaServer2->getMgmtQueue(serverConnKeys2[0]);
  CPPUNIT_ASSERT(
      m_rdmaServer2->send(servMgmtQP3, (void*)remoteMsg12, sizeof(int), true));

  ib_addr_t servMgmtQP4 = m_rdmaServer2->getMgmtQueue(serverConnKeys2[1]);
  CPPUNIT_ASSERT(
      m_rdmaServer2->send(servMgmtQP4, (void*)remoteMsg22, sizeof(int), true));

  // client 1 pulls
  CPPUNIT_ASSERT(m_rdmaClient1->pollReceive(clientMgmtQP1));
  CPPUNIT_ASSERT(m_rdmaClient1->pollReceive(clientMgmtQP2));

  // client 2 pulls
  CPPUNIT_ASSERT(m_rdmaClient2->pollReceive(clientMgmtQP3));
  CPPUNIT_ASSERT(m_rdmaClient2->pollReceive(clientMgmtQP4));

  //see if sent was successful
  CPPUNIT_ASSERT_EQUAL(*mgmMessage11, 1);
  CPPUNIT_ASSERT_EQUAL(*mgmMessage21, 1);
  CPPUNIT_ASSERT_EQUAL(*mgmMessage12, 1);
  //CPPUNIT_ASSERT_EQUAL(*mgmMessage22,1);
}

