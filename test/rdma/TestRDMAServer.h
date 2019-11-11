/**
 * @file TestRDMAServer.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestRDMAServer_H_
#define SRC_TEST_NET_TestRDMAServer_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"

using namespace rdma;

class TestRDMAServer : public CppUnit::TestFixture {
RDMA_UNIT_TEST_SUITE (TestRDMAServer);
  RDMA_UNIT_TEST_RC(testWrite);
  RDMA_UNIT_TEST_RC(testRemoteAlloc);
  RDMA_UNIT_TEST_RC(testRemoteFree);
  RDMA_UNIT_TEST_RC(testSendRecieve);
  RDMA_UNIT_TEST_RC(testAtomics);
  // RDMA_UNIT_TEST_UD(testSendRecieve);
  RDMA_UNIT_TEST_SUITE_END()
  ;

 public:
  void setUp();
  void tearDown();

  void testWrite();
  void testRemoteAlloc();
  void testRemoteFree();
  void testSendRecieve();
  void testAtomics();

 private:
  RDMAServer<ReliableRDMA>* m_rdmaServer;
  RDMAClient<ReliableRDMA>* m_rdmaClient;
  NodeIDSequencer *m_nodeIDSequencer;
  string m_connection;

  NodeID m_nodeId = 0;
  
  struct testMsg {
  int id;
  char a;
  testMsg(int n, char t)
      : id(n),
        a(t)  // Create an object of type _book.
  {
  };
};

};

#endif /* SRC_TEST_NET_TESTDATANODE_H_ */
