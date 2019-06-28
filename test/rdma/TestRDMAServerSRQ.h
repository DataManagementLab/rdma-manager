/**
 * @file TestRDMAServerSRQ.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestRDMAServerSRQ_H_
#define SRC_TEST_NET_TestRDMAServerSRQ_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"


using namespace rdma;

class TestRDMAServerSRQ : public CppUnit::TestFixture {
DPI_UNIT_TEST_SUITE (TestRDMAServerSRQ);
  DPI_UNIT_TEST_RC(testSendRecieve);
  // DPI_UNIT_TEST_UD(testSendRecieve); SRQ not implemented for UD yet...
DPI_UNIT_TEST_SUITE_END ()
  ;

 public:
  void setUp();
  void tearDown();

  void testSendRecieve();

 private:
  RDMAServer* m_rdmaServer;
  RDMAClient* m_rdmaClient_0;
  RDMAClient* m_rdmaClient_1;
  RDMAClient* m_rdmaClient_2;
  RDMAClient* m_rdmaClient_3;
  
  NodeID m_nodeId = 0;

  string m_connection;
  size_t m_srq_id;
  
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
