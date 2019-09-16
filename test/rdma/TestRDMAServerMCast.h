/**
 * @file TestRDMAServerMCast.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestRDMAServerMCast_H_
#define SRC_TEST_NET_TestRDMAServerMCast_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"

using namespace rdma;

class TestRDMAServerMCast : public CppUnit::TestFixture {


RDMA_UNIT_TEST_SUITE (TestRDMAServerMCast);
  RDMA_UNIT_TEST_UD (testSendReceive);
  RDMA_UNIT_TEST_UD (testSendReceiveWithIbAdress);
  RDMA_UNIT_TEST_SUITE_END()
  ;

 public:
  void setUp();
  void tearDown();

  void testSendReceive();
  void testSendReceiveWithIbAdress();

 private:
  RDMAServer* m_rdmaServer;
  RDMAClient* m_rdmaClient;
  string m_mCastAddr;
  struct ib_addr_t m_addrClient;
  struct ib_addr_t m_addrServer;

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

#endif
