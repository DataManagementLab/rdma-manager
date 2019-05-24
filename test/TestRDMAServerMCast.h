/**
 * @file TestRDMAServerMCast.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestRDMAServerMCast_H_
#define SRC_TEST_NET_TestRDMAServerMCast_H_

#include "../utils/Config.h"
#include "../../net/rdma/RDMAServer.h"
#include "../../net/rdma/RDMAClient.h"

using namespace rdma;

class TestRDMAServerMCast : public CppUnit::TestFixture {


DPI_UNIT_TEST_SUITE (TestRDMAServerMCast);
  DPI_UNIT_TEST_UD (testSendReceive);
  DPI_UNIT_TEST_UD (testSendReceiveWithIbAdress);
  DPI_UNIT_TEST_SUITE_END()
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
