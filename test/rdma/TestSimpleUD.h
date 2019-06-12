/**
 * @file TestSimpleUD.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestSimpleUD_H_
#define SRC_TEST_NET_TestSimpleUD_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"

using namespace rdma;

class TestSimpleUD : public CppUnit::TestFixture
{
    DPI_UNIT_TEST_SUITE(TestSimpleUD);
    DPI_UNIT_TEST_UD(testSendRecieve);
    // DPI_UNIT_TEST_UD(testSendRecieveMgmt);
    DPI_UNIT_TEST_SUITE_END();

  public:
    void setUp();
    void tearDown();

    void testSendRecieve();
    void testSendRecieveMgmt();

  private:
    RDMAServer *m_rdmaServer1;
    RDMAServer *m_rdmaServer2;

    RDMAClient *m_rdmaClient1;
    RDMAClient *m_rdmaClient2;

    string m_connection1;
    string m_connection2;

    NodeID m_s1_nodeId = 0;
    NodeID m_s2_nodeId = 1;

    struct testMsg
    {
        int id;
        char a;
        testMsg(int n, char t)
            : id(n),
              a(t) // Create an object of type _book.
              {};
    };
};

#endif
