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
    RDMA_UNIT_TEST_SUITE(TestSimpleUD);
    RDMA_UNIT_TEST_UD(testSendRecieve);
    // RDMA_UNIT_TEST_UD(testSendRecieveMgmt);
    RDMA_UNIT_TEST_SUITE_END();

  public:
    void setUp();
    void tearDown();


    std::unique_ptr<NodeIDSequencer> m_nodeIDSequencer;

    std::unique_ptr<RDMAServer<ReliableRDMA>> m_rdmaServer1;
    std::unique_ptr<RDMAServer<ReliableRDMA>> m_rdmaServer2;

    std::unique_ptr<RDMAClient<ReliableRDMA>> m_rdmaClient1;
    std::unique_ptr<RDMAClient<ReliableRDMA>> m_rdmaClient2;

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
