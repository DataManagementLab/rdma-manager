//
// Created by Tilo Gaulke on 2019-06-05.
//

#include "TestRPC.h"

#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"
#include "../../src/RPC/RPCMemory.h"
#include "../../src/utils/Config.h"




using namespace rdma;


void TestRPC::setUp() {

    const static size_t MAX_NUM_RPC_MSG = 32;  // Number of RDMA send MSGs RPCHandler can recv

    Config::RDMA_MEMSIZE = 1024 * 1024 * 16;

    m_rdmaServer = new RDMAServer();
    CPPUNIT_ASSERT(m_rdmaServer->startServer());

    the = new TestRPCHandlerThread(m_rdmaServer,MAX_NUM_RPC_MSG);

    m_connection = "127.0.0.1:" + to_string(Config::RDMA_PORT);
    m_rdmaClient_0 = new RDMAClient();
    m_rdmaClient_1 = new RDMAClient();
    CPPUNIT_ASSERT(m_rdmaClient_0->connect(m_connection, m_nodeId));
    CPPUNIT_ASSERT(m_rdmaClient_1->connect(m_connection, m_nodeId));

    localresp = (testMsg*) m_rdmaClient_0->localAlloc(sizeof(testMsg));
    localstruct = (testMsg*) m_rdmaClient_0->localAlloc(sizeof(testMsg));

    localresp_1 = (testMsg*) m_rdmaClient_1->localAlloc(sizeof(testMsg));
    localstruct_1 = (testMsg*) m_rdmaClient_1->localAlloc(sizeof(testMsg));

    the->startHandler();
}

void TestRPC::tearDown() {
    the->stopHandler();

    delete the;

    m_rdmaClient_0->localFree(localresp);
    m_rdmaClient_0->localFree(localstruct);

    m_rdmaClient_1->localFree(localresp_1);
    m_rdmaClient_1->localFree(localstruct_1);


    if (m_rdmaServer != nullptr) {
        m_rdmaServer->stopServer();
        delete m_rdmaServer;
        m_rdmaServer = nullptr;
    }
    if (m_rdmaClient_0 != nullptr) {
        delete m_rdmaClient_0;
        m_rdmaClient_0 = nullptr;
    }

    if (m_rdmaClient_1 != nullptr) {
        delete m_rdmaClient_1;
        m_rdmaClient_1 = nullptr;
    }

}

void TestRPC::testRPC() {


    for (int i = 0; i < 100; ++i) {
        localstruct->id = i;
        localstruct->a = 'B';

        localstruct_1->id = 100-i;
        localstruct_1->a = 'C';

        m_rdmaClient_0->receive(m_nodeId, (void*) localresp, sizeof(testMsg));
        m_rdmaClient_0->send(m_nodeId, (void*) localstruct, sizeof(testMsg), true);



        auto ret = m_rdmaClient_0->pollReceive(m_nodeId);
        if(ret){

        }

        m_rdmaClient_1->receive(m_nodeId, (void*) localresp_1, sizeof(testMsg));
        m_rdmaClient_1->send(m_nodeId, (void*) localstruct_1, sizeof(testMsg), true);



        auto ret_1 = m_rdmaClient_1->pollReceive(m_nodeId);
        if(ret_1){

        }

        CPPUNIT_ASSERT_EQUAL(i+1,localresp->id);
        CPPUNIT_ASSERT_EQUAL(localstruct->id+1,localresp->id );

        CPPUNIT_ASSERT_EQUAL( 'A',localresp->a);
        CPPUNIT_ASSERT_EQUAL( (char)(localstruct->a -1),localresp->a);


        CPPUNIT_ASSERT_EQUAL((100-i)+1,localresp_1->id);
        CPPUNIT_ASSERT_EQUAL(localstruct_1->id+1,localresp_1->id );

        CPPUNIT_ASSERT_EQUAL( 'B',localresp_1->a);
        CPPUNIT_ASSERT_EQUAL( (char)(localstruct_1->a -1),localresp_1->a);
    }






}
