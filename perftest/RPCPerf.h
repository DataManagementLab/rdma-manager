//
// Created by Tilo Gaulke on 15.10.19.
//

#ifndef RDMACC_RPCPERF_H
#define RDMACC_RPCPERF_H

#include "../utils/Config.h"
#include "../utils/StringHelper.h"
#include "../thread/Thread.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "PerfTest.h"
#include "../RPC/RPCVoidHandlerThread.h"
#include "../RPC/RPCHandlerThread.h"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {

    struct testMsg {
        int id;
        size_t offset;

        testMsg(int n, size_t t)
                : id(n),
                  offset(t)  // Create an object of type _book.
        {
        };
    };

    /*struct testPage {
        int id;
        size_t offset;
        char payload[4096];

        testPage(int n, size_t t)
                : id(n),
                  offset(t)  // Create an object of type _book.
        {
        };
    };*/

    struct testFooter{
        NodeID to;
        int64_t counter;
        int64_t idBack;
        volatile int64_t ret;
    };



    //todo fix to page response
    class TestRPCHandlerThreadOld : public RPCHandlerThreadOld<testMsg,ReliableRDMA>{
    public:

        TestRPCHandlerThreadOld(RDMAServer<ReliableRDMA> *rdmaServer, size_t srqID,size_t maxNumberMsgs,std::size_t pageSize,std::size_t returnMethod):
        RPCHandlerThreadOld (rdmaServer,srqID,maxNumberMsgs),m_pageSize(pageSize),m_returnMethod(returnMethod)
        {
            localPageBuffer = (char*)m_rdmaServer->localAlloc(m_msgSize);
        }

        void handleRDMARPC(testMsg* msg, NodeID &returnAdd) override {

            // std::cout << "local_counter  " <<  local_counter << std::endl;

            auto foot = new(localPageBuffer + m_pageSize - sizeof(testFooter)) testFooter();
            foot->counter= local_counter;
            foot->idBack = msg->id;
            foot->ret = 0;


            local_counter++;



            m_rdmaServer->write(returnAdd, 0, localPageBuffer, m_pageSize, false);
        };

        char* localPageBuffer;
        size_t local_counter = 0;
        std::size_t m_pageSize;
        std::size_t m_returnMethod;
    };





    class TestRPCHandlerThreadAb: public RPCHandlerThread<testMsg,ReliableRDMA> {
    public:
        TestRPCHandlerThreadAb(RDMAServer<ReliableRDMA> *rdmaServer, size_t srqID,
                             size_t maxNumberMsgs,std::size_t pageSize,bool signaled):
                             RPCHandlerThread (rdmaServer,  srqID,maxNumberMsgs),m_pageSize(pageSize),

                             m_signaled(signaled)
        {
            //todo maybe a timing problem here with unsignaled
            localPageBuffer = (char*)m_rdmaServer->localAlloc(m_pageSize);
        }



    char* localPageBuffer;
    size_t local_counter = 0;
    size_t m_pageSize;
    bool m_signaled;

    };

    class TestRPCHandlerThreadPoll : public TestRPCHandlerThreadAb {
    public:
        using TestRPCHandlerThreadAb::TestRPCHandlerThreadAb;

        void handleRDMARPC( testMsg *msg, NodeID &returnAdd) override {


            auto foot = new(localPageBuffer + m_pageSize - sizeof(testFooter)) testFooter();
            foot->counter = local_counter;
            foot->idBack = msg->id;
            foot->to = returnAdd;
            foot->ret = 0;

            // std::cout << "response id  " <<  ((testMsg*)m_intermediateRspBufferVoid)->id << std::endl;
            local_counter++;

            m_rdmaServer->write(returnAdd, msg->offset, localPageBuffer, m_pageSize, m_signaled);
        };
    };

    class TestRPCHandlerThreadImm : public TestRPCHandlerThreadAb {
    public:
        using TestRPCHandlerThreadAb::TestRPCHandlerThreadAb;

        void handleRDMARPC( testMsg *msg, NodeID &returnAdd) override {

            auto foot = new(localPageBuffer + m_pageSize - sizeof(testFooter)) testFooter();
            foot->counter = local_counter;
            foot->idBack = msg->id;
            foot->to = returnAdd;


            local_counter++;

            m_rdmaServer->writeImm(returnAdd, msg->offset, localPageBuffer, m_pageSize,0, m_signaled);
        };
    };

    class TestRPCHandlerThreadSend : public TestRPCHandlerThreadAb {
    public:
        using TestRPCHandlerThreadAb::TestRPCHandlerThreadAb;

        void handleRDMARPC( testMsg *msg, NodeID &returnAdd) override {

            auto foot = new(localPageBuffer + m_pageSize - sizeof(testFooter)) testFooter();
            foot->counter = local_counter;
            foot->idBack = msg->id;
            foot->to = returnAdd;

            local_counter++;
            m_rdmaServer->send(returnAdd,localPageBuffer,m_pageSize,m_signaled);

        };
    };




    class RPCPerfThread: public Thread {
    public:
        RPCPerfThread(vector<string>& conns, size_t size, size_t iter,std::size_t returnMethod,bool signaled);
        ~RPCPerfThread() override;

        void run() override;
        bool ready() const {
            return m_ready;
        }

    private:
        bool m_ready = false;
        RDMAClient<ReliableRDMA> m_client;
        size_t m_size;
        size_t m_iter;
        vector<string> m_conns;
        vector<NodeID> m_addr;
        char *localresp;
        testMsg *localsend;
        size_t returnOffset;
        size_t m_returnMethod;
        bool m_signaled;

        void poll();

        void imm();

        void send();
    };

    class RPCPerf: public PerfTest {
    public:
        RPCPerf(config_t config, bool isClient);

        RPCPerf(string& region, size_t serverPort, size_t size,
                         size_t iter, size_t threads,std::size_t returnMethod,bool old,bool signaled);

        ~RPCPerf() override;

        void printHeader() override {
            cout << "Iter\t bw MB/s \tmops" << endl;
        }

        void printResults() override {
            double time = (this->time()) / (1e9);
            size_t bw = (((double) m_size * m_iter * m_numThreads ) / (1024 * 1024)) / time;
            double mops = (((double) m_iter * m_numThreads) / time) / (1e6);

            cout  << m_iter << "\t"  << bw << "\t" << mops << endl;

            cout <<  time  << " time" << endl;
        }

        void printUsage() override {
            cout << "perf_test ... -s #servers ";
            cout << "(-p #serverPort -t #threadNum)?" << endl;
        }

        void runClient() override;
        void runServer() override;
        double time() override;

        static mutex waitLock;
        static condition_variable waitCv;
        static bool signaled;

    private:
        vector<string> m_conns;
        size_t m_serverPort;
        size_t m_size{};
        size_t m_iter;
        size_t m_numThreads;
        unique_ptr<NodeIDSequencer> m_nodeIDSequencer;

        vector<RPCPerfThread*> m_threads;

        RDMAServer<ReliableRDMA>* m_dServer{};

        RPCVoidHandlerBase<ReliableRDMA> *the{};
        bool m_old;
        size_t m_returnMethod;
        bool m_signaled;
    };

}




#endif //RDMACC_RPCPERF_H
