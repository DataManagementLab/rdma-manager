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
#include "../RPC/RPCHandlerThread.h"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {

    struct testMsg {
        int id;
        char a;

        testMsg(int n, char t)
                : id(n),
                  a(t)  // Create an object of type _book.
        {
        };
    };




    class TestRPCHandlerThread: public RPCHandlerThread<testMsg> {
    public:
        TestRPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                             size_t maxNumberMsgs,char* rpcbuffer):RPCHandlerThread<testMsg>
                                                                           (rdmaServer,  srqID,
                                                                            maxNumberMsgs,rpcbuffer)
        { }

        TestRPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                             size_t maxNumberMsgs):RPCHandlerThread<testMsg>
                                                           (rdmaServer, srqID,
                                                            maxNumberMsgs)
        { }

        TestRPCHandlerThread(RDMAServer *rdmaServer,
                             size_t maxNumberMsgs):RPCHandlerThread<testMsg>
                                                           (rdmaServer,
                                                            maxNumberMsgs)
        { }



        void handleRDMARPC(testMsg* msg, ib_addr_t &returnAdd){

            m_intermediateRspBuffer->id= msg->id;
            m_intermediateRspBuffer->a = msg->a;

            //todo
            //signaled auf false?
            m_rdmaServer->send(returnAdd, (void *)m_intermediateRspBuffer,
                               sizeof(testMsg), true);

        };

    };


    class RPCPerfThread: public Thread {
    public:
        RPCPerfThread(vector<string>& conns, size_t size, size_t iter);
        ~RPCPerfThread();
        void run();
        bool ready() {
            return m_ready;
        }

    private:
        bool m_ready = false;
        RDMAClient m_client;
        //void* m_data;
        size_t m_size;
        size_t m_iter;
        vector<string> m_conns;
        vector<NodeID> m_addr;
        //size_t* m_remOffsets;
        testMsg *localresp;
        testMsg *localsend;
    };

    class RPCPerf: public PerfTest {
    public:
        RPCPerf(config_t config, bool isClient);

        RPCPerf(string& region, size_t serverPort, size_t size,
                         size_t iter, size_t threads);

        ~RPCPerf();

        void printHeader() {
            cout << "Iter\t bw \tmopsPerS RPC calls" << endl;
        }

        void printResults() {
            double time = (this->time()) / (1e9);
            size_t bw = (((double) sizeof(testMsg) * m_iter * m_numThreads ) / (1024 * 1024)) / time;
            double mops = (((double) 2* m_iter * m_numThreads) / time) / (1e6);

            cout  << "\t" << m_iter << "\t"  << bw << "\t" << mops << endl;

            cout <<  time  << "time" << endl;
        }

        void printUsage() {
            cout << "perf_test ... -s #servers ";
            cout << "(-p #serverPort -t #threadNum)?" << endl;
        }

        void runClient();
        void runServer();
        double time();

        static mutex waitLock;
        static condition_variable waitCv;
        static bool signaled;

    private:
        vector<string> m_conns;
        size_t m_serverPort;
        size_t m_size;
        size_t m_iter;
        size_t m_numThreads;

        vector<RPCPerfThread*> m_threads;

        RDMAServer* m_dServer;

        TestRPCHandlerThread *the;
    };

}




#endif //RDMACC_RPCPERF_H
