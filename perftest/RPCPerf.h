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

    struct testPage {
        int id;
        size_t offset;
        char payload[4096];

        testPage(int n, size_t t)
                : id(n),
                  offset(t)  // Create an object of type _book.
        {
        };
    };




    class TestRPCHandlerThread: public RPCVoidHandlerThread<ReliableRDMA> {
    public:
        TestRPCHandlerThread(RDMAServer<ReliableRDMA> *rdmaServer, size_t srqID,
                             size_t maxNumberMsgs):RPCVoidHandlerThread<ReliableRDMA> (rdmaServer,  srqID,sizeof(testMsg),
                                                                            maxNumberMsgs)
        { 

             localPageBuffer = m_rdmaServer->localAlloc(sizeof(testPage));
        }



        virtual void handleRDMARPCVoid(void* msg, NodeID &returnAdd) override{


            // std::cout << "local_counter  " <<  local_counter << std::endl;
            
            m_intermediateRspBufferVoid = msg;
            ((testMsg*)localPageBuffer)->id = local_counter;
            
            // std::cout << "response id  " <<  ((testMsg*)m_intermediateRspBufferVoid)->id << std::endl;
            local_counter++;
            //todo make sendreturn work

            // m_rdmaServer->send(returnAdd, (void *)m_intermediateRspBuffer,
            //                    sizeof(testMsg), true);
            // m_rdmaServer->send(returnAdd, m_intermediateRspBufferVoid, sizeof(testMsg), true);
            // std::cout << "Return address " << returnAdd << std::endl;
            m_rdmaServer->write(returnAdd, 0, localPageBuffer, sizeof(testPage), false);
        };

    void* localPageBuffer;
    size_t local_counter = 0;


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
        RDMAClient<ReliableRDMA> m_client;
        //void* m_data;
        size_t m_size;
        size_t m_iter;
        vector<string> m_conns;
        vector<NodeID> m_addr;
        //size_t* m_remOffsets;
        testPage *localresp;
        testMsg *localsend;
        size_t returnOffset;
    };

    class RPCPerf: public PerfTest {
    public:
        RPCPerf(config_t config, bool isClient);

        RPCPerf(string& region, size_t serverPort, size_t size,
                         size_t iter, size_t threads);

        ~RPCPerf();

        void printHeader() {
            cout << "Iter\t bw \tmPingPongs" << endl;
        }

        void printResults() {
            double time = (this->time()) / (1e9);
            size_t bw = (((double) sizeof(testPage) * m_iter * m_numThreads ) / (1024 * 1024)) / time;
            double mops = (((double) m_iter * m_numThreads) / time) / (1e6);

            cout  << "\t" << m_iter << "\t"  << bw << "\t" << mops << endl;

            cout <<  time  << " time" << endl;
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
        unique_ptr<NodeIDSequencer> m_nodeIDSequencer;

        vector<RPCPerfThread*> m_threads;

        RDMAServer<ReliableRDMA>* m_dServer;

        TestRPCHandlerThread *the;
    };

}




#endif //RDMACC_RPCPERF_H
