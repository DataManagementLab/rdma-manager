#include <stdio.h>
#include <vector>
#include <string>

#include <gflags/gflags.h>

#include "../src/utils/Config.h"
#include "PerfTest.h"
#include "BandwidthPerfTest.h"

DEFINE_string(test, "bandwidth", "Test: bandwidth, latency, atomics, multicast (multiples separated with space)");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_int32(gpu, -1, "Index of GPU for memory allocation (negative for main memory)");
DEFINE_uint64(memsize, 4096, "Memory size in bytes (per thread)");
DEFINE_int32(threads, 1, "Amout of threads used by client for testing");
DEFINE_uint64(iterations, 10000, "Amount of test repeats");
DEFINE_string(addr, "172.18.94.20", "Addresses of NodeIDSequencer to connect/bind to");
DEFINE_int32(port, rdma::Config::RDMA_PORT, "RDMA port");

int main(int argc, char *argv[]){
    std::cout << "Parsing arguments ..." << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::vector<std::string> testNames;
    std::stringstream testNamesRaw(FLAGS_test);
    std::string testName;
    while(std::getline(testNamesRaw, testName, ' ')){
        if(testName.length() == 0)
            continue;
        rdma::PerfTest *test = nullptr;

        if(testName.compare("bandwidth") == 0){
            // Bandwidth Test
            test = new rdma::BandwidthPerfTest(FLAGS_server, FLAGS_addr, FLAGS_port, FLAGS_gpu, FLAGS_threads, FLAGS_memsize, FLAGS_iterations);
        }
        if(test == nullptr){
            std::cerr << "No test with name '" << testName << "' found" << std::endl;
            continue;
        }

        std::cout << std::endl << "SETTING UP ENVIRONMENT FOR TEST '" << testName << "' ..." << std::endl;
        test->setupTest();

        std::cout << "RUN TEST WITH PARAMETERS:  " << test->getTestParameters() << std::endl;
        test->runTest();

        std::cout << "RESULTS: " << test->getTestResults() << std::endl << "DONE TESTING '" << testName << "'" << std::endl << std::endl;

        delete test;
    }

    return 0;

    /*
    std::cout << "Setting up environment ..." << std::endl;
    rdma::BaseMemory *m_Memory = (FLAGS_gpu ? (rdma::BaseMemory*)new rdma::CudaMemory(FLAGS_size) : (rdma::BaseMemory*)new rdma::MainMemory(FLAGS_size));

    std::cout << "Environments:  type=" << (FLAGS_server?"Server":"Client") << " | memory=" << m_Memory->getSize() << " (" << (FLAGS_gpu?"GPU":"main") << ")" << std::endl;


    if(FLAGS_server){
        // Start server
        std::cout << "Starting RDMAServer on: " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << FLAGS_port << std::endl;
        rdma::RDMAServer m_Server* = new rdma::RDMAServer<ReliableRDMA>("TestRDMAServer", FLAGS_port, m_Memory);
        m_Server->startServer();
        while (m_Server->isRunning()) {
            usleep(Config::RDMA_SLEEP_INTERVAL);
        }
        delete m_Server;

    } else {
        // Start client
        for (size_t i = 0; i < FLAGS_threads; i++) {
            RemoteMemoryPerfThread* perfThread = new RemoteMemoryPerfThread(m_conns, m_size, m_iter);
            perfThread->start();
            if (!perfThread->ready()) {
                usleep(Config::RDMA_SLEEP_INTERVAL);
            }
            m_threads.push_back(perfThread);
        }

        //wait for user input
        waitForUser();

        //send signal to run benchmark
        RemoteMemoryPerf::signaled = false;
        unique_lock < mutex > lck(RemoteMemoryPerf::waitLock);
        RemoteMemoryPerf::waitCv.notify_all();
        RemoteMemoryPerf::signaled = true;
        lck.unlock();
        for (size_t i = 0; i < m_threads.size(); i++) {
            m_threads[i]->join();
        }            
    }


    delete m_Memory;

    printf("DONE\n");
    return 0;
    */
}