#include <stdio.h>
#include <vector>
#include <string>
#include <bits/stdc++.h>

#include <gflags/gflags.h>

#include "../src/utils/Config.h"
#include "PerfTest.h"
#include "BandwidthPerfTest.h"
#include "LatencyPerfTest.h"

DEFINE_string(test, "bandwidth", "Test: bandwidth,latency (multiples separated by comma without space)");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_int32(gpu, -1, "Index of GPU for memory allocation (negative for main memory)");
DEFINE_uint64(memsize, 4 * 4096, "Memory size in bytes (per thread)");
DEFINE_int32(threads, 1, "Amout of threads used by client for testing");
DEFINE_uint64(iterations, 500000, "Amount of test repeats");
DEFINE_string(addr, "172.18.94.10", "Addresses of NodeIDSequencer to connect/bind to");
DEFINE_int32(port, rdma::Config::RDMA_PORT, "RDMA port");

int main(int argc, char *argv[]){
    std::cout << "Parsing arguments ..." << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::vector<std::string> testNames;
    std::stringstream testNamesRaw(FLAGS_test);
    std::string testName;
    while(std::getline(testNamesRaw, testName, ',')){
        if(testName.length() == 0)
            continue;
        std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);
        rdma::PerfTest *test = nullptr;

        if(std::string("bandwidth").rfind(testName, 0) == 0){
            // Bandwidth Test
            testName = "Bandwidth";
            test = new rdma::BandwidthPerfTest(FLAGS_server, FLAGS_addr, FLAGS_port, FLAGS_gpu, FLAGS_threads, FLAGS_memsize, FLAGS_iterations);

        } else if(std::string("latency").rfind(testName, 0) == 0){
            // Latency Test
            testName = "Latency";
            test = new rdma::LatencyPerfTest(FLAGS_server, FLAGS_addr, FLAGS_port, FLAGS_gpu, FLAGS_threads, FLAGS_memsize, FLAGS_iterations);
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
}