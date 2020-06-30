#include "PerfTest.h"
#include "BandwidthPerfTest.h"
#include "LatencyPerfTest.h"

#include "../src/utils/Config.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <ctime>
#include <bits/stdc++.h>

#include <gflags/gflags.h>

DEFINE_string(test, "bandwidth", "Test: bandwidth,latency (multiples separated by comma without space)");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_string(gpu, "-1", "Index of GPU for memory allocation (negative for main memory | multiples separated by comma without space)");
DEFINE_string(memsize, "4096", "Memory size in bytes (per thread | multiples separated by comma without space)");
DEFINE_string(threads, "1", "Amout of threads used by client for testing (multiples separated by comma without space)");
DEFINE_string(iterations, "500000", "Amount of test repeats (multiples separated by comma without space)");
DEFINE_string(addr, "172.18.94.10", "Addresses of NodeIDSequencer to connect/bind to");
DEFINE_int32(port, rdma::Config::RDMA_PORT, "RDMA port");
DEFINE_string(csv, "false", "If results should be written into a CSV file. Optionally define file name otherwise automatically");

static std::vector<std::string> parseStrList(std::string str){
    std::vector<std::string> v;
    std::stringstream ss(str);
    while((std::getline(ss, str, ','))){
        v.push_back(str);
    } return v;
}

static std::vector<int> parseIntList(std::string str){
    std::vector<int> v;
    std::stringstream ss(str);
    while((std::getline(ss, str, ','))){
        if(str.length() == 0)
            continue;
        try {
            v.push_back(std::stoi(str));
        } catch (std::exception const &e){
            std::cerr << "Could not parse integer from '" << str << "'" << std::endl;
        }
    } return v;
}
static std::vector<uint64_t> parseUInt64List(std::string str){
    std::vector<uint64_t> v;
    std::stringstream ss(str);
    while((std::getline(ss, str, ','))){
        if(str.length() == 0)
            continue;
        try {
            v.push_back((uint64_t)std::strtoull(str.c_str(), nullptr, 10));
        } catch (std::exception const &e){
            std::cerr << "Could not parse integer from '" << str << "'" << std::endl;
        }
    } return v;
}

int main(int argc, char *argv[]){
    std::cout << FLAGS_csv << std::endl;
    exit(0);

    std::cout << "Parsing arguments ..." << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::vector<std::string> testNames = parseStrList(FLAGS_test);
    std::vector<int> gpus = parseIntList(FLAGS_gpu);
    std::vector<uint64_t> memsizes = parseUInt64List(FLAGS_memsize);
    std::vector<int> thread_counts = parseIntList(FLAGS_threads);
    std::vector<uint64_t> iteration_counts = parseUInt64List(FLAGS_iterations);

    std::string csvFileName = NULL;
    if(FLAGS_csv.compare("false") != 0){
        // enable CSV
        if(FLAGS_csv.empty()){
            std::ostringstream oss;
            oss << "rdma-performance-tests-" << ((int)time(0)) << ".csv";
            csvFileName = oss.str();
        } else {
            csvFileName = FLAGS_csv;
        }
    }

    auto testIt = testNames.begin();
    while(testIt != testNames.end()){
        std::string testName = *testIt;
        if(testName.length() == 0)
            continue;
        std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);

        for(int &gpu_index : gpus){
            for(int &thread_count : thread_counts){
                for(uint64_t &iterations : iteration_counts){
                    bool csvAddHeader = true;
                    for(uint64_t &memsize : memsizes){
                        rdma::PerfTest *test = nullptr;

                        if(std::string("bandwidth").rfind(testName, 0) == 0){
                            // Bandwidth Test
                            testName = "Bandwidth";
                            test = new rdma::BandwidthPerfTest(FLAGS_server, FLAGS_addr, FLAGS_port, gpu_index, thread_count, memsize, iterations);

                        } else if(std::string("latency").rfind(testName, 0) == 0){
                            // Latency Test
                            testName = "Latency";
                            test = new rdma::LatencyPerfTest(FLAGS_server, FLAGS_addr, FLAGS_port, gpu_index, thread_count, memsize, iterations);
                        }

                        if(test == nullptr){
                            std::cerr << "No test with name '" << testName << "' found" << std::endl;
                            testIt = testNames.erase(testIt);
                            continue;
                        }

                        std::cout << std::endl << "SETTING UP ENVIRONMENT FOR TEST '" << testName << "' ..." << std::endl;
                        test->setupTest();

                        std::cout << "RUN TEST WITH PARAMETERS:  " << test->getTestParameters() << std::endl;
                        test->runTest();

                        std::cout << "RESULTS: " << test->getTestResults(csvFileName, csvAddHeader) << std::endl << "DONE TESTING '" << testName << "'" << std::endl << std::endl;

                        delete test;
                    }
                    csvAddHeader = false;
                }
            }
        }
        ++testIt;
    }
    return 0;
}