#include "PerfTest.h"
#include "BandwidthPerfTest.h"
#include "AtomicsBandwidthPerfTest.h"
#include "LatencyPerfTest.h"
#include "AtomicsLatencyPerfTest.h"

#include "../src/utils/Config.h"
#include "../src/utils/StringHelper.h"
#include "../src/rdma/NodeIDSequencer.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <ctime>
#include <bits/stdc++.h>

#include <gflags/gflags.h>

DEFINE_bool(fulltest, false, "Overwrites flags 'test, gpu, memsize, threads, iterations, csv' to execute a broad variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_string(test, "bandwidth", "Test: bandwidth, latency, atomicsbandwidth, atomicslatency (multiples separated by comma without space)");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_string(gpu, "-1", "Index of GPU for memory allocation (negative for main memory | multiples separated by comma without space)");
DEFINE_string(memsize, "4096", "Memory size in bytes (per thread | multiples separated by comma without space)");
DEFINE_string(threads, "1", "Amout of threads used by client for testing (multiples separated by comma without space)");
DEFINE_string(iterations, "500000", "Amount of test repeats (multiples separated by comma without space)");
DEFINE_string(addr, "172.18.94.20", "Addresses of NodeIDSequencer to connect/bind to");
DEFINE_int32(port, rdma::Config::RDMA_PORT, "RDMA port");
DEFINE_bool(csv, false, "Results will be written into an automatically generated CSV file");
DEFINE_string(csvfile, "", "Results will be written into a given CSV file");

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


static void runTest(size_t testNumber, size_t testIterations, std::string testName, rdma::PerfTest *test, std::string csvFileName, bool csvAddHeader){
    std::cout << std::endl << "TEST " << testNumber << " / " << testIterations << " (" << (testNumber*100/testIterations) << "%)" << std::endl;
    std::cout << "SETTING UP ENVIRONMENT FOR TEST '" << testName << "' ..." << std::endl;
    test->setupTest();

    std::cout << "RUN TEST WITH PARAMETERS:  " << test->getTestParameters() << std::endl;
    test->runTest();

    std::cout << "RESULTS: " << test->getTestResults(csvFileName, csvAddHeader) << std::endl << "DONE TESTING '" << testName << "'" << std::endl << std::endl;

    delete test;

    if(!FLAGS_server)
        usleep(100000); // if client then sleep for 100ms to wait for server to restart
}



int main(int argc, char *argv[]){
    std::cout << "Parsing arguments ..." << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::vector<std::string> testNames = rdma::StringHelper::split(FLAGS_test);
    std::vector<int> gpus = parseIntList(FLAGS_gpu);
    std::vector<uint64_t> memsizes = parseUInt64List(FLAGS_memsize);
    std::vector<int> thread_counts = parseIntList(FLAGS_threads);
    std::vector<uint64_t> iteration_counts = parseUInt64List(FLAGS_iterations);
    std::vector<std::string> addresses = rdma::StringHelper::split(FLAGS_addr);
	for (auto &addr : addresses){
		addr += ":" + to_string(FLAGS_port);
	}
    
    if(FLAGS_fulltest){
        FLAGS_csv = true;
        testNames.clear(); testNames.push_back("bandwidth"); testNames.push_back("latency");
        testNames.push_back("atomicsbandwidth"); testNames.push_back("atomicslatency");
        memsizes.clear(); memsizes.push_back(64); memsizes.push_back(512); memsizes.push_back(1024);
        memsizes.push_back(2048); memsizes.push_back(4096); memsizes.push_back(8192);
        memsizes.push_back(16384); memsizes.push_back(32768); memsizes.push_back(65536);
        // TODO REDO thread_counts.clear(); thread_counts.push_back(1); thread_counts.push_back(2); thread_counts.push_back(4); thread_counts.push_back(8);
        iteration_counts.clear(); iteration_counts.push_back(1000); iteration_counts.push_back(500000);
        gpus.clear();
        if(FLAGS_server){
            gpus.push_back(-1); gpus.push_back(0); gpus.push_back(-1); gpus.push_back(0); // Main, GPU, Main, GPU
        } else {                                                                          //  ^     ^     ^    ^
            gpus.push_back(-1); gpus.push_back(-1); gpus.push_back(0); gpus.push_back(0); // Main, Main, GPU, GPU
        }
    }

    std::string csvFileName = FLAGS_csvfile;
    if(FLAGS_csv && csvFileName.empty()){
        std::ostringstream oss;
        oss << "rdma-performance-tests-" << ((int)time(0)) << ".csv";
        csvFileName = oss.str();
    }

    if(FLAGS_server){
        // NodeIDSequencer (Server)
		if (rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) == rdma::Network::getAddressOfConnection(addresses[0])){
			std::cout << "Starting NodeIDSequencer on " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << rdma::Config::SEQUENCER_PORT << std::endl;
			new rdma::NodeIDSequencer();
		}
    }

    #ifndef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		gpus.clear(); gpus.push_back(-1);
	#endif

    size_t testIterations = testNames.size() * gpus.size() * thread_counts.size() * iteration_counts.size() * memsizes.size();
    size_t testCounter = 0;

    auto testIt = testNames.begin();
    while(testIt != testNames.end()){
        std::string testName = *testIt;
        if(testName.length() == 0)
            continue;
        std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);
        std::string test_name = testName; // always lower case

        for(int &gpu_index : gpus){
            for(uint64_t &iterations : iteration_counts){
                bool csvAddHeader = true;
                for(int &thread_count : thread_counts){
                    rdma::PerfTest *test = nullptr;
                    
                    if(std::string("atomicsbandwidth").rfind(test_name, 0) == 0){
                        // Atomics Bandwidth Test
                        testName = "Atomics Bandwidth";
                        test = new rdma::AtomicsBandwidthPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, iterations);

                    } else if(std::string("atomicslatency").rfind(test_name, 0) == 0){
                        // Atomics Latency Test
                        testName = "Atomics Latency";
                        test = new rdma::AtomicsLatencyPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, iterations);
                    }

                    if(test != nullptr){
                        testCounter++;
                        runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                        csvAddHeader = false;
                        continue;
                    }

                    csvAddHeader;
                    for(uint64_t &memsize : memsizes){
                        if(std::string("bandwidth").rfind(test_name, 0) == 0){
                            // Bandwidth Test
                            testName = "Bandwidth";
                            test = new rdma::BandwidthPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, memsize, iterations);

                        } else if(std::string("latency").rfind(test_name, 0) == 0){
                            // Latency Test
                            testName = "Latency";
                            test = new rdma::LatencyPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, memsize, iterations);
                        }

                        if(test == nullptr){
                            std::cerr << "No test with name '" << testName << "' found" << std::endl;
                            testIt = testNames.erase(testIt);
                            continue;
                        }

                        testCounter++;
                        runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                        csvAddHeader = false;
                    }
                }
            }
        }
        ++testIt;
    }
    return 0;
}