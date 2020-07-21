#include "PerfTest.h"
#include "BandwidthPerfTest.h"
#include "AtomicsBandwidthPerfTest.h"
#include "LatencyPerfTest.h"
#include "AtomicsLatencyPerfTest.h"
#include "OperationsCountPerfTest.h"
#include "AtomicsOperationsCountPerfTest.h"

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

DEFINE_bool(fulltest, false, "Overwrites flags 'test, gpu, packetsize, threads, iterations, csv' to execute a broad variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_string(test, "bandwidth", "Test: bandwidth, latency, operationscount, atomicsbandwidth, atomicslatency, atomicsoperationscount (multiples separated by comma without space, not full word required)");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_string(gpu, "-1", "Index of GPU for memory allocation (negative for main memory | multiples separated by comma without space)");
DEFINE_string(packetsize, "4096", "Packet size in bytes (multiples separated by comma without space)");
DEFINE_string(bufferslots, "16", "How many packets the buffer can hold (round-robin distribution of packets inside buffer | multiples separated by comma without space)");
DEFINE_string(threads, "1", "How many individual clients connect to the server. Server has to run same number of threads (multiples separated by comma without space)");
DEFINE_string(iterations, "500000", "Amount of test repeats (multiples separated by comma without space)");
DEFINE_string(addr, "172.18.94.20", "Addresses of NodeIDSequencer to connect/bind to");
DEFINE_int32(port, rdma::Config::RDMA_PORT, "RDMA port");
DEFINE_bool(csv, false, "Results will be written into an automatically generated CSV file");
DEFINE_string(csvfile, "", "Results will be written into a given CSV file");
DEFINE_bool(ignoreerrors, false, "If an error occurs test will be skiped and execution continues");

enum TEST { BANDWIDTH_TEST, LATENCY_TEST, OPERATIONS_COUNT_TEST, ATOMICS_BANDWIDTH_TEST, ATOMICS_LATENCY_TEST, ATOMICS_OPERATIONS_COUNT_TEST };
const uint64_t MINIMUM_PACKET_SIZE = 256;

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
    bool error;
    std::string errorstr = "";
    try {
        std::cout << std::endl << "TEST " << testNumber << " / " << testIterations << " (" << (testNumber*100/testIterations) << "%)" << std::endl;
        std::cout << "SETTING UP ENVIRONMENT FOR TEST '" << testName << "' ..." << std::endl;
        test->setupTest();

        std::cout << "RUN TEST WITH PARAMETERS:  " << test->getTestParameters() << std::endl;
        auto start = rdma::PerfTest::startTimer();
        test->runTest();
        int64_t duration = rdma::PerfTest::stopTimer(start);
        std::cout << "RESULTS: " << test->getTestResults(csvFileName, csvAddHeader) << std::endl;
        std::cout << "DONE TESTING '" << testName << "' (" << rdma::PerfTest::convertTime(duration) << ")" << std::endl << std::endl;
    } catch (const std::exception &ex){
        error = true;
        errorstr = ex.what();
        if(!FLAGS_ignoreerrors)
            throw ex;
    } catch (const std::string &ex){
        error = true;
        errorstr = ex;
        if(!FLAGS_ignoreerrors)
            throw ex;
    } catch (...){
        error = true;
        errorstr = "? ? ?";
        if(!FLAGS_ignoreerrors)
            throw runtime_error("Error occurred while executing test");
    }
    if(error)
        std::cerr << "ERROR '" << errorstr << "' OCCURRED WHILE EXECUTING TEST '" << testName << "' --> JUMP TO NEXT TEST" << std::endl;
    
    delete test;

    if(!FLAGS_server)
        usleep(250000); // if client then sleep for 250ms to wait for server to restart
}



int main(int argc, char *argv[]){
    std::cout << "Parsing arguments ..." << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::vector<std::string> testNames = rdma::StringHelper::split(FLAGS_test);
    std::vector<int> gpus = parseIntList(FLAGS_gpu);
    std::vector<uint64_t> packetsizes = parseUInt64List(FLAGS_packetsize);
    std::vector<int> bufferslots = parseIntList(FLAGS_bufferslots);
    std::vector<int> thread_counts = parseIntList(FLAGS_threads);
    std::vector<uint64_t> iteration_counts = parseUInt64List(FLAGS_iterations);
    std::vector<std::string> addresses = rdma::StringHelper::split(FLAGS_addr);
	for (auto &addr : addresses){
		addr += ":" + to_string(FLAGS_port);
	}
    
    if(FLAGS_fulltest){
        FLAGS_csv = true;
        testNames.clear(); 
        testNames.push_back("bandwidth");
        testNames.push_back("latency");
        testNames.push_back("operationscount");
        testNames.push_back("atomicsbandwidth"); 
        testNames.push_back("atomicslatency"); 
        testNames.push_back("atomicsoperationscount");

        packetsizes.clear();
        // TODO for some reason GPUDirect not working for GPU memory smaller than 128 bytes
        // packetsizes.push_back(64); packetsizes.push_back(128);
        packetsizes.push_back(256); packetsizes.push_back(512); packetsizes.push_back(1024);
        packetsizes.push_back(2048); packetsizes.push_back(4096); packetsizes.push_back(8192);
        packetsizes.push_back(16384); packetsizes.push_back(32768); packetsizes.push_back(131072);
        packetsizes.push_back(262144);

        thread_counts.clear(); 
        thread_counts.push_back(1); thread_counts.push_back(2); 
        thread_counts.push_back(4); thread_counts.push_back(8);

        iteration_counts.clear(); 
        iteration_counts.push_back(1000); iteration_counts.push_back(500000);
        
        gpus.clear();
        if(FLAGS_server){
            gpus.push_back(-1); gpus.push_back(0); gpus.push_back(-1); gpus.push_back(0); // Main, GPU, Main, GPU
        } else {                                                                          //  ^     ^     ^    ^
            gpus.push_back(-1); gpus.push_back(-1); gpus.push_back(0); gpus.push_back(0); // Main, Main, GPU, GPU
        }
    }

    // check thread counts
    for(int &tc : thread_counts){
        if(tc < 1) throw runtime_error("Thread count cannot be smaller than 1");
        if(tc > (int)rdma::Config::RDMA_MAX_WR){
            std::cerr << "Cannot handle " << tc << " threads because Config::RDMA_MAX_WR=" << rdma::Config::RDMA_MAX_WR << " which is also maximum thread number for this tests" << std::endl;
            throw runtime_error("Cannot handle so many threads");
        }
    }

    // check packet sizes
    for(uint64_t &ps : packetsizes){
        if(ps < MINIMUM_PACKET_SIZE){
            std::cerr << "Given packet size " << ps << " must be at least " << MINIMUM_PACKET_SIZE << " bytes for GPUDirect to work" << std::endl;
            throw runtime_error("Packet size too small");
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

    // parse test nams
    size_t testIterations = 0, testCounter = 0;
    std::vector<TEST> tests;
    auto testIt = testNames.begin();
    while(testIt != testNames.end()){
        std::string testName = *testIt;
        if(testName.length() == 0)
            continue;
        std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);

        size_t count = gpus.size() * thread_counts.size() * iteration_counts.size() * bufferslots.size();

        if(std::string("bandwidth").rfind(testName, 0) == 0){
            tests.push_back(BANDWIDTH_TEST);
            count *= packetsizes.size();
        } else if(std::string("latency").rfind(testName, 0) == 0){
            tests.push_back(LATENCY_TEST);
            count *= packetsizes.size();
        } else if(std::string("operationscount").rfind(testName, 0) == 0 || std::string("operationcount").rfind(testName, 0) == 0 || 
                    std::string("ops").rfind(testName, 0) == 0){
            tests.push_back(OPERATIONS_COUNT_TEST);
            count *= packetsizes.size();
        } else if(std::string("atomicsbandwidth").rfind(testName, 0) == 0 || std::string("atomicbandwidth").rfind(testName, 0) == 0){
            tests.push_back(ATOMICS_BANDWIDTH_TEST);
        } else if(std::string("atomicslatency").rfind(testName, 0) == 0 || std::string("atomiclatency").rfind(testName, 0) == 0){
            tests.push_back(ATOMICS_LATENCY_TEST);
        } else if(std::string("atomicsoperationscount").rfind(testName, 0) == 0 || std::string("atomicoperationscount").rfind(testName, 0) == 0 || 
                    std::string("atomicsoperationcount").rfind(testName, 0) == 0 || std::string("atomicoperationcount").rfind(testName, 0) == 0 ||
                    std::string("atomicsops").rfind(testName, 0) == 0 || std::string("atomicops").rfind(testName, 0) == 0){
            tests.push_back(ATOMICS_OPERATIONS_COUNT_TEST);
        } else {
            std::cerr << "No test with name '" << *testIt << "' found" << std::endl;
            testIt++;
            continue;
        }
        testIterations += count;
        testIt++;
    }

     auto totalStart = rdma::PerfTest::startTimer();
    for(TEST &t : tests){
        for(int &gpu_index : gpus){
            for(int &thread_count : thread_counts){
                for(int &buffer_slots : bufferslots){
                    bool csvAddHeader = true;
                    for(uint64_t &iterations : iteration_counts){
                        rdma::PerfTest *test = nullptr;
                        std::string testName;
                        
                        if(t == ATOMICS_BANDWIDTH_TEST){
                            // Atomics Bandwidth Test
                            testName = "Atomics Bandwidth";
                            test = new rdma::AtomicsBandwidthPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, buffer_slots, iterations);

                        } else if(t == ATOMICS_LATENCY_TEST){
                            // Atomics Latency Test
                            testName = "Atomics Latency";
                            test = new rdma::AtomicsLatencyPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, buffer_slots, iterations);

                        } else if(t == ATOMICS_OPERATIONS_COUNT_TEST){
                            // Atomics Operations Count Test
                            testName = "Atomics Operations Count";
                            test = new rdma::AtomicsOperationsCountPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, buffer_slots, iterations);
                        }

                        if(test != nullptr){
                            testCounter++;
                            runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                            csvAddHeader = false;
                            continue;
                        }

                        csvAddHeader = true;
                        for(uint64_t &packet_size : packetsizes){
                            if(t == BANDWIDTH_TEST){
                                // Bandwidth Test
                                testName = "Bandwidth";
                                test = new rdma::BandwidthPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, packet_size, buffer_slots, iterations);

                            } else if(t == LATENCY_TEST){
                                // Latency Test
                                testName = "Latency";
                                test = new rdma::LatencyPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, packet_size, buffer_slots, iterations);

                            } else if(t == OPERATIONS_COUNT_TEST){
                                // Operations Count Test
                                testName = "Operations Count";
                                test = new rdma::OperationsCountPerfTest(FLAGS_server, addresses, FLAGS_port, gpu_index, thread_count, packet_size, buffer_slots, iterations);
                            }

                            testCounter++;
                            runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                            csvAddHeader = false;
                        }
                    }
                }
            }
        }
        ++testIt;
    }

    int64_t totalDuration = rdma::PerfTest::stopTimer(totalStart);
    std::cout << std::endl << "TOTAL EXECUTION TIME " << rdma::PerfTest::convertTime(totalDuration) << std::endl;
    return 0;
}