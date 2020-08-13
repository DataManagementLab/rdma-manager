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
#include <chrono>
#include <bits/stdc++.h>

#include <gflags/gflags.h>

DEFINE_bool(fulltest, false, "Overwrites flags 'test, gpu, remote_gpu, packetsize, threads, iterations, csv' to execute a broad variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_bool(halftest, false, "Overwrites flags 'test, gpu, remote_gpu, packetsize, threads, iterations, csv' to execute a smaller variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_bool(quicktest, false, "Overwrites flags 'test, gpu, remote_gpu, packetsize, threads, iterations, csv' to execute a very smaller variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_string(test, "bandwidth", "Test: bandwidth, latency, operationscount, atomicsbandwidth, atomicslatency, atomicsoperationscount (multiples separated by comma without space, not full word required)");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_string(gpu, "-3", "Index of GPU for memory allocation (-3=Main memory, -2=NUMA aware GPU, -1=Default GPU, 0..n=fixed GPU | multiples separated by comma without space)");
DEFINE_string(remote_gpu, "", "Just for prettier result printing and therefore not required. Same as gpu flag but for remote side (should be empty or same length as gpu flag)");
DEFINE_string(packetsize, "4096", "Packet size in bytes (multiples separated by comma without space)");
DEFINE_string(bufferslots, "16", "How many packets the buffer can hold (round-robin distribution of packets inside buffer | multiples separated by comma without space)");
DEFINE_string(threads, "1", "How many individual clients connect to the server. Server has to run same number of threads (multiples separated by comma without space)");
DEFINE_string(iterations, "500000", "Amount of test repeats (multiples separated by comma without space)");
DEFINE_bool(csv, false, "Results will be written into an automatically generated CSV file");
DEFINE_string(csvfile, "", "Results will be written into a given CSV file");
DEFINE_string(seqaddr, "", "Address of NodeIDSequencer to connect/bind to. If empty then config value will be used");
DEFINE_int32(seqport, -1, "Port of NodeIDSequencer to connect/bind to. If empty then config value will be used");
DEFINE_string(ownaddr, "", "Address of own RDMA interface. If empty then config value 'RDMA_INTERFACE' will be used");
DEFINE_string(addr, "", "RDMA address of RDMAServer to connect/bind to. If empty then config value 'RDMA_INTERFACE' will be used");
DEFINE_int32(port, -1, "RDMA port. If negative then config value will be used");
DEFINE_string(writemode, "auto", "Which RDMA write mode should be used. Possible values are 'immediate' where remote receives and completion entry after a write, 'normal' where remote possibly has to pull the memory constantly to detect changes, 'auto' which uses preferred (ignored by atomics tests | multiples separated by comma without space)");
DEFINE_bool(ignoreerrors, false, "If an error occurs test will be skiped and execution continues");
DEFINE_string(config, "./bin/conf/RDMA.conf", "Path to the config file");

enum TEST { BANDWIDTH_TEST, LATENCY_TEST, OPERATIONS_COUNT_TEST, ATOMICS_BANDWIDTH_TEST, ATOMICS_LATENCY_TEST, ATOMICS_OPERATIONS_COUNT_TEST };
const uint64_t MINIMUM_PACKET_SIZE = 1; // only GPUDirect doesn't work with smaller sizes

static std::vector<int> parseIntList(std::string str){
    std::vector<int> v;
    if(str.length()==0) return v;
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
    if(str.length()==0) return v;
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
        char timestr[255];
        std::time_t systime = std::time(NULL);
        std::strftime(timestr, 255, "%F %T", std::localtime(&systime));
        std::cout << "SETTING UP ENVIRONMENT FOR TEST '" << testName << "' (" << timestr << ") ..." << std::endl;
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
    std::cout << "Arguments parsed" << std::endl << "Loading config ..." << std::endl;
    rdma::Config *config = new rdma::Config(FLAGS_config, false);
    if(FLAGS_seqaddr.empty()) FLAGS_seqaddr=rdma::Config::SEQUENCER_IP;
    if(FLAGS_seqport<=0) FLAGS_seqport=rdma::Config::SEQUENCER_PORT;
    if(FLAGS_ownaddr.empty()) FLAGS_ownaddr=(rdma::Network::isValidIP(FLAGS_ownaddr) ? rdma::Config::RDMA_INTERFACE : rdma::Config::getIP(FLAGS_ownaddr));
    if(FLAGS_addr.empty()) FLAGS_addr=rdma::Config::RDMA_INTERFACE;
    if(FLAGS_port<=0) FLAGS_port=rdma::Config::RDMA_PORT;
    std::cout << "Config loaded" << std::endl;


    std::vector<std::string> testNames = rdma::StringHelper::split(FLAGS_test);
    std::vector<int> local_gpus = parseIntList(FLAGS_gpu);
    std::vector<int> remote_gpus = parseIntList(FLAGS_remote_gpu);
    std::vector<uint64_t> packetsizes = parseUInt64List(FLAGS_packetsize);
    std::vector<int> bufferslots = parseIntList(FLAGS_bufferslots);
    std::vector<int> thread_counts = parseIntList(FLAGS_threads);
    std::vector<uint64_t> iteration_counts = parseUInt64List(FLAGS_iterations);
    std::vector<std::string> writeModeNames = rdma::StringHelper::split(FLAGS_writemode);
    std::vector<std::string> addresses = rdma::StringHelper::split(FLAGS_addr);
	for (auto &addr : addresses){
        if(!rdma::Network::isValidIP(addr)) addr=rdma::Config::getIP(addr);
		addr += ":" + to_string(FLAGS_port);
	}
    
    if(FLAGS_fulltest || FLAGS_halftest || FLAGS_quicktest){
        FLAGS_csv = true;
        testNames.clear(); 
        testNames.push_back("bandwidth");
        testNames.push_back("latency");
        testNames.push_back("operationscount");
        testNames.push_back("atomicsbandwidth"); 
        testNames.push_back("atomicslatency"); 
        testNames.push_back("atomicsoperationscount");

        packetsizes.clear();
        thread_counts.clear();
        iteration_counts.clear(); 
        
        
        local_gpus.clear(); remote_gpus.clear();
        if(!FLAGS_server){
            local_gpus.push_back(-3); local_gpus.push_back(-3); local_gpus.push_back(-2); local_gpus.push_back(-2); //     Main, Main, GPU, GPU
            remote_gpus.push_back(-3); remote_gpus.push_back(-2); remote_gpus.push_back(-3); remote_gpus.push_back(-2); // Main, GPU, Main, GPU
        } else {
            local_gpus.push_back(-3); local_gpus.push_back(-2); local_gpus.push_back(-3); local_gpus.push_back(-2); //     Main, GPU, Main, GPU
            remote_gpus.push_back(-3); remote_gpus.push_back(-3); remote_gpus.push_back(-2); remote_gpus.push_back(-2); // Main, Main, GPU, GPU
        }
    }
    if(FLAGS_fulltest){
        // TODO for some reason GPUDirect not working for GPU memory smaller than 128 bytes
        // packetsizes.push_back(64); packetsizes.push_back(128);
        packetsizes.push_back(256); packetsizes.push_back(512); packetsizes.push_back(1024);
        packetsizes.push_back(2048); packetsizes.push_back(4096); packetsizes.push_back(8192);
        packetsizes.push_back(16384); packetsizes.push_back(32768); packetsizes.push_back(65536);
        packetsizes.push_back(131072); packetsizes.push_back(262144); packetsizes.push_back(524288);
        packetsizes.push_back(1048576); // > 1MB

        thread_counts.push_back(1); thread_counts.push_back(2); 
        thread_counts.push_back(4); thread_counts.push_back(8);
        thread_counts.push_back(16);

        iteration_counts.push_back(500); iteration_counts.push_back(500000);

    } else if(FLAGS_halftest){
        // TODO for some reason GPUDirect not working for GPU memory smaller than 128 bytes
        //packetsizes.push_back(64);
        packetsizes.push_back(256); packetsizes.push_back(1024); packetsizes.push_back(4096);
        packetsizes.push_back(16384); packetsizes.push_back(65536);
        packetsizes.push_back(262144); packetsizes.push_back(1048576); // > 1MB

        thread_counts.push_back(1); thread_counts.push_back(4); thread_counts.push_back(16);

        iteration_counts.push_back(500); iteration_counts.push_back(500000);

    } else if(FLAGS_quicktest){
        //packetsizes.push_back(64);
        packetsizes.push_back(256);
        packetsizes.push_back(1024);
        packetsizes.push_back(4096);
        packetsizes.push_back(16384);
        packetsizes.push_back(65536);

        thread_counts.push_back(1); thread_counts.push_back(4);

        iteration_counts.push_back(500); iteration_counts.push_back(50000);
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

    // check CSV file
    std::string csvFileName = FLAGS_csvfile;
    if(FLAGS_csv && csvFileName.empty()){
        std::ostringstream oss;
        oss << "rdma-performance-tests-";
        if(FLAGS_fulltest){
            oss << "fulltest-";
        } else if(FLAGS_halftest){
            oss << "halftest-";
        } else if(FLAGS_quicktest){
            oss << "quicktest-";
        }
        oss << ((int)time(0)) << ".csv";
        csvFileName = oss.str();
    }

    // check CUDA support
    #ifndef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		local_gpus.clear(); local_gpus.push_back(-3);
	#endif

    if(remote_gpus.size() == 0) remote_gpus.push_back(-404);

    // Parse write mode names
    std::vector<rdma::WriteMode> write_modes;
    auto writeModeIt = writeModeNames.begin();
    while(writeModeIt != writeModeNames.end()){
        std::string writeModeName = *writeModeIt;
        if(writeModeName.length() == 0)
            continue;
        std::transform(writeModeName.begin(), writeModeName.end(), writeModeName.begin(), ::tolower);
        if(std::string("automatic").rfind(writeModeName, 0) == 0 || std::string("preferred").rfind(writeModeName, 0) == 0 || std::string("default").rfind(writeModeName, 0) == 0){
            write_modes.push_back(rdma::WRITE_MODE_AUTO);
        } else if(std::string("normal").rfind(writeModeName, 0) == 0){
            write_modes.push_back(rdma::WRITE_MODE_NORMAL);
        } else  if(std::string("immediate").rfind(writeModeName, 0) == 0){
            write_modes.push_back(rdma::WRITE_MODE_IMMEDIATE);
        } else {
            std::cerr << "No write mode with name '" << *writeModeIt << "' found" << std::endl;
        }
        writeModeIt++;
    }

    // Parse test names
    size_t testIterations = 0, testCounter = 0;
    std::vector<TEST> tests;
    auto testIt = testNames.begin();
    while(testIt != testNames.end()){
        std::string testName = *testIt;
        if(testName.length() == 0)
            continue;
        std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);

        size_t count = local_gpus.size() * thread_counts.size() * iteration_counts.size() * bufferslots.size();

        if(std::string("bandwidth").rfind(testName, 0) == 0){
            tests.push_back(BANDWIDTH_TEST);
            count *= packetsizes.size() * write_modes.size();
        } else if(std::string("latency").rfind(testName, 0) == 0){
            tests.push_back(LATENCY_TEST);
            count *= packetsizes.size() * write_modes.size();
        } else if(std::string("operationscount").rfind(testName, 0) == 0 || std::string("operationcount").rfind(testName, 0) == 0 || 
                    std::string("ops").rfind(testName, 0) == 0){
            tests.push_back(OPERATIONS_COUNT_TEST);
            count *= packetsizes.size() * write_modes.size();
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


    // start NodeIDSequencer
    if(FLAGS_server){
        if(FLAGS_seqaddr=="*" || FLAGS_seqaddr=="localhost" || FLAGS_seqaddr=="127.0.0.1" || FLAGS_seqaddr == rdma::Network::getOwnAddress()){
            std::cout << "Starting NodeIDSequencer on port " << FLAGS_seqport << std::endl;
            new rdma::NodeIDSequencer(FLAGS_seqport, "*");
        }
    }
    std::string sequencerIpAddr = FLAGS_seqaddr+":"+to_string(FLAGS_seqport);


    // EXECUTE TESTS
    auto totalStart = rdma::PerfTest::startTimer();
    for(TEST &t : tests){
        for(size_t gpui = 0; gpui < local_gpus.size(); gpui++){
            const int local_gpu_index = local_gpus[gpui];
            const int remote_gpu_index = remote_gpus[gpui % sizeof(remote_gpu_index)];
            for(int &thread_count : thread_counts){
                for(int &buffer_slots : bufferslots){
                    bool csvAddHeader = true;
                    for(uint64_t &iterations : iteration_counts){
                        rdma::PerfTest *test = nullptr;
                        std::string testName;

                        uint64_t iterations_per_thread = iterations / thread_count;
                        if(iterations_per_thread==0) iterations_per_thread = 1;
                        
                        if(t == ATOMICS_BANDWIDTH_TEST){
                            // Atomics Bandwidth Test
                            testName = "Atomics Bandwidth";
                            test = new rdma::AtomicsBandwidthPerfTest(FLAGS_server, addresses, FLAGS_port, local_gpu_index, remote_gpu_index, thread_count, buffer_slots, iterations_per_thread);

                        } else if(t == ATOMICS_LATENCY_TEST){
                            // Atomics Latency Test
                            testName = "Atomics Latency";
                            test = new rdma::AtomicsLatencyPerfTest(FLAGS_server, addresses, FLAGS_port, local_gpu_index, remote_gpu_index, thread_count, buffer_slots, iterations_per_thread);

                        } else if(t == ATOMICS_OPERATIONS_COUNT_TEST){
                            // Atomics Operations Count Test
                            testName = "Atomics Operations Count";
                            test = new rdma::AtomicsOperationsCountPerfTest(FLAGS_server, addresses, FLAGS_port, local_gpu_index, remote_gpu_index, thread_count, buffer_slots, iterations_per_thread);
                        }

                        if(test != nullptr){
                            testCounter++;
                            runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                            csvAddHeader = false;
                            continue;
                        }

                        for(rdma::WriteMode &write_mode : write_modes){
                            csvAddHeader = true;
                            for(uint64_t &packet_size : packetsizes){
                                if(t == BANDWIDTH_TEST){
                                    // Bandwidth Test
                                    testName = "Bandwidth";
                                    test = new rdma::BandwidthPerfTest(FLAGS_server, addresses, FLAGS_port, local_gpu_index, remote_gpu_index, thread_count, packet_size, buffer_slots, iterations_per_thread, write_mode);

                                } else if(t == LATENCY_TEST){
                                    // Latency Test
                                    testName = "Latency";
                                    test = new rdma::LatencyPerfTest(FLAGS_server, addresses, FLAGS_port, local_gpu_index, remote_gpu_index, thread_count, packet_size, buffer_slots, iterations_per_thread, write_mode);

                                } else if(t == OPERATIONS_COUNT_TEST){
                                    // Operations Count Test
                                    testName = "Operations Count";
                                    test = new rdma::OperationsCountPerfTest(FLAGS_server, addresses, FLAGS_port, local_gpu_index, remote_gpu_index, thread_count, packet_size, buffer_slots, iterations_per_thread, write_mode);
                                }

                                testCounter++;
                                runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                                csvAddHeader = false;
                            }
                        }
                    }
                }
            }
        }
        ++testIt;
    }

    int64_t totalDuration = rdma::PerfTest::stopTimer(totalStart);
    std::cout << std::endl << "TOTAL EXECUTION TIME " << rdma::PerfTest::convertTime(totalDuration) << std::endl;
    delete config;
    return 0;
}