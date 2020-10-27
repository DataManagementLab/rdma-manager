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
#include "../src/rdma/RDMAServer.h"
#include "../src/rdma/RDMAClient.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <ctime>
#include <chrono>
#include <bits/stdc++.h>

#include <gflags/gflags.h>

DEFINE_bool(fulltest, false, "Sets default values for flags 'test, gpu, remote_gpu, packetsize, threads, iterations, bufferslots, csv' to execute a broad variety of predefined tests. Flags can still be overwritten. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_bool(halftest, false, "Sets default values for flags 'test, gpu, remote_gpu, packetsize, threads, iterations, bufferslots, csv' to execute a smaller variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_bool(quicktest, false, "Sets default values for flags 'test, gpu, remote_gpu, packetsize, threads, iterations, csv' to execute a very smaller variety of predefined tests. If GPUs are supported then gpu=-1,-1,0,0 on client side and gpu=-1,0,-1,0 on server side to test all memory combinations: Main->Main, Main->GPU, GPU->Main, GPU->GPU");
DEFINE_string(test, "", "Tests: [bandwidth, latency, operationscount, atomicsbandwidth, atomicslatency, atomicsoperationscount] OR MORE GRANULAR [write_bw, write_lat, write_ops, read_bw, read_lat, read_ops, send_bw, send_lat, send_ops, fetch_bw, fetch_lat, fetch_ops, swap_bw, swap_lat, swap_ops] (multiples separated by comma without space, not full word required) [Default bandwidth]");
DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_int32(clients, 1, "Required by all servers as well as all clients to know how many actual client processes are running. It is irelevant how many threads actually used just how often an instance of the performance tool got started in client mode.");
DEFINE_string(memtype, "", "Memory type or index of GPU for memory allocation ('-3' or 'MAIN' for Main memory, '-2' or 'GPU.NUMA' for NUMA aware GPU, '-1' or 'GPU.D' for default GPU, '0..n' or 'GPU.i' i index for fixed GPU | multiples separated by comma without space) [Default -3]");
DEFINE_string(remote_memtype, "", "Just for prettier result printing and therefore not essential. Same as  --memtype  flag but for remote side (should be empty or same length as  --memtype  flag)");
DEFINE_string(packetsize, "", "Packet size in bytes or 'all' for all powers of two, 'small' for a bunch of small packet sizes, 'big' for a bunch of big packet sizes, 'mid' for a bunch between small and big (multiple numbers separated by comma without space) [Default 4096B, Min 4B]");
DEFINE_string(bufferslots, "", "How many packets the buffer can hold (round-robin distribution of packets inside buffer | multiples separated by comma without space) [Default 16]");
DEFINE_string(threads, "", "Each thread starts its own connection to the server. Server and other clients need exactly the same value (multiples separated by comma without space) [Default 1]");
DEFINE_string(iterations, "", "Amount of transfers for latency and all atomics tests (multiples separated by comma without space) [Default 500000]");
DEFINE_string(maxtransfersize, "24GB", "Limits the iterations base on the maximal transfer size just for the latency tests. Set to zero or negative to disable this flag. Doesn't affect  --transfersize  or  --maxiterations  flag or atomics tests");
DEFINE_string(transfersize, "", "How much data should be transfered in bandwidth and operations/sec tests but not for atomics tests (multiples separated by comma without space) [Default 24GB]");
DEFINE_int32(maxiterations, 500000, "Amount of iterations for bandwidth and operations/sec are calculated via transfersize/packetsize and this flag sets a maximum limit to speed up tests for very small packetsizes. Set to zero or negative value to ignore this flag. Doesn't affect the  --iterations  or  --maxtransfersize  flag");
DEFINE_bool(csv, false, "Results will be written into an automatically generated CSV file");
DEFINE_string(csvfile, "", "Results will be written into a given CSV file");
DEFINE_string(seqaddr, "", "Address of NodeIDSequencer to connect/bind to. If empty then config value will be used");
DEFINE_int32(seqport, -1, "Port of NodeIDSequencer to connect/bind to. If empty then config value will be used");
DEFINE_string(ownaddr, "", "Address of own RDMA interface that the RDMAServer will use to bind to and the RDMAClient the retriev its node id. If empty then config value 'RDMA_INTERFACE' will be used");
DEFINE_string(addr, "", "RDMA address for the RDMACLient to connect to. If empty then config value 'RDMA_SERVER_ADDRESSES' will be used. (multiples separated by comma without space will open a connection to each address in parallel)");
DEFINE_int32(port, -1, "RDMA port. If negative then config value will be used");
DEFINE_string(writemode, "auto", "Which RDMA write mode should be used. Possible values are 'immediate' where remote receives and completion entry after a write, 'normal' where remote possibly has to pull the memory constantly to detect changes, 'auto' which uses preferred (ignored by atomics tests | multiples separated by comma without space)");
DEFINE_bool(ignoreerrors, false, "If an error occurs test will be skiped and execution continues");
DEFINE_string(config, "./bin/conf/RDMA.conf", "Path to the config file");

enum TEST { BANDWIDTH_TEST=1, LATENCY_TEST=2, OPERATIONS_COUNT_TEST=3, ATOMICS_BANDWIDTH_TEST=4, ATOMICS_LATENCY_TEST=5, ATOMICS_OPERATIONS_COUNT_TEST=6 };
extern const uint64_t MINIMUM_PACKET_SIZE = 4; // >=4 for latency to transfer remote offset


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
            v.push_back((uint64_t)std::strtoull(str.c_str(), nullptr, 10)); // base 10
        } catch (std::exception const &e){
            std::cerr << "Could not parse integer from '" << str << "'" << std::endl;
        }
    } return v;
}
static std::vector<uint64_t> parseByteSizesList(std::string str){
    std::vector<uint64_t> v;
    if(str.empty()) return v;
    std::stringstream ss(str);
    while((std::getline(ss, str, ','))){
        if(str.length() == 0)
            continue;
        try {
            v.push_back((uint64_t)rdma::StringHelper::parseByteSize(str));
        } catch (std::exception const &e){
            std::cerr << "Could not parse integer from '" << str << "'" << std::endl;
        }
    } return v;
}
static std::vector<int> parseMemoryTypeList(std::string str){
    std::vector<int> v;
    if(str.empty()) return v;
    std::stringstream ss(str);
    while((std::getline(ss, str, ','))){
        if(str.length() == 0)
            continue;
        try {
            v.push_back(std::stoi(str));
        } catch (std::exception const &e){
            std::transform(str.begin(), str.end(), str.begin(), ::toupper);
            if(std::string("MAIN").find(str) != std::string::npos){
                v.push_back(-3);
            } else if(str.find("G") != std::string::npos){
                size_t start = str.find(".");
                if(start != std::string::npos){
                    str = str.substr(start+1);
                    try {
                        v.push_back(std::stoi(str));
                    } catch (std::exception const &ex){
                        if(str.find("D") != std::string::npos){
                            v.push_back(-1); // default
                        } else { v.push_back(-2); } // NUMA
                    }
                } else {
                    v.push_back(-2); // NUMA
                }
            } else if(std::string("NUMA").find(str) != std::string::npos){
                v.push_back(-2); // GPU NUMA
            } else {
                std::cerr << "Unknown memory type '" << str << "'. Use flag --help for details" << std::endl;
            }
        }
    } return v;
}


static void runTest(size_t testNumber, size_t testIterations, std::string testName, rdma::PerfTest *test, std::string csvFileName, bool csvAddHeader){
    bool error = false;
    std::string errorname = "", errorstr = "";
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
        try { errorname = std::string(typeid(ex).name()) + ":  "; } catch (...){}
        errorstr = ex.what();
        if(!FLAGS_ignoreerrors)
            throw ex;
    } catch (const std::string &ex){
        error = true;
        errorname = "string: ";
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
        std::cerr << "ERROR " << errorname << "'" << errorstr << "' OCCURRED WHILE EXECUTING TEST '" << testName << "' --> JUMP TO NEXT TEST" << std::endl;
    
    delete test;

    if(!FLAGS_server)
        usleep(250000); // if client then sleep for 250ms to wait for server to restart
}


static void initialSyncAsServer(std::string ownIpPort, std::string sequencerIpPort, size_t expected_clients){
    int port = rdma::Network::getPortOfConnection(ownIpPort);
    std::string addr = rdma::Network::getAddressOfConnection(ownIpPort);
    uint64_t mem_size = 1; //if(mem_size < rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE) mem_size =  rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE;
    rdma::RDMAServer<rdma::ReliableRDMA> *server = new rdma::RDMAServer<rdma::ReliableRDMA>(std::string("IntialSyncServer"), port, addr, mem_size, sequencerIpPort);
    server->startServer();
    rdma::PerfTest::global_barrier_server(server, expected_clients);
    while(!server->getConnectedConnIDs().empty()){
        usleep(rdma::Config::RDMA_SLEEP_INTERVAL); // wait for all clients to disconnect
    }
    delete server;
}

static void initialSyncAsClient(const std::vector<std::string> &serverIpAndPorts, std::string ownIpPort, std::string sequencerIpPort){
    uint64_t mem_size = 1; //if(mem_size < rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE) mem_size =  rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE;
    rdma::RDMAClient<rdma::ReliableRDMA> *client = new rdma::RDMAClient<rdma::ReliableRDMA>(mem_size, "InitialSyncClient", ownIpPort, sequencerIpPort);
    std::vector<NodeID> nodeIds;
    for(auto &ipPort : serverIpAndPorts){
        NodeID nodeId;
        if(!client->connect(ipPort, nodeId))
            throw runtime_error("Could not connect initial sync client with '" + ipPort + "'");
        nodeIds.push_back(nodeId);
    }
    rdma::PerfTest::global_barrier_client(client, nodeIds);
    delete client;
    usleep(2*rdma::Config::RDMA_SLEEP_INTERVAL + rdma::Config::PERFORMANCE_TEST_SERVER_TIME_ADVANTAGE); // ensure that all clients have disconnected
}


static bool checkInvalidTestParams(size_t packet_size, int local_gpu_index, int remote_gpu_index){
    // skip if GPU and packet size < Config::GPUDIRECT_MINIMUM_MSG_SIZE  (same if condition lower)
    if((FLAGS_server ? remote_gpu_index : local_gpu_index) > (int)rdma::MEMORY_TYPE::MAIN && packet_size < rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE){
        std::cout << "SKIPPING TEST BECAUSE GPU " << (FLAGS_server ? remote_gpu_index : local_gpu_index) << " (>" << (int)rdma::MEMORY_TYPE::MAIN;
        std::cout << ") AND PACKET SIZE " << packet_size << " (<" << rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE << ")" << std::endl;
        return true;
    } return false;
}


int main(int argc, char *argv[]){
    std::cout << std::endl << "INFO:  All bandwidth measurements are correctly measured in MiB/s to be compareable to other tools and network stats but labeled as MB/s as this is commonly practiced!" << std::endl << std::endl;
    std::cout << "Parsing arguments ..." << std::endl;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "Arguments parsed" << std::endl << "Loading config ..." << std::endl;
    rdma::Config *config = new rdma::Config(FLAGS_config, false);
    if(FLAGS_seqaddr.empty()) FLAGS_seqaddr=rdma::Config::SEQUENCER_IP;
    if(FLAGS_seqport<=0) FLAGS_seqport=rdma::Config::SEQUENCER_PORT;
    if(FLAGS_ownaddr.empty()) FLAGS_ownaddr=rdma::Config::RDMA_INTERFACE;
        if(auto find = FLAGS_ownaddr.find(":")) if(find != std::string::npos) FLAGS_ownaddr=FLAGS_ownaddr.substr(0, find);
        if(!rdma::Network::isValidIP(FLAGS_ownaddr)) FLAGS_ownaddr=rdma::Config::getIP(FLAGS_ownaddr);
    if(FLAGS_addr.empty()) FLAGS_addr=rdma::Config::RDMA_SERVER_ADDRESSES;
    if(FLAGS_port<=0) FLAGS_port=rdma::Config::RDMA_PORT;
    std::cout << "Config loaded" << std::endl;

    if(FLAGS_fulltest || FLAGS_halftest || FLAGS_quicktest){
        FLAGS_csv = true;
        if(FLAGS_test.empty()) FLAGS_test = "write_bw,write_lat,write_ops,read_bw,read_lat,read_ops,send_bw,send_lat,send_ops,fetch_bw,fetch_lat,fetch_ops,swap_bw,swap_lat,swap_ops";
        if(FLAGS_memtype.empty()) FLAGS_memtype = (FLAGS_server ? "-3,-2,-3,-2" : "-3,-3,-2,-2");
        if(FLAGS_remote_memtype.empty()) FLAGS_remote_memtype = (FLAGS_server ? "-3,-3,-2,-2" : "-3,-2,-3,-2");
    }
    if(FLAGS_fulltest){
        // TODO for some reason GPUDirect not working for GPU memory smaller than 128 bytes
        if(FLAGS_packetsize.empty()) FLAGS_packetsize = "all";
        if(FLAGS_threads.empty()) FLAGS_threads = "1,2,4,8,16";
        if(FLAGS_iterations.empty()) FLAGS_iterations = "500,500000";
        if(FLAGS_transfersize.empty()) FLAGS_transfersize = "1GB,24GB";
        if(FLAGS_bufferslots.empty()) FLAGS_bufferslots = "1,16";

    } else if(FLAGS_halftest){
        // TODO for some reason GPUDirect not working for GPU memory smaller than 128 bytes
        if(FLAGS_packetsize.empty()) FLAGS_packetsize = "8,64,256,1024,4096,16384,65536,262144,1048576";
        if(FLAGS_threads.empty()) FLAGS_threads = "1,4,16";
        if(FLAGS_bufferslots.empty()) FLAGS_bufferslots = "1,16";

    } else if(FLAGS_quicktest){
        // TODO for some reason GPUDirect not working for GPU memory smaller than 128 bytes
        if(FLAGS_packetsize.empty()) FLAGS_packetsize = "256,1024,4096,16384,65536";
        if(FLAGS_threads.empty()) FLAGS_threads = "1,4";
        if(FLAGS_iterations.empty()) FLAGS_iterations = "50000";
        if(FLAGS_transfersize.empty()) FLAGS_transfersize = "1GB";
    }


    // Loading default values if no value is set
    if(FLAGS_test.empty()) FLAGS_test = "bandwidth";
    if(FLAGS_memtype.empty()) FLAGS_memtype = "-3"; // do not set remote_memtype
    if(FLAGS_packetsize.empty()) FLAGS_packetsize = "4096";
    if(FLAGS_bufferslots.empty()) FLAGS_bufferslots = "16";
    if(FLAGS_threads.empty()) FLAGS_threads = "1";
    if(FLAGS_iterations.empty()) FLAGS_iterations = "500000";
    if(FLAGS_transfersize.empty()) FLAGS_transfersize = "24GB";

    // Checking if default packet sizes are requested
    std::string packetSizeStr = FLAGS_packetsize;
    std::transform(packetSizeStr.begin(), packetSizeStr.end(), packetSizeStr.begin(), ::tolower);
    if(std::string("all").find(packetSizeStr) != std::string::npos || std::string("full").find(packetSizeStr) != std::string::npos){
        FLAGS_packetsize = "4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576";
    } else if(std::string("smalls").find(packetSizeStr) != std::string::npos || std::string("tiny").find(packetSizeStr) != std::string::npos){
        FLAGS_packetsize = "4,8,16,32,64,128,256,512,1024,2048,4096";
    } else if(std::string("middles").find(packetSizeStr) != std::string::npos || std::string("mediums").find(packetSizeStr) != std::string::npos){
        FLAGS_packetsize = "512,1024,2048,4096,8192,16384,32768,65536";
    } else if(std::string("bigs").find(packetSizeStr) != std::string::npos || std::string("huges").find(packetSizeStr) != std::string::npos){
        FLAGS_packetsize = "8192,16384,32768,65536,131072,262144,524288,1048576";
    }


    // Parsing values
    std::vector<std::string> testNames = rdma::StringHelper::split(FLAGS_test);
    std::vector<TEST> tests; // tests that should be executed (order important)
    std::unordered_map<TEST, int> testOperations; // stores which operations should be executed per test
    std::vector<int> local_memtypes = parseMemoryTypeList(FLAGS_memtype);
    std::vector<int> remote_memtypes = parseMemoryTypeList(FLAGS_remote_memtype);
    std::vector<uint64_t> packetsizes = parseByteSizesList(FLAGS_packetsize);
    std::vector<int> bufferslots = parseIntList(FLAGS_bufferslots);
    std::vector<int> thread_counts = parseIntList(FLAGS_threads);
    std::vector<uint64_t> iteration_counts = parseUInt64List(FLAGS_iterations);
    int64_t maxtransfersize = rdma::StringHelper::parseByteSize(FLAGS_maxtransfersize);
    std::vector<uint64_t> transfersizes = parseByteSizesList(FLAGS_transfersize);
    std::vector<std::string> writeModeNames = rdma::StringHelper::split(FLAGS_writemode);
    std::vector<std::string> addresses = rdma::StringHelper::split(FLAGS_addr);
	for (auto &addr : addresses){
        if(!rdma::Network::isValidIP(addr)) addr=rdma::Config::getIP(addr);
		addr += ":" + to_string(FLAGS_port);
	}


    if(FLAGS_server && addresses.size()!= 1){
        throw runtime_error("As server the -addr flag is only allowed to contain just a single value");
    }

    // check thread counts and if server then multiply by amount of clients
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
            std::cerr << "Given packet size " << ps << " must be at least " << MINIMUM_PACKET_SIZE << " bytes" << std::endl;
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
		local_memtypes.clear(); local_memtypes.push_back(-3);
	#endif

    if(remote_memtypes.empty()) remote_memtypes.push_back(-404);

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

    auto testIt = testNames.begin();
    while(testIt != testNames.end()){
        std::string testName = *testIt;
        testIt++;
        if(testName.length() == 0)
            continue;
        std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower); // lowercase

        size_t count = local_memtypes.size() * thread_counts.size() * bufferslots.size();
        TEST test;
        int test_ops = 0;
        bool parse_op = true;

        // Parse test type
        if(std::string("bandwidth").find(testName) == 0 || std::string("bw").find(testName) == 0){
            test = BANDWIDTH_TEST;
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
            if(test_ops == 0){ count *= transfersizes.size() * packetsizes.size() * write_modes.size(); }
            parse_op = false;
        } else if(std::string("latency").find(testName) == 0){
            test = LATENCY_TEST;
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
            if(test_ops == 0){ count *= iteration_counts.size() * packetsizes.size() * write_modes.size(); }
            parse_op = false;
        } else if(std::string("operationscount").find(testName) == 0 || std::string("ops").find(testName) == 0){
            test = OPERATIONS_COUNT_TEST;
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
            if(test_ops == 0){ count *= transfersizes.size() * packetsizes.size() * write_modes.size(); }
            parse_op = false;
        } else if(std::string("atomicbandwidth").find(testName) == 0 || std::string("atomicsbandwidth").find(testName) == 0 || 
                    std::string("atomicbw").find(testName) == 0 || std::string("atomicsbw").find(testName) == 0){
            test = ATOMICS_BANDWIDTH_TEST;
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
            if(test_ops == 0){ count *= iteration_counts.size(); }
            parse_op = false;
        } else if(std::string("atomiclatency").find(testName) == 0 || std::string("atomicslatency").find(testName) == 0){
            test = ATOMICS_LATENCY_TEST;
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
            if(test_ops == 0){ count *= iteration_counts.size(); }
            parse_op = false;
        } else if(std::string("atomicoperationscount").find(testName) == 0 || std::string("atomicops").find(testName) == 0 || 
                    std::string("atomicsoperationscount").find(testName) == 0 || std::string("atomicsops").find(testName) == 0){
            test = ATOMICS_OPERATIONS_COUNT_TEST;
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
            if(test_ops == 0){ count *= iteration_counts.size(); }
            parse_op = false;

        } else { 

            if(testName.rfind("bw") != std::string::npos){
                test = BANDWIDTH_TEST;
            } else if(testName.rfind("lat") != std::string::npos){
                test = LATENCY_TEST;
            } else if(testName.rfind("op") != std::string::npos){
                test = OPERATIONS_COUNT_TEST;
            } else {
                std::cerr << "Unknown test '" << testName << "'" << std::endl;
                continue;
            }
            test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
        }

        // Parse test operations
        if(parse_op){
            if(testName.find("wri") != std::string::npos){
                if(test_ops == 0){ count *= (test!=LATENCY_TEST ? transfersizes.size() : iteration_counts.size()) * packetsizes.size() * write_modes.size(); }
                test_ops = (test_ops | (int)rdma::WRITE_OPERATION);
            } else if(testName.find("rea") != std::string::npos){
                if(test_ops == 0){ count *= (test!=LATENCY_TEST ? transfersizes.size() : iteration_counts.size()) * packetsizes.size() * write_modes.size(); }
                test_ops = (test_ops | (int)rdma::READ_OPERATION);
            } else if(testName.find("sen") != std::string::npos || testName.find("rec") != std::string::npos){
                if(test_ops == 0){ count *= (test!=LATENCY_TEST ? transfersizes.size() : iteration_counts.size()) * packetsizes.size() * write_modes.size(); }
                test_ops = (test_ops | (int)rdma::SEND_RECEIVE_OPERATION);
            } else if(testName.find("fet") != std::string::npos || testName.find("add") != std::string::npos){
                test = (test==BANDWIDTH_TEST?ATOMICS_BANDWIDTH_TEST:(test==LATENCY_TEST?ATOMICS_LATENCY_TEST:ATOMICS_OPERATIONS_COUNT_TEST));
                test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
                if(test_ops == 0){ count *= iteration_counts.size(); }
                test_ops = (test_ops | (int)rdma::FETCH_ADD_OPERATION);
            } else if(testName.find("com") != std::string::npos || testName.find("swa") != std::string::npos){
                test = (test==BANDWIDTH_TEST?ATOMICS_BANDWIDTH_TEST:(test==LATENCY_TEST?ATOMICS_LATENCY_TEST:ATOMICS_OPERATIONS_COUNT_TEST));
                test_ops = (testOperations.find(test) != testOperations.end() ? testOperations[test] : 0);
                if(test_ops == 0){ count *= iteration_counts.size(); }
                test_ops = (test_ops | (int)rdma::COMPARE_SWAP_OPERATION);
            } else {
                std::cerr << "Could not detect RDMA operation from '" << testName << "'" << std::endl;
                continue;
            }
        } else {
            test_ops = (int)rdma::WRITE_OPERATION | (int)rdma::READ_OPERATION | (int)rdma::SEND_RECEIVE_OPERATION |
                        (int)rdma::FETCH_ADD_OPERATION | (int)rdma::COMPARE_SWAP_OPERATION;  // all operations
        }

        if(std::find(tests.begin(), tests.end(), test) == tests.end()){
            tests.push_back(test);
        }
        testOperations[test] = test_ops;

        testIterations += count;
    }


    // start NodeIDSequencer
    if(FLAGS_server){
        if(FLAGS_seqaddr=="*" || FLAGS_seqaddr=="localhost" || FLAGS_seqaddr=="127.0.0.1" || FLAGS_seqaddr == rdma::Network::getOwnAddress()){
            std::cout << "Starting NodeIDSequencer on port " << FLAGS_seqport << std::endl;
            new rdma::NodeIDSequencer(FLAGS_seqport, "*");
        }
    }
    std::string ownIpPort = FLAGS_ownaddr+":"+to_string(FLAGS_port);
    std::string sequencerIpAddr = FLAGS_seqaddr+":"+to_string(FLAGS_seqport);


    // INTIAL SYNC
    if(FLAGS_server){
        std::cout << "Waiting for " << FLAGS_clients << " clients to connect..." << std::endl;
        initialSyncAsServer(ownIpPort, sequencerIpAddr, FLAGS_clients);
        std::cout << "All clients are connected!" << std::endl;
    } else {
        std::cout << "Waiting for all clients to connect" << std::endl;
        initialSyncAsClient(addresses, ownIpPort, sequencerIpAddr);
        std::cout << "All clients are connected!" << std::endl;
    }


    // EXECUTE TESTS
    auto totalStart = rdma::PerfTest::startTimer();
    for(TEST &t : tests){
        int test_ops = testOperations[t];
        for(size_t gpui = 0; gpui < local_memtypes.size(); gpui++){
            const int local_gpu_index = local_memtypes[gpui];
            const int remote_gpu_index = remote_memtypes[gpui % remote_memtypes.size()];

            for(int &thread_count : thread_counts){
                for(int &buffer_slots : bufferslots){
                    bool csvAddHeader = true;
                    
                    for(uint64_t &transfersize : transfersizes){
                        rdma::PerfTest *test = nullptr;
                        std::string testName;

                        for(rdma::WriteMode &write_mode : write_modes){
                            csvAddHeader = true;
                            for(uint64_t &packet_size : packetsizes){
                                test = nullptr;
                                uint64_t iterations_per_thread = (uint64_t)((long double)transfersize / (long double)packet_size + 0.5);
                                if(FLAGS_maxiterations > 0 && iterations_per_thread > (uint64_t)FLAGS_maxiterations) iterations_per_thread = FLAGS_maxiterations;
                                iterations_per_thread = (uint64_t)((long double)iterations_per_thread / (long double)thread_count + 0.5);
                                if(iterations_per_thread==0) iterations_per_thread = 1;

                                if(t == BANDWIDTH_TEST){
                                    if(checkInvalidTestParams(packet_size, local_gpu_index, remote_gpu_index)){
                                        testCounter++; csvAddHeader = true; continue;
                                    }
                                    // Bandwidth Test
                                    testName = "Bandwidth";
                                    test = new rdma::BandwidthPerfTest(test_ops, FLAGS_server, addresses, FLAGS_port, ownIpPort, sequencerIpAddr, local_gpu_index, remote_gpu_index, FLAGS_clients, thread_count, packet_size, buffer_slots, iterations_per_thread, write_mode);

                                } else if(t == OPERATIONS_COUNT_TEST){
                                    if(checkInvalidTestParams(packet_size, local_gpu_index, remote_gpu_index)){
                                        testCounter++; csvAddHeader = true; continue;
                                    }
                                    // Operations Count Test
                                    testName = "Operations Count";
                                    test = new rdma::OperationsCountPerfTest(test_ops, FLAGS_server, addresses, FLAGS_port, ownIpPort, sequencerIpAddr, local_gpu_index, remote_gpu_index, FLAGS_clients, thread_count, packet_size, buffer_slots, iterations_per_thread, write_mode);
                                }

                                if(test != nullptr){
                                    testCounter++;
                                    runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                                    csvAddHeader = false;
                                }
                            }
                        }
                    }

                    csvAddHeader = true;
                    for(uint64_t &iterations : iteration_counts){
                        rdma::PerfTest *test = nullptr;
                        std::string testName;

                        uint64_t iterations_per_thread = iterations / thread_count;
                        if(iterations_per_thread==0) iterations_per_thread = 1;
                        
                        if(t == ATOMICS_BANDWIDTH_TEST){
                            // Atomics Bandwidth Test
                            testName = "Atomics Bandwidth";
                            test = new rdma::AtomicsBandwidthPerfTest(test_ops, FLAGS_server, addresses, FLAGS_port, ownIpPort, sequencerIpAddr, local_gpu_index, remote_gpu_index, FLAGS_clients, thread_count, buffer_slots, iterations_per_thread);

                        } else if(t == ATOMICS_LATENCY_TEST){
                            // Atomics Latency Test
                            testName = "Atomics Latency";
                            test = new rdma::AtomicsLatencyPerfTest(test_ops, FLAGS_server, addresses, FLAGS_port, ownIpPort, sequencerIpAddr, local_gpu_index, remote_gpu_index, FLAGS_clients, thread_count, buffer_slots, iterations_per_thread);

                        } else if(t == ATOMICS_OPERATIONS_COUNT_TEST){
                            // Atomics Operations Count Test
                            testName = "Atomics Operations Count";
                            test = new rdma::AtomicsOperationsCountPerfTest(test_ops, FLAGS_server, addresses, FLAGS_port, ownIpPort, sequencerIpAddr, local_gpu_index, remote_gpu_index, FLAGS_clients, thread_count, buffer_slots, iterations_per_thread);
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
                                test = nullptr;

                                if(maxtransfersize > 0){
                                    uint64_t maxiters = (uint64_t)((long double)maxtransfersize / (long double)packet_size + 0.5);
                                    if(iterations > maxiters){
                                        iterations_per_thread = maxiters / thread_count;
                                        if(iterations_per_thread==0) iterations_per_thread = 1;
                                    }
                                }

                                if(t == LATENCY_TEST){
                                    if(checkInvalidTestParams(packet_size, local_gpu_index, remote_gpu_index)){
                                        testCounter++; csvAddHeader = true; continue;
                                    }
                                    // Latency Test
                                    testName = "Latency";
                                    test = new rdma::LatencyPerfTest(test_ops, FLAGS_server, addresses, FLAGS_port, ownIpPort, sequencerIpAddr, local_gpu_index, remote_gpu_index, FLAGS_clients, thread_count, packet_size, buffer_slots, iterations_per_thread, write_mode);

                                }

                                if(test != nullptr){
                                    testCounter++;
                                    runTest(testCounter, testIterations, testName, test, csvFileName, csvAddHeader);
                                    csvAddHeader = false;
                                }
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