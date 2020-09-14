#include "PerfTests.h"
//#include "RemoteMemoryPerf.h"

//
// Created by Tilo Gaulke on 11.10.19.
//
using namespace rdma;

static void printUsage() {
    cout << "istore2_perftest -n #testNum options" << endl;

    cout << "Tests:" << endl;
    cout << "1: \t RemoteMemoryClient" << endl;
    cout << "2: \t RemoteMemoryServer" << endl;
    cout << "11: \t XRC_RemoteMemoryClient" << endl;
    cout << "12: \t XRC_RemoteMemoryServer" << endl;
    // cout << "3: \t RemoteScanClient" << endl;
    // cout << "4: \t RemoteScanServer" << endl;
    cout << "101: \t MulticastClient" << endl;
    cout << "102: \t MulticastServer" << endl;
    cout << "103: \t MulticastLatClient" << endl;
    cout << "104: \t MulticastLatServer" << endl;
    cout << "201: \t RPCPerfClient" << endl;
    cout << "202: \t RPCPerfServer" << endl;
    cout << "301: \t FetchAndAddPerfClient" << endl;
    cout << "302: \t FetchAndAddPerfServer" << endl;
}


rdma::PerfTest* createTest(config_t& config) {
    PerfTest* test = nullptr;
    switch (config.number) {
        case 1:
            test = new rdma::RemoteMemoryPerf(config, true);
            break;
        case 2:
            test = new rdma::RemoteMemoryPerf(config, false);
            break;
        case 11:
            test = new rdma::XRC_RemoteMemoryPerf(config, true);
            break;
        case 12:
            test = new rdma::XRC_RemoteMemoryPerf(config, false);
            break;
        case 3:
            //test = new RemoteScanPerf(config, true);
            break;
        case 4:
            //test = new RemoteScanPerf(config, false);
            break;
        case 101:
            test = new MulticastPerf(config, true);
            break;
        case 102:
            test = new MulticastPerf(config, false);
            break;
        case 103:
            test = new MulticastPerfLat(config, true);
            break;
        case 104:
            test = new MulticastPerfLat(config, false);
            break;
        case 201:
            test = new RPCPerf(config,true);
            break;
        case 202:
            test = new RPCPerf(config,false);
            break;
        case 301:
            test = new FetchAndAddPerf(config,true);
            break;
        case 302:
            test = new FetchAndAddPerf(config,false);
            break;
    }

    return test;
}


int main(int argc, char *argv[])
{
    // parse parameters
    struct config_t config = rdma::PerfTest::parseParameters(argc, argv);
    
    // load configuration
    static Config conf(argv[0]);
    conf.RDMA_NUMAREGION = config.numa;
    conf.RDMA_INTERFACE = config.interface;

    // check if test number is defined
    if (config.number <= 0) {
        printUsage();
        return -1;
    }

    // create test and validate
    rdma::PerfTest* test = createTest(config);
    // check if test number is a valid test
    if (test == nullptr) {
        cout << "Invalid Test specified." << endl;
        printUsage();
        return -1;
    }
    // and check if test is runnable
    if (!test->isRunnable()) {
        test->printUsage();
        return -1;
    }

    // run test client or server
    if (test->isClient()) {
        test->runClient();
        test->printHeader();
        test->printResults();
    } else {
        test->runServer();
    }
    delete test;

    //done
    return 0;

}
