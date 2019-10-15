#include "../src/perftest/PerfTest.h"
#include "../src/perftest/RemoteMemoryPerf.h"
//
// Created by Tilo Gaulke on 11.10.19.
//
using namespace rdma;

static void printUsage() {
    cout << "istore2_perftest -n #testNum options" << endl;

    cout << "Tests:" << endl;
    cout << "1: \t RemoteMemoryClient" << endl;
    cout << "2: \t RemoteMemoryServer" << endl;
    /*
    cout << "3: \t RemoteScanClient" << endl;
    cout << "4: \t RemoteScanServer" << endl;
    cout << "101: \t MulticastClient" << endl;
    cout << "102: \t MulticastServer" << endl;
    cout << "103: \t SWMulticastClient" << endl;
    cout << "104: \t SWMulticastServer" << endl;*/

}


rdma::PerfTest* createTest(config_t& config) {
    PerfTest* test;
    switch (config.number) {
        case 1:
            test = new rdma::RemoteMemoryPerf(config, true);
            break;
        case 2:
            test = new rdma::RemoteMemoryPerf(config, false);
            break;
        case 3:
            //test = new RemoteScanPerf(config, true);
            break;
        case 4:
            //test = new RemoteScanPerf(config, false);
            break;
        case 101:
            //test = new MulticastPerf(config, true);
            break;
        case 102:
            //test = new MulticastPerf(config, false);
            break;
        case 103:
            //test = new SWMulticastPerf(config, true);
            break;
        case 104:
            //test = new SWMulticastPerf(config, false);
            break;
    }

    return test;
}


int main(int argc, char *argv[])
{

    // load configuration
    //static Config conf;

    // parse parameters
    struct config_t config = rdma::PerfTest::parseParameters(argc, argv);

    // check if test number is defined
    if (config.number <= 0) {
        printUsage();
        return -1;
    }

    // create test and check if test is runnable
    rdma::PerfTest* test = createTest(config);
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
    return 1;

}
