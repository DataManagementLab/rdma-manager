//CPPUnit
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>

#include <unistd.h>
#include <stdio.h>
#include <getopt.h>

#include "../src/utils/Config.h"
#include "Tests.h"
// #include "Test.h"

#define no_argument 0
#define required_argument 1
#define optional_argument 2

static void usage()
{
    cout << "rdma_test -n #number" << endl;
    cout << endl;
    cout << "Tests:" << endl;
    
    //RDMA
    cout << "101: \t rdma/TestRDMAServer" << endl;
    cout << "102: \t rdma/TestRDMAServerMultipleClients" << endl;
    cout << "103: \t rdma/TestSimpleUD" << endl;
    cout << "104: \t rdma/TestRDMAServerMCast" << endl;
    cout << "105: \t rdma/TestRDMAServerSRQ" << endl;
    cout << "106: \t RPC/TestRDMARPC" << endl;
    
    //MISC
    cout << "201: \t utils/TestConfig" << endl;
    cout << "202: \t thread/TestThread" << endl;
    cout << "203: \t proto/TestProtoServer" << endl;

    cout << endl;
}

static void runtest(int t)
{
    // Adds the test to the list of test to run
    // Create the event manager and test controller
    CPPUNIT_NS::TestResult controller;

    // Add a listener that colllects test result
    CPPUNIT_NS::TestResultCollector result;
    controller.addListener(&result);

    // Add a listener that print dots as test run.
    CPPUNIT_NS::BriefTestProgressListener progress;
    controller.addListener(&progress);

    //controller.push

    // Add the top suite to the test runner
    CPPUNIT_NS::TestRunner runner;

    //
    //  CppUnit::TextUi::TestRunner runner;
    //
    //  // Change the default outputter to a compiler error format outputter
    //  runner.setOutputter(
    //      new CppUnit::CompilerOutputter(&runner.result(), std::cerr));


    switch (t)
    {
    //RDMA
    case 101:
        runner.addTest(TestRDMAServer::suite());
        break;
    // case 102:
    //     runner.addTest(TestRDMAServerMultClients::suite());
    //     break;
    // case 103:
    //     runner.addTest(TestSimpleUD::suite());
    //     break;
    // case 104:
    //     runner.addTest(TestRDMAServerMCast::suite());
    //     break;
    // case 105:
    //     runner.addTest(TestRDMAServerSRQ::suite());
    //     break;

    // case 106:
    //     runner.addTest(TestRPC::suite());
    //     break;

    // //MISC
    // case 201:
    //     runner.addTest(TestConfig::suite());
    //     break;
    // case 202:
    //     runner.addTest(TestThread::suite());
    //     break;
    // case 203:
    //     runner.addTest(TestProtoServer::suite());
    //     break;
    default:
        cout << "No test with number " << t << " exists." << endl;
        return;
    }

    runner.run(controller);

    // Print test in a compiler compatible format.
    CPPUNIT_NS::CompilerOutputter outputter(&result, std::cerr);
    outputter.write();
}

struct config_t
{
    // string runmode = "";
    int number = 0;
    int port = 0;
};

int main(int argc, char *argv[])
{
    struct config_t config;

    while (1)
    {
        struct option long_options[] = {{"number", required_argument, 0, 'n'}, {"port", required_argument, 0, 'p'}};

        int c = getopt_long(argc, argv, "r:n:p:", long_options, NULL);
        if (c == -1)
            break;

        switch (c)
        {
        // case 'r':
        //     config.runmode = string(optarg);
        //     break;
        case 'n':
            config.number = strtoul(optarg, NULL, 0);
            break;
        case 'p':
            std::cout << "P" << std::endl;
            config.port = strtoul(optarg, NULL, 0);
            std::cout << "Port " << config.port << std::endl;
            break;
        default:
            usage();
            return 1;
        }
    }

    // load  configuration
    string prog_name = string(argv[0]); 
    static Config conf(prog_name);

    // run program
    if (config.number > 0)
    {
        runtest(config.number);
        return 0;
    }
    usage();
}
