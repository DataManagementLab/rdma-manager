/**
 * @file Config.h
 * @author cbinnig, lthostrup, tziegler
 * @date 2018-08-17
 */



#ifndef CONFIG_HPP_
#define CONFIG_HPP_

//Includes
#include <iostream>
#include <stddef.h>
#include <sstream>
#include <unistd.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <google/protobuf/stubs/common.h>
#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h> /* For strncpy */
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>

using namespace std;

#define RDMA_TRANSPORT 0 //0=RC, 1=UD

#if RDMA_TRANSPORT == 0
#define DPI_UNIT_TEST_RC(test) CPPUNIT_TEST(test)
#define DPI_UNIT_TEST_UD(test)
#elif RDMA_TRANSPORT == 1
#define DPI_UNIT_TEST_RC(test)
#define DPI_UNIT_TEST_UD(test) CPPUNIT_TEST(test)
#endif

//#define DEBUGCODEANOTHER
#if defined(DEBUGCODEANOTHER)

#define DEBUG_WRITE(outputStream, className, funcName, message)                     \
    do                                                                              \
    {                                                                               \
        std::string header = std::string("[") + className + "::" + funcName + "] "; \
        outputStream << std::left << header << message << std::endl;                \
    } while (false)

#define RESULT_WRITE(outputStream, message)                \
    do                                                     \
    {                                                      \
        outputStream << std::left << message << std::endl; \
    } while (false)

#define DEBUG_OUT(x)                     \
    do                                   \
    {                                    \
        if (debugging_enabled)           \
        {                                \
            std::cout << x << std::endl; \
        }                                \
    } while (0);

#else
#define DEBUG_WRITE(outputStream, className, funcName, message)
#define RESULT_WRITE(outputStream, message)
#define DEBUG_OUT(x)
#endif



// #define DEBUGCODE
#if defined(DEBUGCODE)
#define DebugCode(code_fragment) \
    {                            \
        code_fragment            \
    }
#else
#define DebugCode(code_fragment)
#endif


//To be implemented MACRO
#define TO_BE_IMPLEMENTED(code_fragment)
#define DPI_UNIT_TEST_SUITE(suite) CPPUNIT_TEST_SUITE(suite)
#define DPI_UNIT_TEST(test) CPPUNIT_TEST(test)
#define DPI_UNIT_TEST_SUITE_END() CPPUNIT_TEST_SUITE_END()

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

//typedefs
typedef unsigned long long uint128_t;
typedef uint64_t NodeID;
// typedef uint64_t Offset;

namespace rdma
{

//Constants
class Config
{
  public:
    Config(const string& prog_name)
    {
        load(prog_name);
    }

    ~Config()
    {
        unload();
    }

    //RDMA
    static size_t RDMA_MEMSIZE;
    static uint32_t RDMA_PORT;
    static uint32_t RDMA_NUMAREGION;
    static uint32_t RDMA_DEVICE;
    static uint32_t RDMA_IBPORT;
    static uint32_t RDMA_MAX_WR;
    const static uint32_t RDMA_MAX_SGE = 1;
    const static size_t RDMA_UD_OFFSET = 40;
    const static int RDMA_SLEEP_INTERVAL = 100 * 1000;
    
    const static int PROTO_MAX_SOCKETS = 1024;

    //SYSTEM
    static uint32_t CACHELINE_SIZE;

    //THREAD
    static vector<int> THREAD_CPUS;

    //LOGGING
    static int LOGGING_LEVEL; //0=all, 1=ERR, 2=DBG, 3=INF, (>=4)=NONE

    //TEST
    static int HELLO_PORT;

    // static string& getIPFromNodeId(NodeID& nodeid);
    // static string& getIPFromNodeId(const NodeID& nodeid);

  private:
    static void load(const string& exec_path);
    static void unload();

    static void set(string key, string value);
    static void init_vector(vector<string> &values, string csv_list);
    static void init_vector(vector<int> &values, string csv_list);

    static string getIP();
};

} // end namespace rdma

#endif /* CONFIG_HPP_ */
