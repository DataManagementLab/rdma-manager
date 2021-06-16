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
#include <stdio.h>
#include <unistd.h>
#include <string.h> /* For strncpy */
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>

#include "CpuNumaUtils.h"

using namespace std;

// #define DEBUGCODE
#if defined(DEBUG)
#define DebugCode(code_fragment) \
    {                            \
        code_fragment            \
    }
#else
#define DebugCode(code_fragment)
#endif

//To be implemented MACRO
#define TO_BE_IMPLEMENTED(code_fragment)

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
    Config(const std::string& prog_name);
    ~Config();

    //RDMA
    static size_t RDMA_MEMSIZE;
    static uint32_t RDMA_PORT;
    static uint32_t RDMA_NUMAREGION;
    static std::string RDMA_DEVICE_FILE_PATH;
    static uint32_t RDMA_IBPORT;
    static uint32_t RDMA_GID_INDEX; // if set to -1 do not use the gid, else use the gid on the port with the given index
    static uint32_t RDMA_MAX_WR;
    const static uint32_t RDMA_MAX_SGE = 1;
    const static size_t RDMA_UD_OFFSET = 40;
    const static int RDMA_SLEEP_INTERVAL = 100 * 1000;
    
    static uint32_t RDMA_UD_MTU;

    const static int PROTO_MAX_SOCKETS = 1024;

    static std::string SEQUENCER_IP;
    static uint32_t SEQUENCER_PORT;

    static std::string RDMA_INTERFACE;
    static std::string RDMA_DEV_NAME;

    const static uint32_t MAX_RC_INLINE_SEND = 220;
    const static uint32_t MAX_UD_INLINE_SEND = 188;

    static uint32_t MLX5_SINGLE_THREADED; //If set to 1 -> disables all spin locking on queues. Note: overwrites environment the variable!

    //SYSTEM
    const static uint32_t CACHELINE_SIZE = 64;

    //THREAD
    static vector<int> THREAD_CPUS;
    static vector<vector<int>> NUMA_THREAD_CPUS;
    
    //LOGGING
    static int LOGGING_LEVEL; //0=all, 1=ERR, 2=DBG, 3=INF, (>=4)=NONE

    //TEST
    static int HELLO_PORT;

    // static string& getIPFromNodeId(NodeID& nodeid);
    // static string& getIPFromNodeId(const NodeID& nodeid);
    static string getIP(std::string &interface);

  private:
    static void load(const string& exec_path);
    static void unload();

    static void set(string key, string value);
    static void init_vector(vector<string> &values, string csv_list);
    static void init_vector(vector<int> &values, string csv_list);

};

} // end namespace rdma

#endif /* CONFIG_HPP_ */
