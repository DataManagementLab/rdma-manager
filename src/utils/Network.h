/**
 * @file Network.h
 * @author cbinnig, lthostrup, tziegler
 * @date 2018-08-17
 */



#ifndef NETWORK_H_
#define NETWORK_H_

#include "./Config.h"
#include <endian.h>
#include <stdio.h>
#include <string.h>

namespace rdma {

class Network {
public:

  static inline uint64_t bigEndianToHost(uint64_t be) {
    return be64toh(be);
  }

  static bool isConnection(const string& region) {
    size_t found = region.find(":");
    if (found != std::string::npos) {
      return true;
    }
    return false;
  }

  static string getConnection(const string& address, const int& port) {
    stringstream ss;
    ss << address;
    ss << ":";
    ss << port;
    return ss.str();
  }

  static string getAddressOfConnection(const string& conn) {
    size_t found = conn.find(":");
    if (found != std::string::npos) {
      return conn.substr(0, found);
    }
    throw invalid_argument("Connection has bad format");
  }

  static size_t getPortOfConnection(const string& conn) {
    size_t found = conn.find(":");
    if (found != std::string::npos) {
      found++;
      size_t length = conn.length() - found;
      string portStr = conn.substr(found, length);
      return stoi(portStr);
    }
    throw invalid_argument("Connection has bad format");
  }

  static string getLocalAddress(){
    int sock = socket(PF_INET, SOCK_DGRAM, 0);
    sockaddr_in loopback;

    if(sock == -1){
        std::cerr << "Could not socket\n";
        return NULL;
    }

    memset(&loopback, 0, sizeof(loopback));
    loopback.sin_family = AF_INET;
    loopback.sin_addr.s_addr = INADDR_LOOPBACK;   // using loopback ip address
    loopback.sin_port = htons(9);                 // using debug port

    if(connect(sock, reinterpret_cast<sockaddr*>(&loopback), sizeof(loopback)) == -1){
        close(sock);
        std::cerr << "Could not connect\n";
        return NULL;
    }

    socklen_t addrlen = sizeof(loopback);
    if(getsockname(sock, reinterpret_cast<sockaddr*>(&loopback), &addrlen) == -1){
        close(sock);
        std::cerr << "Could not getsockname\n";
        return NULL;
    }

    close(sock);

    char buf[INET_ADDRSTRLEN];
    if(inet_ntop(AF_INET, &loopback.sin_addr, buf, INET_ADDRSTRLEN) == 0x0){
        std::cerr << "Could not inet_ntop\n";
        return NULL;
    }
    return string(buf);
  }
};

}

#endif /* NETWORK_H_ */
