/**
 * @file Network.h
 * @author cbinnig, lthostrup, tziegler
 * @date 2018-08-17
 */



#ifndef NETWORK_H_
#define NETWORK_H_

#include "./Config.h"
#include <endian.h>
#include <arpa/inet.h>

namespace rdma {

class Network {
 public:

  static inline uint64_t bigEndianToHost(uint64_t be) {
    return be64toh(be);
  }

  static bool isValidIP(const string &ip){
    struct sockaddr_in sa;
    int result = inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr));
    return result != 0;
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
};

}

#endif /* NETWORK_H_ */
