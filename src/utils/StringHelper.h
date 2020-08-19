/**
 * @file StringHelper.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef SRC_UTILS_STRINGHELPER_H_
#define SRC_UTILS_STRINGHELPER_H_

#include <algorithm>
#include <cctype>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

namespace rdma {

class StringHelper {
 public:
  static vector<string> split(string value, string sep = ",") {
    vector<string> values;
    char* cvalue = new char[value.length() + 1];
    strcpy(cvalue, value.c_str());
    char* token = strtok(cvalue, sep.c_str());

    while (token) {
      values.push_back(token);
      token = strtok(nullptr, sep.c_str());
    }

    delete[] cvalue;
    return values;
  }


  static void splitPerf(const string& value, std::vector<string>& retValues ,string sep = ",") {
      char* cvalue = new char[value.length() + 1];
      strcpy(cvalue, value.c_str());
      char* token = strtok(cvalue, sep.c_str());

      while (token) {
        retValues.emplace_back(token);
        token = strtok(nullptr, sep.c_str());
    }
    delete[] cvalue;
  }

  static inline string ltrim(string &value){
    if(value.empty()) return value;
    size_t i = value.find_first_not_of(" \n\r\t");
    if(i != string::npos) value.erase(0, i);
    return value;
  }

  static inline string rtrim(string &value){
    if(value.empty()) return value;
    size_t i = value.find_last_not_of(" \n\r\t");
    if(i != string::npos) value.erase(i+1);
    return value;
  }

  static inline string trim(string &value){
    ltrim(value);
    return rtrim(value);
  }

  static inline string lower(string &value){
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return tolower(c); });
    return value;
  }

  static inline string upper(string &value){
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return toupper(c); });
    return value;
  }

  /* Function: parseByteSize
   * ------------------------
   * Parses a size value for data out of a given string 
   * and returns result in bytes. Spaces allowed and 
   * not case sensitive.
   * E.G. "24 B", "12GB", "94 tb "
   * 
   * value:  value that should be 
   * return:  byte value or throws exception if could not parse
   */
  static int64_t parseByteSize(string value){
    value = trim(value);
    value = upper(value);
    size_t i = value.rfind("KB");
    if(i != string::npos){ value = value.substr(0, i); return (int64_t)(std::atof(rtrim(value).c_str())*(int64_t)1000); }
    i = value.rfind("MB");
    if(i != string::npos){ value = value.substr(0, i); return (int64_t)(std::atof(rtrim(value).c_str())*(int64_t)1000000); }
    i = value.rfind("GB");
    if(i != string::npos){ value = value.substr(0, i); return (int64_t)(std::atof(rtrim(value).c_str())*(int64_t)1000000000); }
    i = value.rfind("TB");
    if(i != string::npos){ value = value.substr(0, i); return (int64_t)(std::atof(rtrim(value).c_str())*(int64_t)1000000000000); }
    i = value.rfind("PB");
    if(i != string::npos){ value = value.substr(0, i); return (int64_t)(std::atof(rtrim(value).c_str())*(int64_t)1000000000000000); }
    i = value.rfind("B");
    if(i != string::npos){ value = value.substr(0, i); return (int64_t)std::stoi(rtrim(value)); }
    return (int64_t)std::stoi(value); 
  }

};
}

#endif /* SRC_UTILS_STRINGHELPER_H_ */
