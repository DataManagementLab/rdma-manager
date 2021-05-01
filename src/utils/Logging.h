/**
 * @file Logging.h
 * @author cbinnig, lthostrup, tziegler
 * @date 2018-08-17
 */



#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include "./Config.h"

#include <iostream>
#include <string>

#define ESC_NORMAL "\e[0m"
#define ESC_DIM "\e[2m"
#define ESC_UNSET "\e[39m"
#define ESC_RED "\e[91m"
#define ESC_BLUE "\e[94m"
#define ESC_YELLOW "\e[93m"

namespace rdma {

class Logging {
 public:
  static void debug(string filename, int line, string msg) {
    //avoid unused variable warning
    (void) filename;
    (void) line;
    (void) msg;
#ifdef DEBUG
    if(Config::LOGGING_LEVEL<=1)
    log("[DEBUG]: ", filename, line, msg);
#endif
  }

  static void error(string filename, int line, string msg) {
    if (Config::LOGGING_LEVEL <= 4)
      log(ESC_RED "[ERROR]:" ESC_UNSET " ", filename, line, msg);
  }

  static void errorNo(string filename, int line, char* errorMsg, int errNo) {
    if (Config::LOGGING_LEVEL <= 4)
      cerr << "[ERROR NO]" << filename << " at " << line << " (" << errorMsg
           << ": " << errNo << ")" << endl;
  }

  static void fatal(string filename, int line, string msg) {
    if (Config::LOGGING_LEVEL <= 5)
      log(ESC_RED "[FATAL]:" ESC_UNSET " ", filename, line, msg);
    exit(1);
  }

  static void info(string msg) {
    if (Config::LOGGING_LEVEL <= 2)
      log(ESC_BLUE "[INFO]:" ESC_UNSET "  ", msg);
  }

  static void warn(string msg) {
    if (Config::LOGGING_LEVEL <= 3)
      log(ESC_YELLOW "[WARN]:" ESC_UNSET "  ", msg);
  }
 private:
  static void log(string type, string filename, int line, string msg) {
    cerr << type << msg << endl
         << " at " ESC_DIM << filename << ":" << line << ESC_NORMAL << endl;
  }

  static void log(string type, string msg) {
    cerr << type << msg << endl;
  }
};

}  // end namespace rdma

#endif /* LOGGING_HPP_ */
