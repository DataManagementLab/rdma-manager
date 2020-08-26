#ifndef PerfTest_H
#define PerfTest_H

#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

namespace rdma {

enum TestMode { TEST_WRITE=0x00, TEST_READ=0x01, TEST_SEND_AND_RECEIVE=0x02, TEST_FETCH_AND_ADD=0x03, TEST_COMPARE_AND_SWAP=0x04 };
enum WriteMode { WRITE_MODE_AUTO=0x00, WRITE_MODE_NORMAL=0x01, WRITE_MODE_IMMEDIATE=0x02 };
const int ATOMICS_SIZE = 8; // 8 bytes = 64bit
const uint64_t NANO_SEC = 1000000000;
const auto CONSOLE_PRINT_NOTATION = std::fixed; // prevents scientific representation of numbers
const auto CONSOLE_PRINT_PRECISION = std::setprecision(6); // decimal precision for numbers
const auto CSV_PRINT_NOTATION = std::fixed; // prevents scientific representation of numbers
const auto CSV_PRINT_PRECISION = std::setprecision(6); // decimal precision for numbers

class PerfTest {
public:
    virtual ~PerfTest() = default;

    virtual std::string getTestParameters() = 0;

    virtual void setupTest() = 0;

    virtual void runTest() = 0;

    virtual std::string getTestResults(std::string csvFileName="", bool csvAddHeader=true) = 0;

    static std::chrono::high_resolution_clock::time_point startTimer(){
        return std::chrono::high_resolution_clock::now();
    }

    static int64_t stopTimer(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point finish){
        std::chrono::duration<double> diff = finish - start;
        return (int64_t)(diff.count() * NANO_SEC);
    }

    static int64_t stopTimer(std::chrono::high_resolution_clock::time_point start){
        return stopTimer(start, startTimer());
    }

    static std::string getMemoryName(int input_gpu_index, int actual_gpu_index=-1){
        if(input_gpu_index >= 0) return "GPU."+std::to_string(input_gpu_index);
        if(input_gpu_index == -1) return "GPU."+(actual_gpu_index>=0 ? std::to_string(actual_gpu_index)+"(D)" : "D");
        if(input_gpu_index == -2) return "GPU."+(actual_gpu_index>=0 ? std::to_string(actual_gpu_index)+"(NUMA)" : "NUMA");
        if(input_gpu_index != -404) return "MAIN";
        return "???";
    }

    static std::string convertTime(long double nanoseconds){
        return convertTime((int64_t)nanoseconds);
    }
    static std::string convertTime(int64_t nanoseconds){
        std::ostringstream oss;
        long double ns = (long double) nanoseconds;
        long double u = 1000.0;
        if(ns<0){ ns = -ns; oss << "-"; }
        if(ns >= u*u*u*86400){
            oss << (round(ns/u/u/u/86400 * 1000)/1000.0) << "days"; return oss.str();
        } else if(ns >= u*u*u*3600){
            oss << (round(ns/u/u/u/3600 * 1000)/1000.0) << "h"; return oss.str();
        } else if(ns >= u*u*u*60){
            oss << (round(ns/u/u/u/60 * 1000)/1000.0) << "min"; return oss.str();
        } else if(ns >= u*u*u){
            oss << (round(ns/u/u/u * 1000)/1000.0) << "s"; return oss.str();
        } else if(ns >= u*u){
            oss << (round(ns/u/u * 1000)/1000.0) << "ms"; return oss.str();
        } else if(ns >= u){
            oss << (round(ns/u * 1000)/1000.0) << "us"; return oss.str();
        } oss << nanoseconds << "ns"; return oss.str();
    }

    static std::string convertByteSize(uint64_t bytes){
        std::ostringstream oss;
        long double b = (long double) bytes;
        long double u = 1000.0; // for KiloByte | use 1024.0 for KibiByte
        if(b >= u*u*u*u*u){
			oss << (round(bytes/u/u/u/u/u * 100)/100.0) << " PB"; return oss.str();
        } else if(b >= u*u*u*u){
			oss << (round(b/u/u/u/u * 100)/100.0) << " TB"; return oss.str();
        } else if(b >= u*u*u){
			oss << (round(b/u/u/u * 100)/100.0) << " GB"; return oss.str();
        } else if(b >= u*u){
			oss << (round(b/u/u * 100)/100.0) << " MB"; return oss.str();
		} else if(b >= u){
            oss << (round(b/u * 100)/100.0) << " KB"; return oss.str();
		} oss << bytes << " B"; return oss.str();
    }

    static std::string convertBandwidth(uint64_t bytes){
        return convertByteSize(bytes) + "/s";
    }

    static std::string convertCountPerSec(long double count){
        std::ostringstream oss;
        long double u = 1000.0;
        if(count >= u*u*u*u*u){
            oss << (round(count/u/u/u/u/u * 100)/100.0) << " petaOp/s"; return oss.str();
        } else if(count >= u*u*u*u){
            oss << (round(count/u/u/u/u * 100)/100.0) << " teraOp/s"; return oss.str();
        } else if(count >= u*u*u){
            oss << (round(count/u/u/u * 100)/100.0) << " gigaOp/s"; return oss.str();
        } else if(count >= u*u){
            oss << (round(count/u/u * 100)/100.0) << " megaOp/s"; return oss.str();
        } else if(count >= u){
            oss << (round(count/u * 100)/100.0) << " kiloOp/s"; return oss.str();
        } oss << count << " Op/s"; return oss.str();
    }
};

}
#endif