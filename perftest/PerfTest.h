#ifndef PerfTest_H
#define PerfTest_H

#include <string>
#include <sstream>
#include <cmath>
#include <chrono>

namespace rdma {

enum TestMode { TEST_WRITE=0x00, TEST_READ=0x01, TEST_SEND_AND_RECEIVE=0x02, TEST_FETCH_AND_ADD=0x03, TEST_COMPARE_AND_SWAP=0x04 };

class PerfTest {
public:
    virtual ~PerfTest() = default;

    virtual std::string getTestParameters() = 0;

    virtual void setupTest() = 0;

    virtual void runTest() = 0;

    virtual std::string getTestResults() = 0;

    static std::chrono::high_resolution_clock::time_point startTimer(){
        return std::chrono::high_resolution_clock::now();
    }

    static int64_t stopTimer(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point finish){
        std::chrono::duration<double> diff = finish - start;
        return (int64_t)(diff.count() * 1000000000);
    }

    static int64_t stopTimer(std::chrono::high_resolution_clock::time_point start){
        return stopTimer(start, startTimer());
    }

    static std::string convertTime(long double nanoseconds){
        return convertTime((int64_t)nanoseconds);
    }
    static std::string convertTime(int64_t nanoseconds){
        std::ostringstream oss;
        long double ns = (long double) nanoseconds;
        long double u = 1000.0;
        if(ns<0){ ns = -ns; oss << "-"; }
        if(ns >= u*u*86400){
            oss << (round(ns/u/u/u/3600 * 1000)/1000.0) << "h"; return oss.str();
        } else if(ns >= u*u*3600){
            oss << (round(ns/u/u/u/60 * 1000)/1000.0) << "m"; return oss.str();
        } else if(ns >= u*u*60){
            oss << (round(ns/u/u/u * 1000)/1000.0) << "s"; return oss.str();
        } else if(ns >= u*u){
            oss << (round(ns/u/u * 1000)/1000.0) << "ms"; return oss.str();
        } else if(ns >= u){
            oss << (round(ns/u * 1000)/1000.0) << "us"; return oss.str();
        } oss << (round(ns * 1000)/1000.0) << "ns"; return oss.str();
    }

    static std::string convertByteSize(uint64_t bytes){
        std::ostringstream oss;
        long double b = (long double) bytes;
        long double u = 1024.0;
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
		} oss << (round(b * 100)/100.0) << " B"; return oss.str();
    }

    static std::string convertBandwidth(uint64_t bytes){
        return convertByteSize(bytes) + "/s";
    }
};

}
#endif