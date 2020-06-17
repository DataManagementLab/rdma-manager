#ifndef PerfTest_H
#define PerfTest_H

#include <string>
#include <sstream>
#include <chrono>

namespace rdma {

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

    static int stopTimer(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point finish){
        std::chrono::duration<double> diff = finish - start;
        return (int)(diff.count() * 1000);
    }

    static int stopTimer(std::chrono::high_resolution_clock::time_point start){
        return stopTimer(start, startTimer());
    }

    static std::string convertByteSize(uint64_t bytes){
        std::ostringstream oss;
        uint64_t u = 1024;
        if(bytes >= u*u*u*u*u){
			oss << (bytes/u/u/u/u/u) << " PB"; return oss.str();
        } else if(bytes >= u*u*u*u){
			oss << (bytes/u/u/u/u) << " TB"; return oss.str();
        } else if(bytes >= u*u*u){
			oss << (bytes/u/u/u) << " GB"; return oss.str();
        } else if(bytes >= u*u){
			oss << (bytes/u/u) << " MB"; return oss.str();
		} else if(bytes >= u){
            oss << (bytes/u) << " KB"; return oss.str();
		} oss << bytes << " B"; return oss.str();
    }

    static std::string convertBandwidth(uint64_t bytes){
        return convertByteSize(bytes * 8) + "its/s";
    }
};

}
#endif