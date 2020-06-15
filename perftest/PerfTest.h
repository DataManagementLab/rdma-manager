#ifndef PerfTest_H
#define PerfTest_H

#include <string>

namespace rdma {

class PerfTest {
public:
    virtual ~PerfTest() = default;

    virtual std::string getTestParameters() = 0;

    virtual void setupTest() = 0;

    virtual void runTest() = 0;

    virtual std::string getTestResults() = 0;
};

}
#endif