#ifndef PerfTest_H
#define PerfTest_H

#include <string>

namespace rdma {

class PerfTest {
public:
    virtual PerfTest() = default;
    virtual ~PerfTest() = default;

    virtual void setupTest() = 0;

    virtual std::string getTestParameters() = 0;

    virtual void runTest() = 0;
};

}
#endif