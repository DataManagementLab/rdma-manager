#ifndef LatencyPerfTest_H
#define LatencyPerfTest_H

#include "PerfTest.h"

namespace rdma {

class LatencyPerfTest : public rdma::PerfTest {
public:
    LatencyPerfTest(bool is_server, std::string nodeIdSequencerAddr, int rdma_port, int gpu_index, int thread_count, uint64_t mem_per_thread, uint64_t iterations);
	virtual ~BandwidthPerfTest();
	std::string getTestParameters();
	void setupTest();
	void runTest();
	std::string getTestResults();

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;
	static char testMode; // TODO 0x00=write  0x01=read  0x02=send/recv  0x03=fetch&add  0x04=compare&swap

}

}

#endif