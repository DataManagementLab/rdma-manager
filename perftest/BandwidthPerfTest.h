#ifndef PerfThread_H
#define PerfThread_H

#include "PerfTest.h"
#include "../src/memory/LocalBaseMemoryStub.h"
#include "../src/rdma/RDMAClient.h"
#include "../src/rdma/RDMAServer.h"

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {

class BandwidthPerfThread : public Thread {
public:
	BandwidthPerfThread(std::vector<std::string>& conns, size_t memory_size_per_thread, size_t iterations);
	~BandwidthPerfThread();
	void run();
	bool ready() {
		return m_ready;
	}

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> m_client;
	LocalBaseMemoryStub* m_memory;
	size_t m_memory_size_per_thread;
	size_t m_iterations;
	bool m_is_main_memory;
	std::vector<std::string> m_conns;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};



class BandwidthPerfTest : public rdma::PerfTest {
public:
	BandwidthPerfTest(bool is_server, std::string nodeIdSequencerAddr, int rdma_port, int gpu_index, int thread_count, uint64_t mem_per_thread, uint64_t iterations);
	virtual ~BandwidthPerfTest();
	std::string getTestParameters();
	void setupTest();
	void runTest();
	std::string getTestResults();

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;

private:
	bool m_is_server;
	std::string m_nodeIdSequencerAddr;
	int m_rdma_port;
	int m_gpu_index;
	int m_thread_count;
	uint64_t m_memory_per_thread;
	uint64_t m_memory_size;
	uint64_t m_iterations;
	std::vector<std::string> m_conns;
	std::vector<BandwidthPerfThread*> m_threads;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;
	RDMAClient<ReliableRDMA>* m_client;
};



}
#endif