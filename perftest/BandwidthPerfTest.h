#ifndef BandwidthPerfTest_H
#define BandwidthPerfTest_H

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
	BandwidthPerfThread(std::vector<std::string>& rdma_addresses, size_t memory_size_per_thread, size_t iterations);
	~BandwidthPerfThread();
	void run();
	bool ready() {
		return m_ready;
	}

	int64_t m_elapsedWriteMs = -1;
	int64_t m_elapsedReadMs = -1;
	int64_t m_elapsedSendMs = -1;
	int64_t m_elapsedFetchAddMs = -1;
	int64_t m_elapsedCompareSwapMs = -1;

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> m_client;
	LocalBaseMemoryStub* m_memory;
	size_t m_memory_size_per_thread;
	size_t m_iterations;
	bool m_is_main_memory;
	std::vector<std::string> m_rdma_addresses;
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
	static char testMode; // 0x00=write  0x01=read  0x02=send/recv  0x03=fetch&add  0x04=compare&swap

private:
	bool m_is_server;
	NodeIDSequencer *m_nodeIDSequencer;
	std::vector<std::string> m_rdma_addresses;
	int m_rdma_port;
	int m_gpu_index;
	int m_thread_count;
	uint64_t m_memory_per_thread;
	uint64_t m_memory_extra;
	uint64_t m_memory_size;
	uint64_t m_iterations;
	std::vector<BandwidthPerfThread*> m_threads;
	int64_t m_elapsedWriteMs;
	int64_t m_elapsedReadMs;
	int64_t m_elapsedSendMs;
	int64_t m_elapsedFetchAddMs;
	int64_t m_elapsedCompareSwapMs;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;
	RDMAClient<ReliableRDMA>* m_client;

	void makeThreadsReady(char testMode); // 0x00=write  0x01=read  0x02=send/recv  0x03=fetch&add  0x04=compare&swap
	void runThreads();
};



}
#endif