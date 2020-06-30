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

class BandwidthPerfClientThread : public Thread {
public:
	BandwidthPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, size_t memory_size_per_thread, size_t iterations);
	~BandwidthPerfClientThread();
	void run();
	bool ready() {
		return m_ready;
	}

	int64_t m_elapsedWriteMs = -1;
	int64_t m_elapsedReadMs = -1;
	int64_t m_elapsedSendMs = -1;

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> *m_client;
	LocalBaseMemoryStub *m_local_memory;
	size_t m_memory_size_per_thread;
	size_t m_iterations;
	std::vector<std::string> m_rdma_addresses;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};


class BandwidthPerfServerThread : public Thread {
public:
	BandwidthPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t memory_size_per_thread, size_t iterations);
	~BandwidthPerfServerThread();
	void run();
	bool ready(){
		return m_ready;
	}

private:
	bool m_ready = false;
	size_t m_memory_size_per_thread;
	size_t m_iterations;
	RDMAServer<ReliableRDMA> *m_server;
	LocalBaseMemoryStub *m_local_memory;
};


class BandwidthPerfTest : public rdma::PerfTest {
public:
	BandwidthPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, int gpu_index, int thread_count, uint64_t memory_size_per_thread, uint64_t iterations);
	virtual ~BandwidthPerfTest();
	std::string getTestParameters();
	void setupTest();
	void runTest();
	std::string getTestResults(std::string csvFileName="", bool csvAddHeader=true);

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;
	static TestMode testMode;

private:
	bool m_is_server;
	NodeIDSequencer *m_nodeIDSequencer;
	std::vector<std::string> m_rdma_addresses;
	int m_rdma_port;
	int m_gpu_index;
	int m_thread_count;
	uint64_t m_memory_size_per_thread;
	uint64_t m_memory_size;
	uint64_t m_iterations;
	std::vector<BandwidthPerfClientThread*> m_client_threads;
	std::vector<BandwidthPerfServerThread*> m_server_threads;
	int64_t m_elapsedWriteMs;
	int64_t m_elapsedReadMs;
	int64_t m_elapsedSendMs;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;

	void makeThreadsReady(TestMode testMode);
	void runThreads();
};



}
#endif