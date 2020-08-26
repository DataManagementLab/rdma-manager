#ifndef LatencyPerfTest_H
#define LatencyPerfTest_H

#include "PerfTest.h"
#include "../src/memory/LocalBaseMemoryStub.h"
#include "../src/rdma/RDMAClient.h"
#include "../src/rdma/RDMAServer.h"
#include "../src/thread/Thread.h"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <algorithm>

namespace rdma {

class LatencyPerfClientThread : public Thread {
public:
	LatencyPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, size_t packet_size, int buffer_slots, size_t iterations_per_thread, WriteMode write_mode);
	~LatencyPerfClientThread();
	void run();
	bool ready() {
		return m_ready;
	}

	int64_t m_sumWriteMs = 0, m_minWriteMs=std::numeric_limits<int64_t>::max(), m_maxWriteMs=-1;
	int64_t m_sumReadMs = 0, m_minReadMs=std::numeric_limits<int64_t>::max(), m_maxReadMs=-1;
	int64_t m_sumSendMs = 0, m_minSendMs=std::numeric_limits<int64_t>::max(), m_maxSendMs=-1;
	int64_t *m_arrWriteMs, *m_arrReadMs, *m_arrSendMs;

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> *m_client;
	LocalBaseMemoryStub *m_local_memory;
	size_t m_packet_size;
	int m_buffer_slots;
	size_t m_memory_size_per_thread;
	size_t m_iterations_per_thread;
	WriteMode m_write_mode;
	std::vector<std::string> m_rdma_addresses;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};


class LatencyPerfServerThread : public Thread {
public:
	LatencyPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations_per_thread, WriteMode write_mode, int thread_id);
	~LatencyPerfServerThread();
	void run();
	bool ready(){
		return m_ready;
	}

private:
	bool m_ready = false;
	size_t m_packet_size;
	int m_buffer_slots;
	size_t m_memory_size_per_thread;
	size_t m_iterations_per_thread;
	WriteMode m_write_mode;
	int m_thread_id;
	RDMAServer<ReliableRDMA> *m_server;
	LocalBaseMemoryStub *m_local_memory;
};


class LatencyPerfTest : public rdma::PerfTest {
public:
    LatencyPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations_per_thread, WriteMode write_mode);
	virtual ~LatencyPerfTest();
	std::string getTestParameters();
	void setupTest();
	void runTest();
	std::string getTestResults(std::string csvFileName="", bool csvAddHeader=true);

	static const WriteMode DEFAULT_WRITE_MODE = WRITE_MODE_IMMEDIATE;

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;
	static TestMode testMode;
	static int thread_count;

private:
	bool m_is_server;
	NodeIDSequencer *m_nodeIDSequencer;
	std::vector<std::string> m_rdma_addresses;
	int m_rdma_port;
	std::string m_ownIpPort;
	std::string m_sequencerIpPort;
	int m_local_gpu_index;
	int m_actual_gpu_index;
	int m_remote_gpu_index;
	uint64_t m_packet_size;
	int m_buffer_slots;
	uint64_t m_memory_size;
	uint64_t m_iterations_per_thread;
	WriteMode m_write_mode;
	std::vector<LatencyPerfClientThread*> m_client_threads;
	std::vector<LatencyPerfServerThread*> m_server_threads;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;

	std::string getTestParameters(bool forCSV);
	void makeThreadsReady(TestMode testMode);
	void runThreads();
};

}

#endif