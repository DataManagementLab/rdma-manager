#ifndef OperationsCountPerfTest_H
#define OperationsCountPerfTest_H

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

class OperationsCountPerfClientThread : public Thread {
public:
	OperationsCountPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, size_t packet_size, int buffer_slots, size_t iterations, size_t max_rdma_wr_per_thread);
	~OperationsCountPerfClientThread();
	void run();
	bool ready() {
		return m_ready;
	}

	int64_t m_elapsedWrite = -1;
	int64_t m_elapsedRead = -1;
	int64_t m_elapsedSend = -1;

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> *m_client;
	LocalBaseMemoryStub *m_local_memory;
	size_t m_packet_size;
	int m_buffer_slots;
	size_t m_memory_size_per_thread;
	size_t m_iterations;
	size_t m_max_rdma_wr_per_thread;
	std::vector<std::string> m_rdma_addresses;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};


class OperationsCountPerfServerThread : public Thread {
public:
	OperationsCountPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations, size_t max_rdma_wr_per_thread, int thread_id);
	~OperationsCountPerfServerThread();
	void run();
	bool ready(){
		return m_ready;
	}

private:
	bool m_ready = false;
	size_t m_packet_size;
	int m_buffer_slots;
	size_t m_memory_size_per_thread;
	size_t m_iterations;
	size_t m_max_rdma_wr_per_thread;
	int m_thread_id;
	RDMAServer<ReliableRDMA> *m_server;
	LocalBaseMemoryStub *m_local_memory;
};


class OperationsCountPerfTest : public rdma::PerfTest {
public:
	OperationsCountPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, int gpu_index, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations);
	virtual ~OperationsCountPerfTest();
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
	uint64_t m_packet_size;
	int m_buffer_slots;
	uint64_t m_memory_size;
	uint64_t m_iterations;
	std::vector<OperationsCountPerfClientThread*> m_client_threads;
	std::vector<OperationsCountPerfServerThread*> m_server_threads;
	int64_t m_elapsedWrite;
	int64_t m_elapsedRead;
	int64_t m_elapsedSend;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;

	void makeThreadsReady(TestMode testMode);
	void runThreads();
};



}
#endif