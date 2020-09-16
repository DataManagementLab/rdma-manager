#ifndef BandwidthPerfTest_H
#define BandwidthPerfTest_H

#include "PerfTest.h"
#include "../src/memory/LocalBaseMemoryStub.h"
#include "../src/rdma/RDMAClient.h"
#include "../src/rdma/RDMAServer.h"
#include "../src/thread/Thread.h"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {

class BandwidthPerfClientThread : public Thread {
public:
	BandwidthPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, size_t packet_size, int buffer_slots, size_t iterations_per_thread, size_t max_rdma_wr_per_thread, WriteMode write_mode);
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
	size_t m_packet_size;
	int m_buffer_slots;
	size_t m_memory_size_per_thread;
	size_t m_iterations_per_thread;
	size_t m_max_rdma_wr_per_thread;
	WriteMode m_write_mode;
	std::vector<std::string> m_rdma_addresses;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};


class BandwidthPerfServerThread : public Thread {
public:
	BandwidthPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations_per_thread, size_t max_rdma_wr_per_thread, WriteMode write_mode, int thread_id);
	~BandwidthPerfServerThread();
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
	size_t m_max_rdma_wr_per_thread;
	WriteMode m_write_mode;
	int m_thread_id;
	RDMAServer<ReliableRDMA> *m_server;
	LocalBaseMemoryStub *m_local_memory;
};


class BandwidthPerfTest : public rdma::PerfTest {
public:
	BandwidthPerfTest(int testOperations, bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int client_count, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations_per_thread, WriteMode write_mode);
	virtual ~BandwidthPerfTest();
	std::string getTestParameters();
	void setupTest();
	void runTest();
	std::string getTestResults(std::string csvFileName="", bool csvAddHeader=true);

	static const WriteMode DEFAULT_WRITE_MODE = WRITE_MODE_NORMAL;

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;
	static TestOperation testOperation;

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
	int m_thread_count;
	uint64_t m_packet_size;
	int m_buffer_slots;
	uint64_t m_memory_size;
	uint64_t m_iterations_per_thread;
	WriteMode m_write_mode;
	std::vector<BandwidthPerfClientThread*> m_client_threads;
	std::vector<BandwidthPerfServerThread*> m_server_threads;
	int64_t m_elapsedWriteMs;
	int64_t m_elapsedReadMs;
	int64_t m_elapsedSendMs;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;

	std::string getTestParameters(bool forCSV);
	void makeThreadsReady(TestOperation testOperation);
	void runThreads();
};



}
#endif