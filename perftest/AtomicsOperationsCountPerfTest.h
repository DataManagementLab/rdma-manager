#ifndef AtomicsOperationsCountPerfTest_H
#define AtomicsOperationsCountPerfTest_H

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

class AtomicsOperationsCountPerfClientThread : public Thread {
public:
	AtomicsOperationsCountPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, int buffer_slots, size_t iterations_per_thread);
	~AtomicsOperationsCountPerfClientThread();
	void run();
	bool ready() {
		return m_ready;
	}

	int64_t m_elapsedFetchAdd = -1;
	int64_t m_elapsedCompareSwap = -1;

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> *m_client;
	LocalBaseMemoryStub *m_local_memory;
	size_t m_memory_per_thread;
	int m_buffer_slots;
	size_t m_iterations_per_thread;
	std::vector<std::string> m_rdma_addresses;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};


class AtomicsOperationsCountPerfTest : public rdma::PerfTest {
public:
	AtomicsOperationsCountPerfTest(int testOperations, bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, int buffer_slots, uint64_t iterations_per_thread);
	virtual ~AtomicsOperationsCountPerfTest();
	std::string getTestParameters();
	void setupTest();
	void runTest();
	std::string getTestResults(std::string csvFileName="", bool csvAddHeader=true);

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
	uint64_t m_memory_size;
	int m_buffer_slots;
	uint64_t m_iterations_per_thread;
	std::vector<AtomicsOperationsCountPerfClientThread*> m_client_threads;
	int64_t m_elapsedFetchAdd;
	int64_t m_elapsedCompareSwap;

	BaseMemory *m_memory;
	RDMAServer<ReliableRDMA>* m_server;

	std::string getTestParameters(bool forCSV);
	void makeThreadsReady(TestOperation testOperation);
	void runThreads();
};


}
#endif