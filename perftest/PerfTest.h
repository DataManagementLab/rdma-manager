/*
 * PerfTest.h
 *
 *  Created on: 28.11.2015
 *      Author: cbinnig
 */

#ifndef PERFTEST_H_
#define PERFTEST_H_

#include "../utils/Config.h"

#include <getopt.h>
#include <cmath>

namespace rdma {

struct config_t {
	size_t number = 0;
	string server = "";
	size_t port = 5200;
	size_t iter = 10000000;
	size_t threads = 1;
	size_t data = 2048;
	string logfile;
	std::size_t returnMethod = 0;
	bool old = false;
	bool signaled = false;

};

class PerfTest {
public:
	static config_t parseParameters(int argc, char* argv[]) {
		// parse parameters
		struct config_t config;
		while (true) {
			struct option long_options[] = {
			        { "number", required_argument, nullptr,'n' },
			        { "server", optional_argument, nullptr, 's' },
			        { "port",optional_argument, nullptr, 'p' },
			        { "data", optional_argument, nullptr,'d' },
			        { "threads", optional_argument, nullptr, 't' },
			        { "logfile", optional_argument, nullptr, 'f' },
					{ "iterations", optional_argument, nullptr, 'i' },
                    { "return", optional_argument, nullptr, 'r' },
                    { "old", no_argument, nullptr, 'o' },
                    { "signaled", no_argument, nullptr, 'g' },
			};

			int c = getopt_long(argc, argv, "n:d:s:t:p:f:i:r:og", long_options, nullptr);
			if (c == -1)
				break;

			switch (c) {
			case 'n':
				config.number = strtoul(optarg, nullptr, 0);
				break;
			case 'd':
				config.data = strtoul(optarg, nullptr, 0);
				break;
			case 's':
				config.server = string(optarg);
				break;
			case 'p':
				config.port = strtoul(optarg, nullptr, 0);
				break;
			case 't':
				config.threads = strtoul(optarg, nullptr, 0);
				break;
			case 'i':
				config.iter = strtoul(optarg, nullptr, 0);
				break;
			case 'f':
				config.logfile = string(optarg);
				break;
            case 'r':
                config.returnMethod = strtoul(optarg, nullptr,0);
                break;
            case 'o':
                config.old = true;
                break;
            case 'g':
                config.signaled = true;
                break;
            default:
                break;
			}
		}
		return config;
	}

	virtual ~PerfTest() = default;

	virtual void runClient()=0;
	virtual void runServer()=0;
	virtual double time()=0;

	virtual void printHeader()=0;
	virtual void printResults()=0;
	virtual void printUsage()=0;

	void isClient(bool isClient) {
		m_isClient = isClient;
	}

	void isRunnable(bool isRunnable) {
		m_isRunnable = isRunnable;
	}

	bool isClient() const {
		return m_isClient;
	}

	bool isRunnable() const {
		return m_isRunnable;
	}

	static void waitForUser() {
		//wait for user input
		cout << "Press Enter to run Benchmark!" << flush << endl;
		char temp;
		cin.get(temp);
	}

private:
	bool m_isClient = false;
	bool m_isRunnable = false;
};

}

#endif /* PERFTEST_H_ */
