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
#include <math.h>

namespace istore2 {

struct config_t {
	size_t number = 0;
	string server = "";
	size_t port = 5200;
	size_t iter = 10000000;
	size_t threads = 1;
	size_t data = 2048;
};

class PerfTest {
public:
	static config_t parseParameters(int argc, char* argv[]) {
		// parse parameters
		struct config_t config;
		while (1) {
			struct option long_options[] = { { "number", required_argument, 0,
					'n' }, { "server", optional_argument, 0, 's' }, { "port",
					optional_argument, 0, 'p' }, { "data", optional_argument, 0,
					'd' }, { "threads", optional_argument, 0, 't' } };

			int c = getopt_long(argc, argv, "n:d:s:t:p:", long_options, NULL);
			if (c == -1)
				break;

			switch (c) {
			case 'n':
				config.number = strtoul(optarg, NULL, 0);
				break;
			case 'd':
				config.data = strtoul(optarg, NULL, 0);
				break;
			case 's':
				config.server = string(optarg);
				break;
			case 'p':
				config.port = strtoul(optarg, NULL, 0);
				break;
			case 't':
				config.threads = strtoul(optarg, NULL, 0);
				break;
			}
		}
		return config;
	}

	virtual ~PerfTest() {
	}

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

	bool isClient() {
		return m_isClient;
	}

	bool isRunnable() {
		return m_isRunnable;
	}

	void waitForUser() {
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
