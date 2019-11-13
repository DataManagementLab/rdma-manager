/**
 * @file TestConfig.h
 * @author lthostrup
 * @date 2019-04-16
 */

#include "TestConfig.h"

using namespace rdma;

TEST_F(TestConfig, loadConfigFile) {
  static Config conf(program_name + "/");

  ASSERT_TRUE(Config::RDMA_NUMAREGION == 4);
  ASSERT_TRUE(Config::RDMA_DEVICE == 5);
  ASSERT_TRUE(Config::RDMA_IBPORT == 6);
  ASSERT_TRUE(Config::RDMA_PORT == 1234);
  ASSERT_TRUE(Config::LOGGING_LEVEL == 1);
  ASSERT_TRUE(Config::RDMA_MEMSIZE == 1234567);
  int threadcpu = 10;
  for (auto &s : Config::THREAD_CPUS) {
    ASSERT_TRUE(s == threadcpu);
    threadcpu++;
  }
}