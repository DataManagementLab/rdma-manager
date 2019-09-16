/**
 * @file TestConfig.h
 * @author lthostrup
 * @date 2019-04-16
 */

#pragma once

#include "../../src/utils/Config.h"

using namespace rdma;

class TestConfig : public CppUnit::TestFixture
{
  RDMA_UNIT_TEST_SUITE(TestConfig);
  RDMA_UNIT_TEST(loadConfigFile);
  RDMA_UNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();
  void loadConfigFile();

private:
  string program_name = "testconfig";
};