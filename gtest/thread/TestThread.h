/**
 * @file TestThread.h
 * @author cbinnig, lthostrup, tziegler
 * @date 2018-08-17
 */

#pragma once

#include <gtest/gtest.h>
#include "../../src/thread/Thread.h"
#include "../../src/utils/Config.h"

using namespace rdma;

class TestThread : public testing::Test {
 protected:

  class TestingThread : public Thread {
   public:
    bool runned = false;
    void run() { runned = true; }
  };
  TestingThread *m_testingThread;

  void SetUp() override;
  void TearDown() override;
};
