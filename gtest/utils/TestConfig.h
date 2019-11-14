/**
 * @file TestConfig.h
 * @author lthostrup
 * @date 2019-04-16
 */
#pragma once
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

#include <gtest/gtest.h>
#include "../../src/utils/Config.h"

using namespace rdma;

class TestConfig : public testing::Test {
 protected:
  void SetUp() override {
    // Create new test config file
    if (mkdir(program_name.c_str(), 0777) == -1) {
      std::cerr << "Could not create test directory!" << '\n';
      std::cerr << "Error:  " << strerror(errno) << std::endl;
    }
    string subdir = program_name + "/conf";
    if (mkdir(subdir.c_str(), 0777) == -1) {
      std::cerr << "Could not create test directory!" << '\n';
      std::cerr << "Error:  " << strerror(errno) << std::endl;
    }

    std::ofstream confFile(program_name + "/conf/RDMA.conf",
                           std::ofstream::out);

    confFile << " \n\
    RDMA_NUMAREGION = 4 \n\
    RDMA_DEVICE = 5 \n\
    RDMA_IBPORT = 6 \n\
    RDMA_PORT = 1234 \n\
    LOGGING_LEVEL=1 \n\
    RDMA_MEMSIZE = 1234567 \n\
    THREAD_CPUS = 10,11,12,13,   14,15,16, 17,18,19 \n\
  " << std::endl;

    

    confFile.close();
  };

  void TearDown() override {
    // REMOVE FILE AND FOLDERS
    string filename = program_name + "/conf/RDMA.conf";
    std::remove(filename.c_str());
    string subfolder = program_name + "/conf";
    rmdir(subfolder.c_str());
    rmdir(program_name.c_str());
  }

  //   void loadConfigFile();

  string program_name = "testconfig";
};