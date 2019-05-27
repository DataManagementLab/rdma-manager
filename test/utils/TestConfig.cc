#include "TestConfig.h"
#include <bits/stdc++.h> 
#include <iostream> 
#include <sys/stat.h> 
#include <sys/types.h> 

void TestConfig::setUp()
{
  //Create new test config file
  if (mkdir(program_name.c_str(), 0777) == -1)
  {
    std::cerr << "Could not create test directory!" << '\n';
    std::cerr << "Error:  " << strerror(errno) << std::endl;   
  }  
  string subdir = program_name + "/conf";
  if (mkdir(subdir.c_str(), 0777) == -1)
  {
    std::cerr << "Could not create test directory!" << '\n';
    std::cerr << "Error:  " << strerror(errno) << std::endl;   
  }  
  
  std::ofstream confFile (program_name + "/conf/RDMA.conf", std::ofstream::out);

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
}

void TestConfig::tearDown()
{
  // REMOVE FILE AND FOLDERS
  string filename = program_name + "/conf/RDMA.conf";
  std::remove(filename.c_str());
  string subfolder = program_name + "/conf";
  rmdir(subfolder.c_str());
  rmdir(program_name.c_str());

  //Reload the normal test config
  static Config conf("");
}


void TestConfig::loadConfigFile()
{
  static Config conf(program_name+"/");
  
  CPPUNIT_ASSERT_MESSAGE("RDMA_NUMAREGION", Config::RDMA_NUMAREGION == 4);
  CPPUNIT_ASSERT_MESSAGE("RDMA_DEVICE", Config::RDMA_DEVICE == 5);
  CPPUNIT_ASSERT_MESSAGE("RDMA_IBPORT", Config::RDMA_IBPORT == 6);
  CPPUNIT_ASSERT_MESSAGE("RDMA_PORT", Config::RDMA_PORT == 1234);
  CPPUNIT_ASSERT_MESSAGE("LOGGING_LEVEL", Config::LOGGING_LEVEL == 1);
  CPPUNIT_ASSERT_MESSAGE("RDMA_MEMSIZE", Config::RDMA_MEMSIZE == 1234567);
  int threadcpu = 10;
  for(auto &s : Config::THREAD_CPUS)
  {
    CPPUNIT_ASSERT_MESSAGE("THREAD_CPUS", s == threadcpu);
    threadcpu++;
  }

}

