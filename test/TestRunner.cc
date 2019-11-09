
#include <cppunit/CompilerOutputter.h>
#include <cppunit/ui/text/TestRunner.h>
#include <iostream>

// we cannot use the TestRegistryFactory because of this problem:
//http://stackoverflow.com/questions/1016197/linking-in-test-libraries-with-cppunit
//include the test classes
#include "Tests.h"

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;

  string prog_name = string(argv[0]);
  static Config conf(prog_name);

  // Adds the test to the list of test to run
  CppUnit::TextUi::TestRunner runner;
  // runner.addTest(TestThread::suite());
  // runner.addTest(TestProtoServer::suite());
  runner.addTest(TestRDMAServer::suite());
  // runner.addTest(TestRDMAServerMCast::suite());
  // runner.addTest(TestRDMAServerMultClients::suite());
  // runner.addTest(TestRDMAServerSRQ::suite());
  // runner.addTest(TestSimpleUD::suite());
  // runner.addTest(TestConfig::suite());
  // runner.addTest(TestRPC::suite());

  // Change the default outputter to a compiler error format outputter
  runner.setOutputter(
      new CppUnit::CompilerOutputter(&runner.result(), std::cerr));
  // Run the tests.
  bool wasSucessful = runner.run();

  // Return error code 1 if the one of test failed.
  return wasSucessful ? 0 : 1;
}
