#include "TestMainMemory.h"
#include <string.h>

static const size_t MEMORY_SIZE = 1024 * 1024;

void TestMainMemory::SetUp() {

}

void test(bool huge, MainMemory *mem){
    ASSERT_EQ(MEMORY_SIZE, mem->getSize());
    ASSERT_EQ(huge, mem->isHuge());
    ASSERT_TRUE(mem);
    ASSERT_TRUE(mem->pointer());

    const char msg[] = { "Hello World" };
    memcpy(mem->pointer(), (void*)msg, sizeof(msg));
    char* check = new char[sizeof(msg)];
    memcpy((void*)check, mem->pointer(), sizeof(msg));
    ASSERT_TRUE(strcmp(msg, check) == 0);

    memset((void*)check, 0, sizeof(check));
    mem->setMemory(0);
    ASSERT_TRUE(strcmp((char*)mem->pointer(), check) == 0);

    mem->copyFrom(msg);
    mem->copyTo(check);
    ASSERT_TRUE(strcmp(msg, check) == 0);
}

TEST_F(TestMainMemory, testNormalSize) {
    MainMemory *mem = new MainMemory(MEMORY_SIZE, false);
    test(false, mem);
    delete mem;
}

TEST_F(TestMainMemory, testHugeSize) {
    MainMemory *mem = new MainMemory(MEMORY_SIZE, true);
    test(true, mem);
    delete mem;
}
