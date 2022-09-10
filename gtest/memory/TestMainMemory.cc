#include "TestMainMemory.h"
#include <string.h>

static const size_t MEMORY_SIZE = 1024 * 1024;

void TestMainMemory::SetUp() {

}

void test(bool huge, Memory *mem){
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

    mem->copyFrom(msg, MEMORY_TYPE::MAIN);
    mem->copyTo(check, MEMORY_TYPE::MAIN);
    ASSERT_TRUE(strcmp(msg, check) == 0);

    char value = 8, offset = 5;
    mem->set(value, offset);
    ASSERT_TRUE(mem->getChar(offset) == value);
}

TEST_F(TestMainMemory, testNormalSize) {
    Memory *mem = new Memory(MEMORY_SIZE, false);
    test(false, mem);
    delete mem;
}

TEST_F(TestMainMemory, testHugeSize) {
    Memory *mem = new Memory(MEMORY_SIZE, true);
    test(true, mem);
    delete mem;
}
