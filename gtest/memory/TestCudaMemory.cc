#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#include "TestCudaMemory.h"
#include <string.h>
#include <cuda_runtime_api.h>

static const size_t MEMORY_SIZE = 1024 * 1024;

void TestCudaMemory::SetUp() {

}

void test(int device_index, CudaMemory *mem){
    ASSERT_EQ(MEMORY_SIZE, mem->getSize());
    ASSERT_EQ(device_index, mem->getDeviceIndex());
    ASSERT_TRUE(mem);
    ASSERT_TRUE(mem->pointer());

    char msg[] = { "Hello World" };
    cudaMemcpy(mem->pointer(), (void*)msg, sizeof(msg), cudaMemcpyHostToDevice);
    char* check = new char[sizeof(msg)];
    cudaMemcpy((void*)check, mem->pointer(), sizeof(msg), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(strcmp(msg, check) == 0);

    mem->setMemory(0);
    cudaMemcpy((void*)check, mem->pointer(), sizeof(check), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(strlen(check)==0);

    mem->copyFrom(msg);
    mem->copyTo(check);
    ASSERT_TRUE(strcmp(msg, check) == 0);

    char value = 8, offset = 5;
    mem->set(value, offset);
    ASSERT_TRUE(mem->getChar(offset) == value);
}

TEST_F(TestCudaMemory, testMemory) {
    int device_index = 0;
    CudaMemory *mem = new CudaMemory(MEMORY_SIZE, device_index);
    test(device_index, mem);
    delete mem;
}

#endif /* CUDA support */