#include "AbstractMainMemory.h"
#include <stdlib.h>
#include <string.h>

using namespace rdma;

// constructors
AbstractMainMemory::AbstractMainMemory(size_t mem_size) : AbstractBaseMemory(mem_size){}
AbstractMainMemory::AbstractMainMemory(void* buffer, size_t mem_size) : AbstractBaseMemory(buffer, mem_size){}

void AbstractMainMemory::openContext(){

}

void AbstractMainMemory::closeContext(){

}

inline void AbstractMainMemory::setMemory(int value){
    setMemory(value, 0, this->mem_size);
}

inline void AbstractMainMemory::setMemory(int value, size_t num){
    setMemory(value, 0, num);
}

inline void AbstractMainMemory::setMemory(int value, size_t offset, size_t num){
    memset((void*)((size_t)this->buffer + offset), value, num);
}


void AbstractMainMemory::copyTo(void *destination, MEMORY_TYPE memtype){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size), memtype);
}

void AbstractMainMemory::copyTo(void *destination, size_t num, MEMORY_TYPE memtype){
    copyTo(destination, 0, 0, num, memtype);
}

void AbstractMainMemory::copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num, MEMORY_TYPE memtype){
    destination = (void*)((size_t)destination + destOffset);
    void* source = (void*)((size_t)this->buffer + srcOffset);
    #ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
        if((int)memtype > (int)MEMORY_TYPE::MAIN){
            AbstractBaseMemory::checkCudaError(cudaMemcpy(destination, source, num, cudaMemcpyHostToDevice),
                "AbstractMainMemory::copyTo could not copy data from MAIN TO GPU\n");
            return;
        }
    #endif
    memcpy(destination, source, num);
    
}

void AbstractMainMemory::copyFrom(const void *source, MEMORY_TYPE memtype){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size), memtype);
}

void AbstractMainMemory::copyFrom(const void *source, size_t num, MEMORY_TYPE memtype){
    copyFrom(source, 0, 0, num, memtype);
}

void AbstractMainMemory::copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num, MEMORY_TYPE memtype){
    source = (void*)((size_t)source + srcOffset);
    void* destination = (void*)((size_t)this->buffer + destOffset);
    #ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
        if((int)memtype > (int)MEMORY_TYPE::MAIN){
            AbstractBaseMemory::checkCudaError(cudaMemcpy(destination, source, num, cudaMemcpyDeviceToHost),
                "AbstractMainMemory::copyFrom could not copy data from GPU TO MAIN\n");
            return;
        }
    #endif
    memcpy(destination, source, num);
}


inline char AbstractMainMemory::getChar(size_t offset){
    return ((char*)this->buffer)[offset];
}

inline void AbstractMainMemory::set(char value, size_t offset){
    ((char*)this->buffer)[offset] = value;
}

inline int16_t AbstractMainMemory::getInt16(size_t offset){
    return *(int16_t*)((size_t)this->buffer + offset);
}

inline void AbstractMainMemory::set(int16_t value, size_t offset){
    int16_t *tmp = (int16_t*)((size_t)this->buffer + offset);
    *tmp = value;
}

inline uint16_t AbstractMainMemory::getUInt16(size_t offset){
    return *(uint16_t*)((size_t)this->buffer + offset);
}

inline void AbstractMainMemory::set(uint16_t value, size_t offset){
    uint16_t *tmp = (uint16_t*)((size_t)this->buffer + offset);
    *tmp = value;
}

inline int32_t AbstractMainMemory::getInt32(size_t offset){
    return *(int32_t*)((size_t)this->buffer + offset);
}

inline void AbstractMainMemory::set(int32_t value, size_t offset){
    int32_t *tmp = (int32_t*)((size_t)this->buffer + offset);
    *tmp = value;
}

inline uint32_t AbstractMainMemory::getUInt32(size_t offset){
    return *(uint32_t*)((size_t)this->buffer + offset);
}

inline void AbstractMainMemory::set(uint32_t value, size_t offset){
    uint32_t *tmp = (uint32_t*)((size_t)this->buffer + offset);
    *tmp = value;
}

inline int64_t AbstractMainMemory::getInt64(size_t offset){
    return *(int64_t*)((size_t)this->buffer + offset);
}

inline void AbstractMainMemory::set(int64_t value, size_t offset){
    int64_t *tmp = (int64_t*)((size_t)this->buffer + offset);
    *tmp = value;
}

inline uint64_t AbstractMainMemory::getUInt64(size_t offset){
    return *(int64_t*)((size_t)this->buffer + offset);
}

inline void AbstractMainMemory::set(uint64_t value, size_t offset){
    uint64_t *tmp = (uint64_t*)((size_t)this->buffer + offset);
    *tmp = value;
}