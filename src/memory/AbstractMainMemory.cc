#include "AbstractMainMemory.h"
#include <stdlib.h>
#include <string.h>

using namespace rdma;

// constructors
AbstractMainMemory::AbstractMainMemory(size_t mem_size) : AbstractBaseMemory(mem_size){}
AbstractMainMemory::AbstractMainMemory(void* buffer, size_t mem_size) : AbstractBaseMemory(buffer, mem_size){}

inline void AbstractMainMemory::setMemory(int value){
    setMemory(value, 0, this->mem_size);
}

inline void AbstractMainMemory::setMemory(int value, size_t num){
    setMemory(value, 0, num);
}

inline void AbstractMainMemory::setMemory(int value, size_t offset, size_t num){
    memset((void*)((size_t)this->buffer + offset), value, num);
}

void AbstractMainMemory::copyTo(void *destination){
    size_t s = sizeof(destination);
    copyTo(destination, (s < this->mem_size ? s : this->mem_size));
}

void AbstractMainMemory::copyTo(void *destination, size_t num){
    copyTo(destination, 0, 0, num);
}

void AbstractMainMemory::copyTo(void *destination, size_t destOffset, size_t srcOffset, size_t num){
    memcpy((void*)((size_t)destination + destOffset), (void*)((size_t)this->buffer + srcOffset), num);
}

void AbstractMainMemory::copyFrom(const void *source){
    size_t s = sizeof(source);
    copyFrom(source, (s < this->mem_size ? s : this->mem_size));
}

void AbstractMainMemory::copyFrom(const void *source, size_t num){
    copyFrom(source, 0, 0, num);
}

void AbstractMainMemory::copyFrom(const void *source, size_t srcOffset, size_t destOffset, size_t num){
    memcpy((void*)((size_t)this->buffer + destOffset), (void*)((size_t)source + srcOffset), num);
}

inline char AbstractMainMemory::getChar(size_t offset){
    return ((char*)this->buffer)[offset];
}

inline void AbstractMainMemory::set(char value, size_t offset){
    ((char*)this->buffer)[offset] = value;
}

inline int16_t AbstractMainMemory::getInt16(size_t offset){
    return ((int16_t*)this->buffer)[offset];
}

inline void AbstractMainMemory::set(int16_t value, size_t offset){
    ((int16_t*)this->buffer)[offset] = value;
}

inline int32_t AbstractMainMemory::getInt32(size_t offset){
    return ((int32_t*)this->buffer)[offset];
}

inline void AbstractMainMemory::set(int32_t value, size_t offset){
    ((int32_t*)this->buffer)[offset] = value;
}

inline int64_t AbstractMainMemory::getInt64(size_t offset){
    return ((int64_t*)this->buffer)[offset];
}

inline void AbstractMainMemory::set(int64_t value, size_t offset){
    ((int64_t*)this->buffer)[offset] = value;
}