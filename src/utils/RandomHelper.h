#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <functional>
#include <ctime>

namespace rdma {

class RandomHelper {
public:

    inline static uint8_t randomUInt8(){
        std::srand ( unsigned(std::time(0)) );
        return (uint8_t)(std::rand() % 256);
    }

    inline static void randomizeMainMemory(char* arr, const size_t &offset, const size_t &byte_size){
        std::srand ( unsigned(std::time(0)) );
        arr = (char*)((size_t)arr + offset);
        for(size_t i=0; i<byte_size; i++){
            *arr = (uint8_t)(std::rand() % 256);
            arr = (char*)((size_t)arr + sizeof(uint8_t));
        }
    }

    inline static std::vector<uint8_t> generateRandomVector(const size_t &bytes){
        std::vector<uint8_t> data(bytes);
        std::random_device rd;
        std::seed_seq seed = {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint8_t> dist;
        std::generate(data.begin(), data.end(), std::bind(dist, gen));
        return data;
    }
};

}