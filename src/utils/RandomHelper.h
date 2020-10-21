#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <functional>

namespace rdma {

class RandomHelper {
public:

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