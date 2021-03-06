#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include "Config.h"

#include <string>
#include <string_view>

namespace rdma {
    
enum class RdmaCounterTypes
{
    XMIT_DATA, RCV_DATA, //MULTICAST_XMIT_DATA, MULTICAST_RCV_DATA//, XMIT_PACKETS, RCV_PACKETS, MULTICAST_XMIT_PACKETS, MULTICAST_RCV_PACKETS, 
};
struct RdmaCounterOptions {
    std::vector<RdmaCounterTypes> counter_types;
    uint16_t port = 1;
    uint16_t infiniband_lanes = 4;
    std::string rdma_device_path = Config::RDMA_DEVICE_FILE_PATH;
    std::string csvOutput = "";
    std::string tag = "";
};

class RdmaCounter {
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    std::chrono::time_point<std::chrono::steady_clock> stopTime;
    std::map<RdmaCounterTypes, size_t> start_values;

    RdmaCounterOptions options;
    bool scoped = false;
    
    std::ofstream csv;

    const std::map<RdmaCounterTypes, std::string> counter_type_map{
        {RdmaCounterTypes::XMIT_DATA, "port_xmit_data"},
        {RdmaCounterTypes::RCV_DATA, "port_rcv_data"},
        // {RdmaCounterTypes::MULTICAST_XMIT_DATA, "multicast_xmit_data"},
        // {RdmaCounterTypes::MULTICAST_RCV_DATA, "multicast_rcv_data"},
        // {RdmaCounterTypes::XMIT_PACKETS, "port_xmit_packets"},
        // {RdmaCounterTypes::RCV_PACKETS, "port_rcv_packets"},
        // {RdmaCounterTypes::MULTICAST_XMIT_PACKETS, "multicast_xmit_packets"},
        // {RdmaCounterTypes::MULTICAST_RCV_PACKETS, "multicast_rcv_packets"},
    };
    
public:
    RdmaCounter(RdmaCounterOptions options, bool scoped = false) : options(options), scoped(scoped) {

        if (scoped)
            start();
    }

    ~RdmaCounter() {
        if (scoped)
            stop();
    }

    void start() {
        for (auto counter_type : options.counter_types)
        {
            std::ifstream counter_file;
            counter_file.open(options.rdma_device_path+"/ports/"+to_string(options.port)+"/counters/"+this->counter_type_map.at(counter_type));
            counter_file >> start_values[counter_type];
        }
        startTime = std::chrono::steady_clock::now();
    }

    void stop() {
        stopTime = std::chrono::steady_clock::now();
        bool header = false;
        if (!options.csvOutput.empty())
        {
            csv = std::ofstream(options.csvOutput, std::ios_base::app | std::ios_base::out);
            if (!Filehelper::fileExists(options.csvOutput))
                header = true;
        }
        
        //Header
        if (header)
        {
            csv << "duration_s";
        }
        
        std::cout << "Elapsed time(s)";
        for (auto counter_type : options.counter_types)
        {
            if (header)
            {
                csv << "," << this->counter_type_map.at(counter_type) << "_MiB";
                csv << "," << this->counter_type_map.at(counter_type) << "_MiB/s";
            }
            std::cout << "\t" << this->counter_type_map.at(counter_type) << " MiB";
            std::cout << "\t" << this->counter_type_map.at(counter_type) << " MiB/s";
        }
        if (!options.tag.empty())
        {
            if (header)
            {
                csv << ",tag";
            }
            std::cout << "\ttag";
        }
        
        // Values
        if (!options.csvOutput.empty())
        {
            csv << "\n";
            csv << getDuration();
        }   

        std::cout << "\n";
        std::cout << getDuration();
        
        for (auto counter_type : options.counter_types)
        {
            std::ifstream counter_file;
            counter_file.open(options.rdma_device_path+"/ports/"+to_string(options.port)+"/counters/"+this->counter_type_map.at(counter_type));
            size_t end_value;
            counter_file >> end_value;
            double diff = ((end_value - start_values[counter_type]) * options.infiniband_lanes * 1.0) / 1024/1024;
            
            if (!options.csvOutput.empty())
            {
                csv << "," << diff;
                csv << "," << diff/getDuration();
            }

            std::cout << "\t" << diff;
            std::cout << "\t" << diff/getDuration();
        }

        if (!options.tag.empty())
        {
            if (!options.csvOutput.empty())
            {
                csv << "," << this->options.tag;
            }
            std::cout << "\t" << this->options.tag;
        }
        csv << "\n";
        std::cout << std::endl;
    }

    double getDuration() {
        return std::chrono::duration<double>(stopTime - startTime).count();
    }

};

}