#pragma once

#include <iostream>
#include <memory>
#include <fstream>

#include "../src/utils/Config.h"
#include "../src/rdma/RDMAClient.h"
#include "../src/rdma/UnreliableRDMA.h"

struct Msg {
    size_t src;
    size_t seq_no;
};

//Tests the order of received multicast messages (spoiler: its not ordered...)
class MulticastOrderTest
{
    using RDMAClient = rdma::RDMAClient<rdma::UnreliableRDMA>;
public:
    MulticastOrderTest(int node);
    ~MulticastOrderTest();
    

    void run();
    

private:
    void write_to_file();

    std::unique_ptr<RDMAClient> m_rdmaClient;
    std::string m_mCastAddr = "172.18.94.10";

    NodeID m_clientMCastID;

    std::unique_ptr<Msg*[]> recv_msgs;
    std::unique_ptr<Msg*[]> send_msgs;

    int node;

    std::vector<Msg> msgs;
};

//Compares output files. Usage python cmp.py msgs_0.txt msgs_1.txt ...
/*
#!/usr/bin/env python3
import sys
​
files = map(open, sys.argv[1:])
​
for i, lines in enumerate(zip(*files)):
    if len(set(lines)) != 1:
        print(f'unequal on line: {i:10}', end=' '*10+'| ')
        line = []
        for j, l in enumerate(lines):
            line.append(f'File {j}: -> {l.rstrip():10}')
        print('\t'.join(line))
*/