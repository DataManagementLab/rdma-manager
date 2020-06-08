#include "MulticastOrderTest.h"


MulticastOrderTest::MulticastOrderTest(int node) : node(node)
{
    const size_t buffer_size = rdma::Config::RDMA_MAX_WR * (sizeof(Msg) + rdma::Config::RDMA_UD_OFFSET);
    std::cout << "Creating unreliable RDMA client" << std::endl;
    m_rdmaClient = std::make_unique<RDMAClient>(buffer_size * 2 *1.1);

    send_msgs = std::make_unique<Msg*[]>(rdma::Config::RDMA_MAX_WR);
    recv_msgs = std::make_unique<Msg*[]>(rdma::Config::RDMA_MAX_WR);

    m_rdmaClient->joinMCastGroup(m_mCastAddr, m_clientMCastID);
    std::cout << "Joined multicast group: " << m_mCastAddr << std::endl;
    for (size_t i = 0; i < rdma::Config::RDMA_MAX_WR; ++i) {
        send_msgs[i] = reinterpret_cast<Msg*>(m_rdmaClient->localAlloc(sizeof(Msg)));
        send_msgs[i]->src = node;
        recv_msgs[i] = reinterpret_cast<Msg*>(m_rdmaClient->localAlloc(sizeof(Msg)));

        m_rdmaClient->receiveMCast(m_clientMCastID, reinterpret_cast<void*>(recv_msgs[i]), sizeof(Msg));
    }
    std::cout << "Registered multicast receives" << std::endl;
}


MulticastOrderTest::~MulticastOrderTest() {}

void MulticastOrderTest::run()
{

    constexpr size_t num_msgs = 1'000'000'0;
    size_t current_seq_no = 0;
    size_t current_recv_idx = 0;
    msgs.reserve(num_msgs);

    while(true) {
        for (size_t i = 0; i < rdma::Config::RDMA_MAX_WR; ++i)
        {
            send_msgs[i]->seq_no = ++current_seq_no;
            // std::cout << "sent " << current_seq_no << std::endl;
            m_rdmaClient->sendMCast(m_clientMCastID, reinterpret_cast<void*>(send_msgs[i]), sizeof(Msg), false);
    
            while (m_rdmaClient->pollReceiveMCast(m_clientMCastID, false) > 0)
            {
                Msg &msg = *recv_msgs[current_recv_idx];
                // std::cout << "received " << msg.seq_no << std::endl;
                msgs.push_back(msg);
                if (msgs.size() == num_msgs) {
                    write_to_file();
                }

                m_rdmaClient->receiveMCast(m_clientMCastID, reinterpret_cast<void*>(recv_msgs[current_recv_idx]), sizeof(Msg));
                current_recv_idx = ((current_recv_idx + 1) == rdma::Config::RDMA_MAX_WR) ? 0 : current_recv_idx + 1;
            }
        }
    }
}

void MulticastOrderTest::write_to_file() 
{
    std::cout << "Dumping to file..." << std::endl;
    std::ofstream file("msgs_" + to_string(node) + ".txt");
    for (auto &msg : msgs) {
        file << msg.src << ' ' << msg.seq_no << '\n';
    }
    file << std::flush;

    exit(0);
}

int main(int, char const *argv[])
{
    int id = atoi(argv[1]);
    MulticastOrderTest test(id);

    //wait for user input
    do {
        std::cout << '\n' << "Press ENTER to continue...";
    } while (std::cin.get() != '\n');

    test.run();
    
    return 0;
}
