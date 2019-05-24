/**
 * @file RDMAManagerUD.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef RDMAMANAGERUD_H_
#define RDMAMANAGERUD_H_

#include "../utils/Config.h"

#include "./RDMAManager.h"

#include <rdma/rdma_verbs.h>
#include <arpa/inet.h>

namespace rdma {

struct rdma_mcast_conn_t {
    char* mcast_addr;
    struct sockaddr mcast_sockaddr;
    struct rdma_cm_id *id;
    struct rdma_event_channel *channel;
    struct ibv_cq *scq;
    struct ibv_cq *rcq;
    struct ibv_ah *ah;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    uint32_t remote_qpn;
    uint32_t remote_qkey;
    pthread_t cm_thread;
};

class RDMAManagerUD : public RDMAManager {

 public:
    RDMAManagerUD(size_t mem_size = Config::RDMA_MEMSIZE);
    ~RDMAManagerUD();

    bool initQP(struct ib_addr_t& retIbAddr, bool isMgmtQP = false);
    bool connectQP(struct ib_addr_t& ibAddr);

    bool remoteWrite(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr, size_t size,
                     bool signaled);
    bool remoteRead(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr, size_t size,
                    bool signaled);
    bool requestRead(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr, size_t size);
    bool remoteFetchAndAdd(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                           size_t size, bool signaled);
    bool remoteFetchAndAdd(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr, size_t value_to_add,
                                           size_t size, bool signaled);

    bool remoteCompareAndSwap(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                              int toCompare, int toSwap, size_t size, bool signaled);

    bool send(struct ib_addr_t& ibAddr, const void* memAddr, size_t size, bool signaled);
    bool receive(struct ib_addr_t& ibAddr, const void* memAddr, size_t size);
    bool pollReceive(ib_addr_t& retIbAddr, bool doPoll);
    bool pollSend(ib_addr_t& ibAddr, bool doPoll);

    void* localAlloc(const size_t& size);
    bool localFree(const void* ptr);
    bool localFree(const size_t& offset);
    rdma_mem_t remoteAlloc(const size_t& size);
    bool remoteFree(const size_t& offset);

    bool joinMCastGroup(string mCastAddress, struct ib_addr_t& retIbAddr);
    bool leaveMCastGroup(struct ib_addr_t ibAddr);
    bool sendMCast(struct ib_addr_t ibAddr, const void* memAddr, size_t size, bool signaled);
    bool receiveMCast(struct ib_addr_t ibAddr, const void* memAddr, size_t size);
    bool pollReceiveMCast(struct ib_addr_t ibAddr);

 private:
    bool createQP(struct ib_qp_t *qp);
    void destroyQPs();
    bool modifyQPToInit(struct ibv_qp *qp);
    bool modifyQPToRTR(struct ibv_qp *qp);
    bool modifyQPToRTS(struct ibv_qp *qp, const uint32_t psn);

    inline uint64_t nextMCastConnKey() {
        return m_lastMCastConnKey++;
    }

    inline void setMCastConn(ib_addr_t& ibAddr, rdma_mcast_conn_t& conn) {
        size_t connKey = ibAddr.conn_key;
        if (m_udpMcastConns.size() < connKey + 1) {
            m_udpMcastConns.resize(connKey + 1);
        }
        m_udpMcastConns[connKey] = conn;
    }

    inline bool getCmEvent(struct rdma_event_channel *channel, enum rdma_cm_event_type type,
                           struct rdma_cm_event **out_ev) {
        struct rdma_cm_event *event = NULL;
        if (rdma_get_cm_event(channel, &event) != 0) {
            return false;
        }
        /* Verify the event is the expected type */
        if (event->event != type) {
            return false;
        }
        /* Pass the event back to the user if requested */
        if (!out_ev) {
            rdma_ack_cm_event(event);
        } else {
            *out_ev = event;
        }
        return true;
    }

    //only one QP needed for all connections
    ib_qp_t m_udqp;
    ib_conn_t m_udqpConn;
    ib_qp_t m_udqpMgmt;
    ib_conn_t m_udqpMgmtConn;

    //maps mcastConnkey to MCast connections
    size_t m_lastMCastConnKey;
    vector<rdma_mcast_conn_t> m_udpMcastConns;
};

}

#endif /* RDMAMANAGERUD_H_ */
