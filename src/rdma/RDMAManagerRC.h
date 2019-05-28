/**
 * @file RDMAManagerRC.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef RDMAMANAGERRC_H_
#define RDMAMANAGERRC_H_

#include "../utils/Config.h"

#include "RDMAManager.h"

namespace rdma {


struct sharedrq_t{
    ibv_srq* shared_rq;
    ibv_cq* recv_cq;
};


class RDMAManagerRC : public RDMAManager {

 public:
    RDMAManagerRC(size_t mem_size = Config::RDMA_MEMSIZE);
    ~RDMAManagerRC();

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
    bool pollReceive(ib_addr_t& ibAddr, bool doPoll);
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

    //Shared Receive Queue
    vector<ib_addr_t> getIbAddrs(size_t srq_id);
    bool initQP(size_t srq_id, struct ib_addr_t& reIbAddr) override;

    bool receive(size_t srq_id, const void* memAddr,
                 size_t size) override;
    bool pollReceive(size_t srq_id, ib_addr_t& ret_ibaddr, bool doPoll) override;
    bool createSharedReceiveQueue(size_t& ret_srq_id) override;

 protected:
    // RDMA operations
    inline bool __attribute__((always_inline)) remoteAccess(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                             size_t size, bool signaled, bool wait, enum ibv_wr_opcode verb) {
            
        uint64_t connKey = ibAddr.conn_key;
        struct ib_qp_t localQP = m_qps[connKey];
        struct ib_conn_t remoteConn = m_rconns[connKey];

        int ne;

        struct ibv_send_wr sr;
        struct ibv_sge sge;
        memset(&sge, 0, sizeof(sge));
        sge.addr = (uintptr_t) memAddr;
        sge.lkey = m_res.mr->lkey;
        sge.length = size;
        memset(&sr, 0, sizeof(sr));
        sr.sg_list = &sge;
        sr.num_sge = 1;
        sr.opcode = verb;
        sr.next = nullptr;
        sr.send_flags = (signaled) ? IBV_SEND_SIGNALED : 0;

        //calculate remote address using offset in local buffer
        sr.wr.rdma.remote_addr = remoteConn.buffer + offset;
        sr.wr.rdma.rkey = remoteConn.rc.rkey;

        struct ibv_send_wr *bad_wr = nullptr;
        if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
            Logging::error(__FILE__, __LINE__, "RDMA OP not successful! error: " + to_string(errno));
            return false;
        }

        if (signaled && wait) {
            struct ibv_wc wc;

            do {
                wc.status = IBV_WC_SUCCESS;
                ne = ibv_poll_cq(localQP.send_cq, 1, &wc);
                if (wc.status != IBV_WC_SUCCESS) {
                    Logging::error(
                            __FILE__,
                            __LINE__,
                            "RDMA completion event in CQ with error! " + to_string(wc.status));
                    return false;
                }
            } while (ne == 0);

            if (ne < 0) {
                Logging::error(__FILE__, __LINE__, "RDMA polling from CQ failed!");
                return false;
            }
        }

        return true;
    }

    virtual void destroyQPs();
    bool createQP(struct ib_qp_t *qp);
    bool createQP(size_t srq_id, struct ib_qp_t& qp);
    bool modifyQPToInit(struct ibv_qp *qp);
    bool modifyQPToRTR(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, uint8_t *dgid);
    bool modifyQPToRTS(struct ibv_qp *qp);


    //shared receive queues
    map<size_t,sharedrq_t> m_srqs;
    size_t m_srqCounter = 0;
    map<size_t,vector<ib_addr_t>> m_connectedQPs;

};

}

#endif /* RDMAMANAGERRC_H_ */
