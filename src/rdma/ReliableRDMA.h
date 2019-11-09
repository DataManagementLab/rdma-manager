/**
 * @file ReliableRDMA.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef ReliableRDMA_H_
#define ReliableRDMA_H_

#include "../utils/Config.h"

#include "BaseRDMA.h"

namespace rdma {

struct sharedrq_t {
  ibv_srq* shared_rq;
  ibv_cq* recv_cq;
};

class ReliableRDMA : public BaseRDMA {
 public:
  ReliableRDMA();
  ReliableRDMA(size_t mem_size);
  ~ReliableRDMA();

  bool initQPWithSuppliedID(const rdmaConnID suppliedID) override;
  bool initQP(rdmaConnID& retRdmaConnID) override;
  bool connectQP(const rdmaConnID rdmaConnID) override;

  bool write(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
             size_t size, bool signaled);
  bool read(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
            size_t size, bool signaled);
  bool requestRead(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, size_t size);
  bool fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, size_t size, bool signaled);
  bool fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, size_t value_to_add, size_t size,
                   bool signaled);

  bool compareAndSwap(const rdmaConnID rdmaConnID, size_t offset,
                      const void* memAddr, int toCompare, int toSwap,
                      size_t size, bool signaled);

  bool send(const rdmaConnID rdmaConnID, const void* memAddr, size_t size,
            bool signaled) override;
  bool receive(const rdmaConnID rdmaConnID, const void* memAddr,
               size_t size) override;
  bool pollReceive(const rdmaConnID rdmaConnID, bool doPoll) override;
  bool pollReceiveBatch(size_t srq_id, size_t& num_completed, bool& doPoll);
  bool pollSend(const rdmaConnID rdmaConnID, bool doPoll) override;

  void* localAlloc(const size_t& size) override;
  bool localFree(const void* ptr) override;
  bool localFree(const size_t& offset) override;
  rdma_mem_t remoteAlloc(const size_t& size);
  bool remoteFree(const size_t& offset);

  // Shared Receive Queue
  bool initQPForSRQWithSuppliedID(size_t srq_id, const rdmaConnID rdmaConnID);
  bool initQPForSRQ(size_t srq_id, rdmaConnID& retRdmaConnID);

  bool receiveSRQ(size_t srq_id, const void* memAddr, size_t size);
  bool pollReceiveSRQ(size_t srq_id, rdmaConnID& retrdmaConnID, bool& doPoll);
  bool createSharedReceiveQueue(size_t& ret_srq_id);

 protected:
  // RDMA operations
  inline bool __attribute__((always_inline))
  remoteAccess(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
               size_t size, bool signaled, bool wait, enum ibv_wr_opcode verb) {
    DebugCode(
        if (memAddr < m_res.buffer ||
            (char*)memAddr + size > (char*)m_res.buffer + m_res.mr->length) {
          Logging::error(__FILE__, __LINE__,
                         "Passed memAddr falls out of buffer addr space");
        })

        checkSignaled(signaled, rdmaConnID);

    struct ib_qp_t localQP = m_qps[rdmaConnID];
    struct ib_conn_t remoteConn = m_rconns[rdmaConnID];

    int ne;

    struct ibv_send_wr sr;
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)memAddr;
    sge.lkey = m_res.mr->lkey;
    sge.length = size;
    memset(&sr, 0, sizeof(sr));
    sr.sg_list = &sge;
    sr.num_sge = 1;
    sr.opcode = verb;
    sr.next = nullptr;
    sr.send_flags = (signaled) ? IBV_SEND_SIGNALED : 0;

    // calculate remote address using offset in local buffer
    sr.wr.rdma.remote_addr = remoteConn.buffer + offset;
    sr.wr.rdma.rkey = remoteConn.rc.rkey;

    struct ibv_send_wr* bad_wr = nullptr;
    if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
      Logging::error(__FILE__, __LINE__,
                     "RDMA OP not successful! error: " + to_string(errno));
      return false;
    }

    if (signaled && wait) {
      struct ibv_wc wc;

      do {
        wc.status = IBV_WC_SUCCESS;
        ne = ibv_poll_cq(localQP.send_cq, 1, &wc);
        if (wc.status != IBV_WC_SUCCESS) {
          Logging::error(__FILE__, __LINE__,
                         "RDMA completion event in CQ with error! " +
                             to_string(wc.status));
          return false;
        }

#ifdef BACKOFF
        if (ne == 0) {
          __asm__("pause");
        }
#endif
      } while (ne == 0);

      if (ne < 0) {
        Logging::error(__FILE__, __LINE__, "RDMA polling from CQ failed!");
        return false;
      }
    }

    return true;
  }

  virtual void destroyQPs() override;
  bool createQP(struct ib_qp_t* qp) override;
  bool createQP(size_t srq_id, struct ib_qp_t& qp);
  bool modifyQPToInit(struct ibv_qp* qp);
  bool modifyQPToRTR(struct ibv_qp* qp, uint32_t remote_qpn, uint16_t dlid,
                     uint8_t* dgid);
  bool modifyQPToRTS(struct ibv_qp* qp);

  // shared receive queues
  map<size_t, sharedrq_t> m_srqs;
  size_t m_srqCounter = 0;
  map<size_t, vector<rdmaConnID>> m_connectedQPs;
};

}  // namespace rdma

#endif /* ReliableRDMA_H_ */
