/**
 * @file UnreliableRDMA.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef UnreliableRDMA_H_
#define UnreliableRDMA_H_

#include "../utils/Config.h"

#include "BaseRDMA.h"

#include <arpa/inet.h>
#include <rdma/rdma_verbs.h>

namespace rdma {

struct rdma_mcast_conn_t {
  char *mcast_addr;
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

class UnreliableRDMA : public BaseRDMA {
 public:
  UnreliableRDMA(size_t mem_size = Config::RDMA_MEMSIZE);
  ~UnreliableRDMA();

  bool initQPWithSuppliedID(const rdmaConnID suppliedID) override;
  bool initQP(rdmaConnID &retRdmaConnID) override;
  bool connectQP(const rdmaConnID rdmaConnID) override;

  bool send(const rdmaConnID rdmaConnID, const void *memAddr, size_t size,
            bool signaled) override;
  bool receive(const rdmaConnID rdmaConnID, const void *memAddr,
               size_t size) override;
  bool pollReceive(const rdmaConnID rdmaConnID, bool doPoll) override;
  bool pollSend(const rdmaConnID rdmaConnID, bool doPoll) override;

  void *localAlloc(const size_t &size) override;
  bool localFree(const void *ptr) override;
  bool localFree(const size_t &offset) override;

  bool joinMCastGroup(string mCastAddress, rdmaConnID &retRdmaConnID);
  bool leaveMCastGroup(const rdmaConnID rdmaConnID);
  bool sendMCast(const rdmaConnID rdmaConnID, const void *memAddr, size_t size,
                 bool signaled);
  bool receiveMCast(const rdmaConnID rdmaConnID, const void *memAddr,
                    size_t size);
  bool pollReceiveMCast(const rdmaConnID rdmaConnID);

 private:
  bool createQP(struct ib_qp_t *qp) override;
  void destroyQPs() override;
  bool modifyQPToInit(struct ibv_qp *qp);
  bool modifyQPToRTR(struct ibv_qp *qp);
  bool modifyQPToRTS(struct ibv_qp *qp, const uint32_t psn);

  inline uint64_t nextMCastConnKey() { return m_lastMCastConnKey++; }

  void setMCastConn(const rdmaConnID rdmaConnID, rdma_mcast_conn_t &conn);

  bool getCmEvent(struct rdma_event_channel *channel,
                  enum rdma_cm_event_type type, struct rdma_cm_event **out_ev);

  // only one QP needed for all connections
  ib_qp_t m_udqp;
  ib_conn_t m_udqpConn;
  ib_qp_t m_udqpMgmt;

  // maps mcastConnkey to MCast connections
  size_t m_lastMCastConnKey;
  vector<rdma_mcast_conn_t> m_udpMcastConns;
};

}  // namespace rdma

#endif /* UnreliableRDMA_H_ */
