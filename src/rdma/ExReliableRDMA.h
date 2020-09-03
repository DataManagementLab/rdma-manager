/**
 * @file ExReliableRDMA.h
 * @author cbinnig, tziegler, dfailing, anewswanger
 * @date 2020-08-27
 */

#ifndef EXRELIABLERDMA_H_
#define EXRELIABLERDMA_H_

#include "../utils/Config.h"
#include <atomic>
#include "ReliableRDMA.h"

namespace rdma {

class ExReliableRDMA : public ReliableRDMA {
 public:
  ExReliableRDMA();
  ExReliableRDMA(size_t mem_size);
  ExReliableRDMA(size_t mem_size, int numaNode);
  ~ExReliableRDMA();

  void initQPWithSuppliedID(const rdmaConnID suppliedID) ;
  //void initQPWithSuppliedID(struct ib_qp_t** qp ,struct ib_conn_t ** localConn) override;

  void initQP(const rdmaConnID retRdmaConnID) ;
  void connectQP(const rdmaConnID rdmaConnID) override;

  // void write(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
  //            size_t size, bool signaled);

  void createSharedReceiveQueue();

 protected:
  void initXRC();
  void destroyXRC();
  virtual void destroyQPs() override;
  void createQP(struct ib_qp_t*, ibv_qp_type qp_type);
  virtual void destroyCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq) override;

  void modifyQPToInit(struct ibv_qp* qp);
  void modifyQPToRTR(struct ibv_qp* qp, uint32_t remote_qpn, uint16_t dlid,
                     uint8_t* dgid);
  void modifyQPToRTS(struct ibv_qp* qp);

  // shared receive queues
  //map<size_t, sharedrq_t> m_srqs;
  //size_t m_srqCounter = 0;
  //map<size_t, vector<rdmaConnID>> m_connectedQPs;

  // XRC related
  ibv_xrcd* xrcd;
  int xrc_fd; //!< file descriptor for xrcd file/socket
  //TODO XRC one ib_qp_t can be stored in combination with the connection, but the other one need to be stored seperatly (recv or send qp)
  vector<ib_qp_t> m_xrc_recv_qps;
  ibv_cq* send_cq;
  ibv_cq* recv_cq;
};

}  // namespace rdma

#endif /* EXRELIABLE_RDMA_H_ */
