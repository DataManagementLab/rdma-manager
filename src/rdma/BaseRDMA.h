/**
 * @file BaseRDMA.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef BaseRDMA_H_
#define BaseRDMA_H_

#include "../proto/ProtoClient.h"
#include "../utils/Config.h"

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <list>
#include <mutex>
#include <unordered_map>

namespace rdma {

enum rdma_transport_t { rc, ud };

struct ib_resource_t {
  /* Memory region */
  void *buffer;      /* memory buffer pointer */
  struct ibv_pd *pd; /* PD handle */
  struct ibv_mr *mr; /* MR handle for buf */

  /* Device attributes */
  struct ibv_device_attr device_attr;
  struct ibv_port_attr port_attr; /* IB port attributes */
  struct ibv_context *ib_ctx;     /* device handle */
};

struct ib_qp_t {
  struct ibv_qp *qp;      /* Queue pair */
  struct ibv_cq *send_cq; /* Completion Queue */
  struct ibv_cq *recv_cq;

  ib_qp_t() : qp(nullptr), send_cq(nullptr), recv_cq(nullptr) {}
};

struct ib_conn_t {
  uint64_t buffer; /*  Buffer address */
  uint64_t qp_num; /*  QP number */
  uint16_t lid;    /*  LID of the IB port */
  uint8_t gid[16]; /* GID */

  struct {
    uint32_t rkey; /*  Remote memory key */
  } rc;
  struct {
    uint32_t psn;      /* PSN*/
    struct ibv_ah *ah; /* Route to remote QP*/
  } ud;
};

struct rdma_mem_t {
  size_t size; /* size of memory region */
  bool free;
  size_t offset;
  bool isnull;

  rdma_mem_t(size_t initSize, bool initFree, size_t initOffset)
      : size(initSize), free(initFree), offset(initOffset), isnull(false) {}

  rdma_mem_t() : size(0), free(false), offset(0), isnull(true) {}
};

class BaseRDMA {
 protected:
  typedef size_t rdmaConnID;  // Indexs m_qps

 public:
  // constructors and destructor
  BaseRDMA(size_t mem_size);

  virtual ~BaseRDMA();

  // unicast transfer methods
  virtual bool send(const rdmaConnID rdmaConnID, const void *memAddr,
                    size_t size, bool signaled) = 0;
  virtual bool receive(const rdmaConnID rdmaConnID, const void *memAddr,
                       size_t size) = 0;
  virtual bool pollReceive(const rdmaConnID rdmaConnID, bool doPoll = true) = 0;
  // virtual bool pollReceive(const rdmaConnID rdmaConnID, uint32_t &ret_qp_num)
  // = 0;

  virtual bool pollSend(const rdmaConnID rdmaConnID, bool doPoll = true) = 0;

  // unicast connection management
  virtual bool initQPWithSuppliedID(const rdmaConnID suppliedID) = 0;
  virtual bool initQP(rdmaConnID &retRdmaConnID) = 0;

  virtual bool connectQP(const rdmaConnID rdmaConnID) = 0;

  uint64_t getQPNum(const rdmaConnID rdmaConnID) {
    return m_qps[rdmaConnID].qp->qp_num;
  }

  ib_conn_t getLocalConnData(const rdmaConnID rdmaConnID) {
    return m_lconns[rdmaConnID];
  }

  ib_conn_t getRemoteConnData(const rdmaConnID rdmaConnID) {
    return m_rconns[rdmaConnID];
  }

  void setRemoteConnData(const rdmaConnID rdmaConnID, ib_conn_t &conn);

  // memory management
  virtual void *localAlloc(const size_t &size) = 0;
  virtual bool localFree(const void *ptr) = 0;
  virtual bool localFree(const size_t &offset) = 0;

  void *getBuffer() { return m_res.buffer; }

  const list<rdma_mem_t> getFreeMemList() const { return m_rdmaMem; }

  void *getOffsetToPtr(size_t offset) {
    // check if already allocated
    return (void *)((char *)m_res.buffer + offset);
  }

  size_t getBufferSize() { return m_memSize; }

  void printBuffer();

 protected:
  virtual void destroyQPs() = 0;

  // memory management
  bool createBuffer();

  bool mergeFreeMem(list<rdma_mem_t>::iterator &iter);

  rdma_mem_t internalAlloc(const size_t &size);

  bool internalFree(const size_t &offset);

  uint64_t nextConnKey() { return m_qps.size(); }

  void setQP(const rdmaConnID rdmaConnID, ib_qp_t &qp);

  void setLocalConnData(const rdmaConnID rdmaConnID, ib_conn_t &conn);

  bool createCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq);
  bool destroyCQ(ibv_cq *&send_cq, ibv_cq *&rcv_cq);
  virtual bool createQP(struct ib_qp_t *qp) = 0;

  inline void __attribute__((always_inline))
  checkSignaled(bool &signaled, rdmaConnID rdmaConnID) {
    if (signaled) return;

    ++m_countWR[rdmaConnID];
    if (m_countWR[rdmaConnID] == Config::RDMA_MAX_WR) {
      signaled = true;
      m_countWR[rdmaConnID] = 0;
    }
  }

  vector<size_t> m_countWR;

  ibv_qp_type m_qpType;
  size_t m_memSize;
  int m_numaRegion;
  int m_rdmaDevice;
  int m_ibPort;
  int m_gidIdx;

  struct ib_resource_t m_res;

  // maps ibaddr.conn_keys to connection infos
  size_t m_lastConnKey;
  vector<ib_qp_t> m_qps;  // rdmaConnID is the index of the vector
  vector<ib_conn_t> m_rconns;
  vector<ib_conn_t> m_lconns;

  unordered_map<uint64_t, bool> m_connected;
  unordered_map<uint64_t, rdmaConnID> m_qpNum2connID;

  list<rdma_mem_t> m_rdmaMem;
  unordered_map<size_t, rdma_mem_t> m_usedRdmaMem;

  static rdma_mem_t s_nillmem;
};

}  // namespace rdma

#endif /* BaseRDMA_H_ */
