

#include "ReliableRDMA.h"

using namespace rdma;

/********** constructor and destructor **********/
ReliableRDMA::ReliableRDMA() : ReliableRDMA(Config::RDMA_MEMSIZE) {}

ReliableRDMA::ReliableRDMA(size_t mem_size) : BaseRDMA(mem_size) {
  m_qpType = IBV_QPT_RC;
}

ReliableRDMA::~ReliableRDMA() {
  // destroy QPS
  destroyQPs();
  m_qps.clear();
}

/********** public methods **********/
rdma_mem_t ReliableRDMA::remoteAlloc(const size_t &size) {
  return internalAlloc(size);  // nullptr
}

void *ReliableRDMA::localAlloc(const size_t &size) {
  rdma_mem_t memRes = internalAlloc(size);
  if (!memRes.isnull) {
    return (void *)((char *)m_res.buffer + memRes.offset);
  }
  return nullptr;
}

bool ReliableRDMA::remoteFree(const size_t &offset) {
  return internalFree(offset);
}

bool ReliableRDMA::localFree(const size_t &offset) {
  return internalFree(offset);
}

bool ReliableRDMA::localFree(const void *ptr) {
  char *begin = (char *)m_res.buffer;
  char *end = (char *)ptr;
  size_t offset = end - begin;
  return internalFree(offset);
}

bool ReliableRDMA::initQPWithSuppliedID(const rdmaConnID rdmaConnID) {
  // create completion queues
  struct ib_qp_t qp;
  if (!createCQ(qp.send_cq, qp.recv_cq)) {
    Logging::error(__FILE__, __LINE__, "Failed to create CQ");
    return false;
  }

  // create queues
  if (!createQP(&qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to create QP");
    return false;
  }

  // create local connection data
  struct ib_conn_t localConn;
  union ibv_gid my_gid;
  memset(&my_gid, 0, sizeof my_gid);

  localConn.buffer = (uint64_t)m_res.buffer;
  localConn.rc.rkey = m_res.mr->rkey;
  localConn.qp_num = qp.qp->qp_num;
  localConn.lid = m_res.port_attr.lid;
  memcpy(localConn.gid, &my_gid, sizeof my_gid);

  // init queue pair
  if (!modifyQPToInit(qp.qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to initialize QP");
    return false;
  }

  // done
  setQP(rdmaConnID, qp);
  setLocalConnData(rdmaConnID, localConn);

  Logging::debug(__FILE__, __LINE__, "Created RC queue pair");

  return true;
}

bool ReliableRDMA::initQP(rdmaConnID &retRdmaConnID) {
  retRdmaConnID = nextConnKey();  // qp.qp->qp_num;
  return initQPWithSuppliedID(retRdmaConnID);
}

bool ReliableRDMA::connectQP(const rdmaConnID rdmaConnID) {
  // if QP is connected return

  if (m_connected.count(rdmaConnID) == 1) {
    return true;
  }

  Logging::debug(__FILE__, __LINE__, "ReliableRDMA::connectQP: CONNECT");
  // connect local and remote QP
  struct ib_qp_t qp = m_qps[rdmaConnID];
  struct ib_conn_t remoteConn = m_rconns[rdmaConnID];
  if (!modifyQPToRTR(qp.qp, remoteConn.qp_num, remoteConn.lid,
                     remoteConn.gid)) {
    Logging::error(__FILE__, __LINE__, "Failed to modify QP state to RTR");
    return false;
  }

  if (!modifyQPToRTS(qp.qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to modify QP state to RTS");
    return false;
  }

  m_connected[rdmaConnID] = true;
  Logging::debug(__FILE__, __LINE__, "Connected RC queue pair!");

  return true;
}

void ReliableRDMA::destroyQPs() {
  for (auto &qp : m_qps) {
    if (qp.qp != nullptr) {
      if (ibv_destroy_qp(qp.qp) != 0) {
        Logging::error(__FILE__, __LINE__, "Error, ibv_destroy_qp() failed");
      }

      if (!destroyCQ(qp.send_cq, qp.recv_cq)) {
        Logging::error(__FILE__, __LINE__, "Error, destroyCQ() failed");
      }
    }
  }

  // destroy srq's
  for (auto &kv : m_srqs) {
    if (ibv_destroy_srq(kv.second.shared_rq)) {
      Logging::error(__FILE__, __LINE__,
                     "ReliableRDMASRQ::destroyQPs: ibv_destroy_srq() failed ");
    }
  }
}

bool ReliableRDMA::write(const rdmaConnID rdmaConnID, size_t offset,
                         const void *memAddr, size_t size, bool signaled) {
  return remoteAccess(rdmaConnID, offset, memAddr, size, signaled, true,
                      IBV_WR_RDMA_WRITE);
}

bool ReliableRDMA::read(const rdmaConnID rdmaConnID, size_t offset,
                        const void *memAddr, size_t size, bool signaled) {
  return remoteAccess(rdmaConnID, offset, memAddr, size, signaled, true,
                      IBV_WR_RDMA_READ);
}

bool ReliableRDMA::requestRead(const rdmaConnID rdmaConnID, size_t offset,
                               const void *memAddr, size_t size) {
  return remoteAccess(rdmaConnID, offset, memAddr, size, true, false,
                      IBV_WR_RDMA_READ);
}

bool ReliableRDMA::fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                               const void *memAddr, size_t size,
                               bool signaled) {
  struct ib_qp_t localQP = m_qps[rdmaConnID];
  struct ib_conn_t remoteConn = m_rconns[rdmaConnID];

  int ne = 0;
  struct ibv_send_wr sr;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)memAddr;
  sge.lkey = m_res.mr->lkey;
  sge.length = size;
  memset(&sr, 0, sizeof(sr));
  sr.sg_list = &sge;
  sr.wr_id = 0;
  sr.num_sge = 1;
  sr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  if (signaled) {
    sr.send_flags = IBV_SEND_SIGNALED;
  } else {
    sr.send_flags = 0;
  }
  // calculate remote address using offset in local buffer
  sr.wr.atomic.remote_addr = remoteConn.buffer + offset;
  sr.wr.atomic.rkey = remoteConn.rc.rkey;
  sr.wr.atomic.compare_add = 1ULL;

  struct ibv_send_wr *bad_wr = nullptr;
  if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
    Logging::error(__FILE__, __LINE__, "RDMA OP not successful! ");
    return false;
  }

  if (signaled) {
    struct ibv_wc wc;

    do {
      wc.status = IBV_WC_SUCCESS;
      ne = ibv_poll_cq(localQP.send_cq, 1, &wc);
      if (wc.status != IBV_WC_SUCCESS) {
        Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
        Logging::error(
            __FILE__, __LINE__,
            "RDMA completion event in CQ with error!" + to_string(wc.status));

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

bool ReliableRDMA::compareAndSwap(const rdmaConnID rdmaConnID, size_t offset,
                                  const void *memAddr, int toCompare,
                                  int toSwap, size_t size, bool signaled) {
  // connect local and remote QP

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
  sr.wr_id = 0;
  sr.num_sge = 1;
  sr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  if (signaled) {
    sr.send_flags = IBV_SEND_SIGNALED;
  } else {
    sr.send_flags = 0;
  }

  // calculate remote address using offset in local buffer
  sr.wr.atomic.remote_addr = remoteConn.buffer + offset;
  sr.wr.atomic.rkey = remoteConn.rc.rkey;
  sr.wr.atomic.compare_add = (uint64_t)toCompare;
  sr.wr.atomic.swap = (uint64_t)toSwap;

  struct ibv_send_wr *bad_wr = NULL;
  if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
    Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
    Logging::error(__FILE__, __LINE__, "RDMA OP not successful! ");
    return false;
  }

  if (signaled) {
    struct ibv_wc wc;

    do {
      wc.status = IBV_WC_SUCCESS;
      ne = ibv_poll_cq(localQP.qp->send_cq, 1, &wc);
      if (wc.status != IBV_WC_SUCCESS) {
        Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
        Logging::error(
            __FILE__, __LINE__,
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

bool ReliableRDMA::send(const rdmaConnID rdmaConnID, const void *memAddr,
                        size_t size, bool signaled) {
  DebugCode(
      if (memAddr < m_res.buffer ||
          (char *)memAddr + size > (char *)m_res.buffer + m_res.mr->length) {
        Logging::error(__FILE__, __LINE__,
                       "Passed memAddr falls out of buffer addr space");
      })

      checkSignaled(signaled, rdmaConnID);

  struct ib_qp_t localQP = m_qps[rdmaConnID];

  struct ibv_send_wr sr;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)memAddr;
  sge.lkey = m_res.mr->lkey;
  sge.length = size;
  memset(&sr, 0, sizeof(sr));
  sr.sg_list = &sge;
  sr.num_sge = 1;
  sr.opcode = IBV_WR_SEND;
  sr.next = NULL;

  if (signaled) {
    sr.send_flags = IBV_SEND_SIGNALED;
  } else {
    sr.send_flags = 0;
  }

  struct ibv_send_wr *bad_wr = NULL;
  if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
    Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
    Logging::error(__FILE__, __LINE__, "SEND not successful! ");
    return false;
  }

  int ne = 0;
  if (signaled) {
    struct ibv_wc wc;
    do {
      wc.status = IBV_WC_SUCCESS;
      ne = ibv_poll_cq(localQP.send_cq, 1, &wc);

      if (wc.status != IBV_WC_SUCCESS) {
        Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
        Logging::error(
            __FILE__, __LINE__,
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

bool ReliableRDMA::receive(const rdmaConnID rdmaConnID, const void *memAddr,
                           size_t size) {
  DebugCode(if (memAddr < m_res.buffer ||
                memAddr > (char *)m_res.buffer + m_res.mr->length) {
    Logging::error(__FILE__, __LINE__,
                   "Passed memAddr falls out of buffer addr space");
  })

      struct ib_qp_t localQP = m_qps[rdmaConnID];
  struct ibv_sge sge;
  struct ibv_recv_wr wr;
  struct ibv_recv_wr *bad_wr;

  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)memAddr;
  sge.length = size;
  sge.lkey = m_res.mr->lkey;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  if ((errno = ibv_post_recv(localQP.qp, &wr, &bad_wr))) {
    Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
    Logging::error(__FILE__, __LINE__,
                   "RECV has not been posted successfully! ");
    return false;
  }

  return true;
}

bool ReliableRDMA::pollReceive(const rdmaConnID rdmaConnID, bool doPoll) {
  int ne;
  struct ibv_wc wc;

  struct ib_qp_t localQP = m_qps[rdmaConnID];

  do {
    wc.status = IBV_WC_SUCCESS;
    ne = ibv_poll_cq(localQP.recv_cq, 1, &wc);

    if (wc.status != IBV_WC_SUCCESS) {
      Logging::error(
          __FILE__, __LINE__,
          "RDMA completion event in CQ with error! " + to_string(wc.status));
      return false;
    }
  } while (ne == 0 && doPoll);

  if (doPoll) {
    if (ne < 0) {
      Logging::error(__FILE__, __LINE__, "RDMA polling from CQ failed!");
      return false;
    }
    return true;
  } else if (ne > 0) {
    return true;
  }

  return false;
}

bool ReliableRDMA::pollSend(const rdmaConnID rdmaConnID, bool doPoll) {
  int ne;
  struct ibv_wc wc;

  struct ib_qp_t localQP = m_qps[rdmaConnID];

  do {
    wc.status = IBV_WC_SUCCESS;
    ne = ibv_poll_cq(localQP.send_cq, 1, &wc);

    if (wc.status != IBV_WC_SUCCESS) {
      Logging::error(
          __FILE__, __LINE__,
          "RDMA completion event in CQ with error! " + to_string(wc.status));
      return false;
    }
  } while (ne == 0 && doPoll);

  if (doPoll) {
    if (ne < 0) {
      Logging::error(__FILE__, __LINE__, "RDMA polling from CQ failed!");
      return false;
    }
    return true;
  } else if (ne > 0) {
    return true;
  }

  return false;
}

/********** private methods **********/
bool ReliableRDMA::createQP(struct ib_qp_t *qp) {
  // initialize QP attributes
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  memset(&(m_res.device_attr), 0, sizeof(m_res.device_attr));
  // m_res.device_attr.comp_mask |= IBV_EXP_DEVICE_ATTR_EXT_ATOMIC_ARGS
  //         | IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS;

  if (ibv_query_device(m_res.ib_ctx, &(m_res.device_attr))) {
    Logging::error(__FILE__, __LINE__, "Error, ibv_query_device() failed");
    return false;
  }

  // qp_init_attr.pd = m_res.pd;
  qp_init_attr.send_cq = qp->send_cq;
  qp_init_attr.recv_cq = qp->recv_cq;
  qp_init_attr.sq_sig_all =
      0;  // In every WR, it must be decided whether to generate a WC or not
  qp_init_attr.cap.max_inline_data = 0;

  // TODO: Enable atomic for DM cluster
  // qp_init_attr.max_atomic_arg = 32;
  // qp_init_attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
  // qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS |
  // IBV_EXP_QP_INIT_ATTR_PD; qp_init_attr.comp_mask |=
  // IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;

  qp_init_attr.srq = NULL;  // Shared receive queue
  qp_init_attr.qp_type = m_qpType;

  qp_init_attr.cap.max_send_wr = Config::RDMA_MAX_WR;
  qp_init_attr.cap.max_recv_wr = Config::RDMA_MAX_WR;
  qp_init_attr.cap.max_send_sge = Config::RDMA_MAX_SGE;
  qp_init_attr.cap.max_recv_sge = Config::RDMA_MAX_SGE;

  // create queue pair
  if (!(qp->qp = ibv_create_qp(m_res.pd, &qp_init_attr))) {
    Logging::error(__FILE__, __LINE__, "Cannot create queue pair!");
    return false;
  }
  return true;
}

bool ReliableRDMA::modifyQPToInit(struct ibv_qp *qp) {
  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  struct ibv_qp_attr attr;

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = m_ibPort;
  attr.pkey_index = 0;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    Logging::error(__FILE__, __LINE__, "Failed modifyQPToInit!");
    return false;
  }
  return true;
}

bool ReliableRDMA::modifyQPToRTR(struct ibv_qp *qp, uint32_t remote_qpn,
                                 uint16_t dlid, uint8_t *dgid) {
  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer = 0x12;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = dlid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = m_ibPort;
  if (-1 != m_gidIdx) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    memcpy(&attr.ah_attr.grh.dgid, dgid, 16);
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.sgid_index = m_gidIdx;
    attr.ah_attr.grh.traffic_class = 0;
  }

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    Logging::error(__FILE__, __LINE__, "Failed modifyQPToRTR!");
    return false;
  }

  return true;
}

bool ReliableRDMA::modifyQPToRTS(struct ibv_qp *qp) {
  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 0x12;
  attr.retry_cnt = 6;
  attr.rnr_retry = 0;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 16;

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    Logging::error(__FILE__, __LINE__, "Failed modifyQPToRTS!");
    return false;
  }

  return true;
}

bool rdma::ReliableRDMA::fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                                     const void *memAddr, size_t value_to_add,
                                     size_t size, bool signaled) {
  struct ib_qp_t localQP = m_qps[rdmaConnID];
  struct ib_conn_t remoteConn = m_rconns[rdmaConnID];

  int ne = 0;
  struct ibv_send_wr sr;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)memAddr;
  sge.lkey = m_res.mr->lkey;
  sge.length = size;
  memset(&sr, 0, sizeof(sr));
  sr.sg_list = &sge;
  sr.wr_id = 0;
  sr.num_sge = 1;
  sr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  if (signaled) {
    sr.send_flags = IBV_SEND_SIGNALED;
  } else {
    sr.send_flags = 0;
  }
  // calculate remote address using offset in local buffer
  sr.wr.atomic.remote_addr = remoteConn.buffer + offset;
  sr.wr.atomic.rkey = remoteConn.rc.rkey;
  sr.wr.atomic.compare_add = value_to_add;

  struct ibv_send_wr *bad_wr = nullptr;
  if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
    Logging::error(__FILE__, __LINE__, "RDMA OP not successful! ");
    return false;
  }

  if (signaled) {
    struct ibv_wc wc;

    do {
      wc.status = IBV_WC_SUCCESS;
      ne = ibv_poll_cq(localQP.send_cq, 1, &wc);
      if (wc.status != IBV_WC_SUCCESS) {
        Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
        Logging::error(
            __FILE__, __LINE__,
            "RDMA completion event in CQ with error!" + to_string(wc.status));

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

// Shared receive queue

bool ReliableRDMA::receiveSRQ(size_t srq_id, const void *memAddr, size_t size) {
  struct ibv_sge sge;
  struct ibv_recv_wr wr;
  struct ibv_recv_wr *bad_wr;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)memAddr;
  sge.length = size;
  sge.lkey = m_res.mr->lkey;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  if ((errno = ibv_post_srq_recv(m_srqs.at(srq_id).shared_rq, &wr, &bad_wr))) {
    Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
    Logging::error(__FILE__, __LINE__,
                   "RECV has not been posted successfully! ");
    return false;
  }

  return true;
}

bool ReliableRDMA::pollReceiveBatch(size_t srq_id, size_t &num_completed,
                                    bool &doPoll) {
  int ne;
  const int batchSize = 64;
  struct ibv_wc wc[batchSize];

  do {
    ne = ibv_poll_cq(m_srqs.at(srq_id).recv_cq, batchSize, wc);
    for (int i = 0; i < ne; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        Logging::error(__FILE__, __LINE__,
                       "RDMA completion event in CQ with error! " +
                           to_string(wc[i].status));
        return false;
      }
    }
  } while (ne == 0 && doPoll);

  num_completed = ne;
  if (doPoll) {
    if (ne < 0) {
      Logging::error(__FILE__, __LINE__, "RDMA polling from CQ failed!");
      return false;
    }
    return true;
  } else if (ne > 0) {
    return true;
  }

  return false;
}

bool ReliableRDMA::pollReceiveSRQ(size_t srq_id, rdmaConnID &retRdmaConnID,
                                  bool &doPoll) {
  int ne;
  struct ibv_wc wc;

  do {
    wc.status = IBV_WC_SUCCESS;
    ne = ibv_poll_cq(m_srqs.at(srq_id).recv_cq, 1, &wc);
    if (wc.status != IBV_WC_SUCCESS) {
      Logging::error(
          __FILE__, __LINE__,
          "RDMA completion event in CQ with error! " + to_string(wc.status));
      return false;
    }

  } while (ne == 0 && doPoll);

  if (doPoll) {
    if (ne < 0) {
      Logging::error(__FILE__, __LINE__, "RDMA polling from CQ failed!");
      return false;
    }
    uint64_t qp = wc.qp_num;
    retRdmaConnID = m_qpNum2connID.at(qp);
    return true;
  } else if (ne > 0) {
    return true;
  }

  return false;
}

bool ReliableRDMA::createSharedReceiveQueue(size_t &ret_srq_id) {
  Logging::debug(__FILE__, __LINE__,
                 "ReliableRDMA::createSharedReceiveQueue: Method Called");

  struct ibv_srq_init_attr srq_init_attr;
  sharedrq_t srq;
  memset(&srq_init_attr, 0, sizeof(srq_init_attr));

  srq_init_attr.attr.max_wr = Config::RDMA_MAX_WR;
  srq_init_attr.attr.max_sge = Config::RDMA_MAX_SGE;

  srq.shared_rq = ibv_create_srq(m_res.pd, &srq_init_attr);
  if (!srq.shared_rq) {
    Logging::error(__FILE__, __LINE__, "Error, ibv_create_srq() failed!");
    return false;
  }

  if (!(srq.recv_cq =
            ibv_create_cq(srq.shared_rq->context, Config::RDMA_MAX_WR + 1,
                          nullptr, nullptr, 0))) {
    Logging::error(__FILE__, __LINE__, "Cannot create receive CQ!");
    return false;
  }

  Logging::debug(__FILE__, __LINE__, "Created shared receive queue");

  ret_srq_id = m_srqCounter;
  m_srqs[ret_srq_id] = srq;
  m_srqCounter++;
  return true;
}

bool ReliableRDMA::initQPForSRQWithSuppliedID(size_t srq_id,
                                              const rdmaConnID rdmaConnID) {
  Logging::debug(
      __FILE__, __LINE__,
      "ReliableRDMA::initQP: Method Called with SRQID" + to_string(srq_id));
  struct ib_qp_t qp;
  // create queues
  if (!createQP(srq_id, qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to create QP");
    return false;
  }

  // create local connection data
  struct ib_conn_t localConn;
  union ibv_gid my_gid;
  memset(&my_gid, 0, sizeof my_gid);

  localConn.buffer = (uint64_t)m_res.buffer;
  localConn.rc.rkey = m_res.mr->rkey;
  localConn.qp_num = qp.qp->qp_num;
  localConn.lid = m_res.port_attr.lid;
  memcpy(localConn.gid, &my_gid, sizeof my_gid);

  // init queue pair
  if (!modifyQPToInit(qp.qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to initialize QP");
    return false;
  }

  // done
  setQP(rdmaConnID, qp);
  setLocalConnData(rdmaConnID, localConn);
  m_connectedQPs[srq_id].push_back(rdmaConnID);
  Logging::debug(__FILE__, __LINE__, "Created RC queue pair");

  return true;
}

[[deprecated]] bool ReliableRDMA::initQPForSRQ(size_t srq_id,
                                               rdmaConnID &retRdmaConnID) {
  retRdmaConnID = nextConnKey();  // qp.qp->qp_num;
  return initQPForSRQWithSuppliedID(srq_id, retRdmaConnID);
}

bool ReliableRDMA::createQP(size_t srq_id, struct ib_qp_t &qp) {
  // initialize QP attributes
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  memset(&(m_res.device_attr), 0, sizeof(m_res.device_attr));
  // m_res.device_attr.comp_mask |= IBV_EXP_DEVICE_ATTR_EXT_ATOMIC_ARGS
  //         | IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS;

  if (ibv_query_device(m_res.ib_ctx, &(m_res.device_attr))) {
    Logging::error(__FILE__, __LINE__, "Error, ibv_query_device() failed");
    return false;
  }

  // send queue
  if (!(qp.send_cq = ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1,
                                   nullptr, nullptr, 0))) {
    Logging::error(__FILE__, __LINE__, "Cannot create send CQ!");
    return false;
  }

  qp.recv_cq = m_srqs[srq_id].recv_cq;

  qp_init_attr.send_cq = qp.send_cq;
  qp_init_attr.recv_cq = m_srqs[srq_id].recv_cq;
  qp_init_attr.sq_sig_all =
      0;  // In every WR, it must be decided whether to generate a WC or not
  qp_init_attr.cap.max_inline_data = 0;

  // TODO: Enable atomic for DM cluster
  // qp_init_attr.max_atomic_arg = 32;
  // qp_init_attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
  // qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS |
  // IBV_EXP_QP_INIT_ATTR_PD; qp_init_attr.comp_mask |=
  // IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;

  qp_init_attr.srq = m_srqs[srq_id].shared_rq;  // Shared receive queue
  qp_init_attr.qp_type = m_qpType;

  qp_init_attr.cap.max_send_wr = Config::RDMA_MAX_WR;
  qp_init_attr.cap.max_recv_wr = Config::RDMA_MAX_WR;
  qp_init_attr.cap.max_send_sge = Config::RDMA_MAX_SGE;
  qp_init_attr.cap.max_recv_sge = Config::RDMA_MAX_SGE;

  // create queue pair
  if (!(qp.qp = ibv_create_qp(m_res.pd, &qp_init_attr))) {
    Logging::error(__FILE__, __LINE__, "Cannot create queue pair!");
    return false;
  }
  return true;
}