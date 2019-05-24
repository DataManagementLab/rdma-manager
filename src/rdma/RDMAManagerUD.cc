

#include "./RDMAManagerUD.h"

/********** constructor and destructor **********/
RDMAManagerUD::RDMAManagerUD(size_t mem_size)
    : RDMAManager(mem_size) {
  m_qpType = IBV_QPT_UD;
  m_lastMCastConnKey = 0;
}

RDMAManagerUD::~RDMAManagerUD() {
  destroyManager();
}

/********** public methods **********/
rdma_mem_t RDMAManagerUD::remoteAlloc(const size_t& size) {
  (void) (size);
  Logging::error(__FILE__, __LINE__, "Remote memory not supported with UD");
  return s_nillmem;
}

void* RDMAManagerUD::localAlloc(const size_t& size) {
  rdma_mem_t memRes = internalAlloc(size + Config::RDMA_UD_OFFSET);
  if (!memRes.isnull) {
    return (void*) ((char*) m_res.buffer + memRes.offset
        + Config::RDMA_UD_OFFSET);
  }
  return nullptr;
}

bool RDMAManagerUD::remoteFree(const size_t& offset) {
  (void) (offset);
  Logging::error(__FILE__, __LINE__, "Remote memory not supported with UD");
  return false;
}

bool RDMAManagerUD::localFree(const size_t& offset) {
  return internalFree(offset);
}

bool RDMAManagerUD::localFree(const void* ptr) {
  char* begin = (char*) m_res.buffer;
  char* end = (char*) ptr;
  size_t offset = (end - begin) - Config::RDMA_UD_OFFSET;
  return internalFree(offset);
}

bool RDMAManagerUD::initQP(struct ib_addr_t& retIbAddr, bool isMgmtQP) {
  //assign new QP number
  uint64_t connKey = nextConnKey();
  retIbAddr.conn_key = connKey;

  //check if QP is already created
  if (!isMgmtQP && m_udqp.qp != nullptr) {
    setQP(retIbAddr, m_udqp);
    setLocalConnData(retIbAddr, m_udqpConn);
    return true;
  } else if (isMgmtQP && m_udqpMgmt.qp != nullptr) {
    setQP(retIbAddr, m_udqpMgmt);
    setLocalConnData(retIbAddr, m_udqpMgmtConn);
    return true;
  }

  ib_qp_t* qp = nullptr;
  ib_conn_t* qpConn = nullptr;
  if (!isMgmtQP) {
    qp = &m_udqp;
    qpConn = &m_udqpConn;
  } else {
    qp = &m_udqpMgmt;
    qpConn = &m_udqpMgmtConn;
  }

  //create completion queues
  if (!createCQ(qp->send_cq, qp->recv_cq)) {
    Logging::error(__FILE__, __LINE__, "Failed to create CQ");
    return false;
  }

  //create QP
  if (!createQP(qp)) {
    return false;
  }

  // create local connection data
  union ibv_gid my_gid;
  memset(&my_gid, 0, sizeof my_gid);
  qpConn->buffer = (uintptr_t) m_res.buffer;
  qpConn->qp_num = m_udqp.qp->qp_num;
  qpConn->lid = m_res.port_attr.lid;
  memcpy(qpConn->gid, &my_gid, sizeof my_gid);
  qpConn->ud.psn = lrand48() & 0xffffff;
  qpConn->ud.ah = nullptr;

  // init queue pair
  if (!modifyQPToInit(qp->qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to initialize QP");
    return false;
  }

  if (!modifyQPToRTR(qp->qp)) {
    Logging::error(__FILE__, __LINE__, "Failed to modify QP state to RTR");
    return false;
  }

  if (!modifyQPToRTS(qp->qp, qpConn->ud.psn)) {
    Logging::error(__FILE__, __LINE__, "Failed to modify QP state to RTS");
    return false;
  }

  //done
  setQP(retIbAddr, *qp);
  setLocalConnData(retIbAddr, *qpConn);

  Logging::debug(__FILE__, __LINE__, "Created UD queue pair ");

  return true;
}

bool RDMAManagerUD::connectQP(struct ib_addr_t& ibAddr) {
  // if QP is connected return
  uint64_t connKey = ibAddr.conn_key;
  if (m_connected.count(connKey) == 1) {
    return true;
  }

  //create address handle
  struct ibv_ah_attr ah_attr;
  memset(&ah_attr, 0, sizeof ah_attr);
  ah_attr.is_global = 0;
  ah_attr.dlid = m_rconns[connKey].lid;
  ah_attr.sl = 0;
  ah_attr.src_path_bits = 0;
  ah_attr.port_num = m_ibPort;
  struct ibv_ah* ah = ibv_create_ah(m_res.pd, &ah_attr);
  m_rconns[connKey].ud.ah = ah;

  m_connected[connKey] = true;
  Logging::debug(__FILE__, __LINE__, "Connected UD queue pair!");

  return true;
}

void RDMAManagerUD::destroyQPs() {
  if (m_udqp.qp != nullptr) {
    if (ibv_destroy_qp(m_udqp.qp) != 0) {
      Logging::error(__FILE__, __LINE__,
          "Error, ibv_destroy_qp() failed");
    }

    if (!destroyCQ(m_udqp.send_cq, m_udqp.recv_cq)) {
      Logging::error(__FILE__, __LINE__, "Error, destroyCQ() failed");
    }
    m_udqp.qp = nullptr;
  }

  if (m_udqpMgmt.qp != nullptr) {
    if (ibv_destroy_qp(m_udqpMgmt.qp) != 0) {
      Logging::error(__FILE__, __LINE__,
          "Error, ibv_destroy_qp() failed");
    }
    m_udqpMgmt.qp = nullptr;
  }

}

bool RDMAManagerUD::remoteWrite(struct ib_addr_t& ibAddr, size_t offset,
                                const void* memAddr, size_t size,
                                bool signaled) {
  (void) (ibAddr);
  (void) (offset);
  (void) (memAddr);
  (void) (size);
  (void) (signaled);
  Logging::error(__FILE__, __LINE__, "RDMA WRITE not supported with UD");
  return false;
}

bool RDMAManagerUD::remoteRead(struct ib_addr_t& ibAddr, size_t offset,
                               const void* memAddr, size_t size,
                               bool signaled) {
  (void) (ibAddr);
  (void) (offset);
  (void) (memAddr);
  (void) (size);
  (void) (signaled);
  Logging::error(__FILE__, __LINE__, "RDMA READ not supported with UD");
  return false;
}

bool RDMAManagerUD::requestRead(struct ib_addr_t& ibAddr, size_t offset,
                                const void* memAddr, size_t size) {
  (void) (ibAddr);
  (void) (offset);
  (void) (memAddr);
  (void) (size);
  Logging::error(__FILE__, __LINE__, "RDMA READ not supported with UD");
  return false;
}

bool RDMAManagerUD::remoteFetchAndAdd(struct ib_addr_t& ibAddr, size_t offset,
                                      const void* memAddr, size_t size,
                                      bool signaled) {
  (void) (ibAddr);
  (void) (offset);
  (void) (memAddr);
  (void) (size);
  (void) (signaled);
  Logging::error(__FILE__, __LINE__, "RDMA F&A not supported with UD");
  return false;
}

bool RDMAManagerUD::remoteCompareAndSwap(struct ib_addr_t& ibAddr,
                                         size_t offset, const void* memAddr,
                                         int toCompare, int toSwap, size_t size,
                                         bool signaled) {
  (void) (ibAddr);
  (void) (offset);
  (void) (memAddr);
  (void) (toCompare);
  (void) (toSwap);
  (void) (size);
  (void) (signaled);
  Logging::error(__FILE__, __LINE__, "RDMA C&S not supported with UD");
  return false;
}

bool RDMAManagerUD::send(struct ib_addr_t& ibAddr, const void* memAddr,
                         size_t size, bool signaled) {

  uint64_t connKey = ibAddr.conn_key;
  struct ib_qp_t localQP = m_qps[connKey];
  struct ib_conn_t remoteConn = m_rconns[connKey];

  struct ibv_send_wr sr;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t) memAddr;
  sge.lkey = m_res.mr->lkey;
  sge.length = size;
  memset(&sr, 0, sizeof(sr));
  sr.sg_list = &sge;
  sr.num_sge = 1;
  sr.opcode = IBV_WR_SEND;
  sr.next = NULL;

  sr.wr.ud.ah = remoteConn.ud.ah;
  sr.wr.ud.remote_qpn = remoteConn.qp_num;
  sr.wr.ud.remote_qkey = 0x11111111;  //remoteConn.ud.qkey;
  sr.send_flags = (signaled) ? IBV_SEND_SIGNALED : 0;

  struct ibv_send_wr *bad_wr = NULL;
  if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr)) != 0) {
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
        Logging::errorNo(__FILE__, __LINE__, std::strerror(errno),
        errno);
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

bool RDMAManagerUD::receive(struct ib_addr_t& ibAddr, const void* memAddr,
                            size_t size) {

  uint64_t connKey = ibAddr.conn_key;
  struct ib_qp_t localQP = m_qps[connKey];

  struct ibv_sge sge;
  struct ibv_recv_wr wr;
  struct ibv_recv_wr *bad_wr;

  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t) (((char*) memAddr) - Config::RDMA_UD_OFFSET);
  sge.length = size + Config::RDMA_UD_OFFSET;
  sge.lkey = m_res.mr->lkey;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.next = nullptr;

  if ((errno = ibv_post_recv(localQP.qp, &wr, &bad_wr)) != 0) {
    Logging::errorNo(__FILE__, __LINE__, std::strerror(errno), errno);
    Logging::error(__FILE__, __LINE__,
                   "RECV has not been posted successfully! ");
    return false;
  }

  return true;
}

bool RDMAManagerUD::pollReceive(ib_addr_t& ibAddr, bool doPoll) {
  int ne;
  struct ibv_wc wc;

  uint64_t connKey = ibAddr.conn_key;
  struct ib_qp_t localQP = m_qps[connKey];

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

bool RDMAManagerUD::pollSend(ib_addr_t& ibAddr, bool doPoll) {
  int ne;
  struct ibv_wc wc;

  uint64_t connKey = ibAddr.conn_key;
  struct ib_qp_t localQP = m_qps[connKey];

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

bool RDMAManagerUD::joinMCastGroup(string mCastAddress,
                                   struct ib_addr_t& retIbAddr) {
  uint64_t connKey = nextMCastConnKey();
  retIbAddr.conn_key = connKey;

  rdma_mcast_conn_t mCastConn;
  mCastConn.mcast_addr = const_cast<char*>(mCastAddress.c_str());

  //create event channel
  mCastConn.channel = rdma_create_event_channel();
  if (!mCastConn.channel) {
    Logging::error(__FILE__, __LINE__,
                   "Could not create event channel for multicast!");
    return false;
  }

  //create connection
  if (rdma_create_id(mCastConn.channel, &mCastConn.id, NULL, RDMA_PS_UDP)
      != 0) {
    Logging::error(__FILE__, __LINE__,
                   "Could not create connection for multicast!");
    return false;
  }

  //resolve multicast address
  rdma_addrinfo* mcast_rai = nullptr;
  rdma_addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_port_space = RDMA_PS_UDP;
  hints.ai_flags = 0;
  if (rdma_getaddrinfo(mCastConn.mcast_addr, nullptr, &hints, &mcast_rai)
  != 0) {
    Logging::error(__FILE__, __LINE__,
    "Could not resolve info for multicast address (1)!");
    return false;
  }

  if (rdma_resolve_addr(mCastConn.id, nullptr, mcast_rai->ai_dst_addr, 2000)
  != 0) {
    Logging::error(__FILE__, __LINE__,
    "Could not resolve info for multicast address (2)!");
    return false;
  }

  if (!getCmEvent(mCastConn.channel, RDMA_CM_EVENT_ADDR_RESOLVED, nullptr)) {
    Logging::error(__FILE__, __LINE__,
    "Could not resolve info for multicast address (3)!");
    return false;
  }
  memcpy(&mCastConn.mcast_sockaddr, mcast_rai->ai_dst_addr,
         sizeof(struct sockaddr));

  // create protection domain
  mCastConn.pd = ibv_alloc_pd(mCastConn.id->verbs);
  if (!mCastConn.pd) {
    Logging::error(__FILE__, __LINE__,
                   "Could not create multicast protection domain!");
    return false;
  }

  mCastConn.mr = ibv_reg_mr(mCastConn.pd, m_res.buffer, m_memSize,
                            IBV_ACCESS_LOCAL_WRITE);
  if (!mCastConn.mr) {
    Logging::error(
        __FILE__, __LINE__,
        "Could not assign memory region to multicast protection domain!");
    return false;
  }

  // create multicast queues
  ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(attr));

  mCastConn.scq = ibv_create_cq(mCastConn.id->verbs, Config::RDMA_MAX_WR + 1,
  nullptr, nullptr, 0);
  mCastConn.rcq = ibv_create_cq(mCastConn.id->verbs, Config::RDMA_MAX_WR + 1,
  nullptr, nullptr, 0);
  if (!mCastConn.scq || !mCastConn.rcq) {
    Logging::error(__FILE__, __LINE__,
                   "Could not create multicast completion queues!");
    return false;
  }

  attr.qp_type = IBV_QPT_UD;
  attr.send_cq = mCastConn.scq;
  attr.recv_cq = mCastConn.rcq;
  attr.cap.max_send_wr = Config::RDMA_MAX_WR;
  attr.cap.max_recv_wr = Config::RDMA_MAX_WR;
  attr.cap.max_send_sge = Config::RDMA_MAX_SGE;
  attr.cap.max_recv_sge = Config::RDMA_MAX_SGE;
  if (rdma_create_qp(mCastConn.id, mCastConn.pd, &attr) != 0) {
    Logging::error(__FILE__, __LINE__,
                   "Could not create multicast queue pairs!");
    return false;
  }

  // join multicast group
  if (rdma_join_multicast(mCastConn.id, &mCastConn.mcast_sockaddr, nullptr)
  != 0) {
    Logging::error(__FILE__, __LINE__,
    "Could not join multicast group (1)!");
    return false;
  }

  // verify that we successfully joined the multicast group
  rdma_cm_event *event;
  if (!getCmEvent(mCastConn.channel, RDMA_CM_EVENT_MULTICAST_JOIN, &event)) {
    Logging::error(__FILE__, __LINE__, "Could not join multicast group(2)!");
    return false;
  }

  mCastConn.remote_qpn = event->param.ud.qp_num;
  mCastConn.remote_qkey = event->param.ud.qkey;
  mCastConn.ah = ibv_create_ah(m_res.pd, &event->param.ud.ah_attr);
  if (!mCastConn.ah) {
    Logging::error(__FILE__, __LINE__,
                   "Could not join multicast address handle!");
    return false;
  }
  rdma_ack_cm_event(event);

  //done
  setMCastConn(retIbAddr, mCastConn);

  return true;
}

bool RDMAManagerUD::leaveMCastGroup(struct ib_addr_t ibAddr) {
  rdma_mcast_conn_t mCastConn = m_udpMcastConns[ibAddr.conn_key];

  // leave group
  if (rdma_leave_multicast(mCastConn.id, &mCastConn.mcast_sockaddr) != 0) {
    return false;
  }

  // destroy resources
  if (mCastConn.ah)
    ibv_destroy_ah(mCastConn.ah);
  if (mCastConn.id && mCastConn.id->qp)
    rdma_destroy_qp(mCastConn.id);
  if (mCastConn.scq)
    ibv_destroy_cq(mCastConn.scq);
  if (mCastConn.rcq)
    ibv_destroy_cq(mCastConn.rcq);
  if (mCastConn.mr)
    rdma_dereg_mr(mCastConn.mr);
  if (mCastConn.pd)
    ibv_dealloc_pd(mCastConn.pd);
  if (mCastConn.id)
    rdma_destroy_id(mCastConn.id);

  return true;
}

bool RDMAManagerUD::sendMCast(struct ib_addr_t ibAddr, const void* memAddr,
                              size_t size, bool signaled) {
  rdma_mcast_conn_t mCastConn = m_udpMcastConns[ibAddr.conn_key];

  struct ibv_send_wr wr, *bad_wr;
  struct ibv_sge sge;
  sge.length = size;
  sge.lkey = mCastConn.mr->lkey;
  sge.addr = (uintptr_t) memAddr;

  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = (signaled) ? IBV_SEND_SIGNALED : 0;
  wr.wr_id = 0;
  wr.imm_data = htonl(mCastConn.id->qp->qp_num);
  wr.wr.ud.ah = mCastConn.ah;
  wr.wr.ud.remote_qpn = mCastConn.remote_qpn;
  wr.wr.ud.remote_qkey = mCastConn.remote_qkey;
  size_t ret = ibv_post_send(mCastConn.id->qp, &wr, &bad_wr);
  if (ret != 0) {
    Logging::error(
        __FILE__, __LINE__,
        "Sending multicast data failed (error: " + to_string(ret) + ")");
    return false;
  }

  int ne = 0;
  if (signaled) {
    struct ibv_wc wc;
    do {
      wc.status = IBV_WC_SUCCESS;
      ne = ibv_poll_cq(mCastConn.scq, 1, &wc);

      if (wc.status != IBV_WC_SUCCESS) {
        Logging::error(
            __FILE__,
            __LINE__,
            "RDMA completion event in multicast CQ with error! "
                + to_string(wc.status));
        return false;
      }
    } while (ne == 0);

    if (ne < 0) {
      Logging::error(__FILE__, __LINE__,
                     "RDMA polling from multicast CQ failed!");
      return false;
    }
  }

  return true;
}

bool RDMAManagerUD::receiveMCast(struct ib_addr_t ibAddr, const void* memAddr,
                                 size_t size) {
  rdma_mcast_conn_t mCastConn = m_udpMcastConns[ibAddr.conn_key];

  void* buffer = (void*) (((char*) memAddr) - Config::RDMA_UD_OFFSET);
  if (rdma_post_recv(mCastConn.id, nullptr, buffer,
  size + Config::RDMA_UD_OFFSET, mCastConn.mr) != 0) {

    Logging::error(__FILE__, __LINE__, "Receiving multicast data failed");
    return false;
  }

  return true;
}

bool RDMAManagerUD::pollReceiveMCast(struct ib_addr_t ibAddr) {
  rdma_mcast_conn_t mCastConn = m_udpMcastConns[ibAddr.conn_key];
  int ne = 0;
  struct ibv_wc wc;
  do {
    wc.status = IBV_WC_SUCCESS;
    ne = ibv_poll_cq(mCastConn.rcq, 1, &wc);

    if (wc.status != IBV_WC_SUCCESS) {
      Logging::error(
          __FILE__,
          __LINE__,
          "RDMA completion event in multicast CQ with error! "
              + to_string(wc.status));
      return false;
    }

  } while (ne == 0);

  if (ne < 0) {
    Logging::error(__FILE__, __LINE__,
                   "RDMA polling from multicast CQ failed!");
    return false;
  }

  return true;
}

/********** private methods **********/
bool RDMAManagerUD::createQP(struct ib_qp_t* qp) {
// initialize QP attributes
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));

  qp_init_attr.send_cq = qp->send_cq;
  qp_init_attr.recv_cq = qp->recv_cq;
  qp_init_attr.sq_sig_all = 0;  // In every WR, it must be decided whether to generate a WC or not
  qp_init_attr.srq = NULL;
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

bool RDMAManagerUD::modifyQPToInit(struct ibv_qp *qp) {
  int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
  struct ibv_qp_attr attr;

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = m_ibPort;
  attr.pkey_index = 0;
  attr.qkey = 0x11111111;

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    Logging::error(__FILE__, __LINE__, "Failed modifyQPToInit!");
    return false;
  }
  return true;
}

bool RDMAManagerUD::modifyQPToRTR(struct ibv_qp *qp) {
  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_RTR;

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    Logging::error(__FILE__, __LINE__, "Failed modifyQPToRTR!");
    return false;
  }

  return true;
}

bool RDMAManagerUD::modifyQPToRTS(struct ibv_qp *qp, const uint32_t psn) {
  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = psn;

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    Logging::error(__FILE__, __LINE__, "Failed modifyQPToRTS!");
    return false;
  }

  return true;
}

bool dpi::RDMAManagerUD::remoteFetchAndAdd(struct ib_addr_t& ibAddr, size_t offset,
                                               const void* memAddr, size_t value_to_add,
                                               size_t size, bool signaled) {
    (void) (ibAddr);
    (void) (offset);
    (void) (memAddr);
    (void) (size);
    (void) (signaled);
    (void) (value_to_add);
    Logging::error(__FILE__, __LINE__, "RDMA F&A not supported with UD");
    return false;

}
