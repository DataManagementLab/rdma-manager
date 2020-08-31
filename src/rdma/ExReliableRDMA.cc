

#include "ExReliableRDMA.h"
#include <fcntl.h>
#include <stdlib.h>

using namespace rdma;

/********** constructor and destructor **********/
ExReliableRDMA::ExReliableRDMA() : ExReliableRDMA(Config::RDMA_MEMSIZE) {}

//------------------------------------------------------------------------------------//

ExReliableRDMA::ExReliableRDMA(size_t mem_size) : ReliableRDMA(mem_size) {
  m_qpType = IBV_QPT_XRC_RECV;
  initXRC();
}

//------------------------------------------------------------------------------------//

ExReliableRDMA::ExReliableRDMA(size_t mem_size, int numaNode) : ReliableRDMA(mem_size, numaNode) {
  m_qpType = IBV_QPT_XRC_RECV;
  initXRC();
}

//------------------------------------------------------------------------------------//

ExReliableRDMA::~ExReliableRDMA() {
  // destroy QPS
  //destroyQPs();
  //m_qps.clear();
  destroyXRC();
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::initXRC() {
  xrc_fd = open("/tmp/xrc_domain", O_RDONLY | O_CREAT, S_IRUSR | S_IRGRP);
  if (xrc_fd < 0) {
    fprintf(stderr,
        "Couldn't create the file for the XRC Domain "
        "but not stopping %d\n", errno);
    xrc_fd = -1;
  }

  struct ibv_xrcd_init_attr xrcd_attr;
  memset(&xrcd_attr, 0, sizeof xrcd_attr);
  xrcd_attr.comp_mask = IBV_XRCD_INIT_ATTR_FD | IBV_XRCD_INIT_ATTR_OFLAGS;
  xrcd_attr.fd = xrc_fd;
  xrcd_attr.oflags = O_CREAT;
  xrcd = ibv_open_xrcd(m_res.ib_ctx, &xrcd_attr);
  if (!xrcd) {
    throw runtime_error("Failed initXRC, could not open XRC Domain!");
  }

  createCQ(send_cq, recv_cq);

  createSharedReceiveQueue();
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::destroyXRC() {
  if (xrcd) {
    ibv_close_xrcd(xrcd);
  }
  if (xrc_fd >= 0) {
    close(xrc_fd);
  }
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::connectQP(const rdmaConnID rdmaConnID) {
  // if QP is connected return

  //TODO XRC we need to connect two pairs and both need to be set to RTS
  if (m_connected.find(rdmaConnID) != m_connected.end()) {
    return;
  }

  Logging::debug(__FILE__, __LINE__, "ExReliableRDMA::connectQP: CONNECT");
  // connect local and remote QP
  struct ib_qp_t send_qp = m_qps[rdmaConnID];
  struct ib_qp_t recv_qp = m_xrc_recv_qps[rdmaConnID];
  struct ib_conn_t remoteConn = m_rconns[rdmaConnID];

  modifyQPToRTR(send_qp.qp, remoteConn.xrc.recv_qp_num, remoteConn.lid, remoteConn.gid);
  modifyQPToRTS(send_qp.qp);

  modifyQPToRTR(recv_qp.qp, remoteConn.qp_num, remoteConn.lid, remoteConn.gid);
  modifyQPToRTS(recv_qp.qp);

  m_connected[rdmaConnID] = true;
  Logging::debug(__FILE__, __LINE__, "ExConnected RC queue pair!");
}

//------------------------------------------------------------------------------------//

//TODO qp_type?
void ExReliableRDMA::createQP(struct ib_qp_t *qp, ibv_qp_type qp_type) {
  Logging::debug(__FILE__, __LINE__, "ExReliableRDMA::createQP: ");
  // initialize QP attributes
  struct ibv_qp_init_attr_ex qp_init_attr_ex;
  memset(&qp_init_attr_ex, 0, sizeof(qp_init_attr_ex));
  memset(&(m_res.device_attr), 0, sizeof(m_res.device_attr));
  // m_res.device_attr.comp_mask |= IBV_EXP_DEVICE_ATTR_EXT_ATOMIC_ARGS
  //         | IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS;

  //TODO XRC need to use qp_init_attr_ex and using ib_create_qp_ex

  if (ibv_query_device(m_res.ib_ctx, &(m_res.device_attr))) {
    throw runtime_error("Error, ibv_query_device() failed");
  }

  qp_init_attr_ex.qp_type = qp_type;

  if(qp_type == IBV_QPT_XRC_SEND) {
    qp_init_attr_ex.send_cq = qp->send_cq;
    qp_init_attr_ex.cap.max_send_wr = Config::RDMA_MAX_WR;
    qp_init_attr_ex.cap.max_send_sge = Config::RDMA_MAX_SGE;
    qp_init_attr_ex.comp_mask        = IBV_QP_INIT_ATTR_PD;
    qp_init_attr_ex.pd = m_res.pd;
  } else if(qp_type == IBV_QPT_XRC_RECV) {
    qp_init_attr_ex.recv_cq = qp->recv_cq;
    qp_init_attr_ex.cap.max_recv_wr = Config::RDMA_MAX_WR;
    qp_init_attr_ex.cap.max_recv_sge = Config::RDMA_MAX_SGE;
    qp_init_attr_ex.xrcd = xrcd;
    qp_init_attr_ex.comp_mask = IBV_QP_INIT_ATTR_XRCD;
  } else {
    throw runtime_error("Unknown qp_type in ExReliableRDMA::createQP");
  }
  qp_init_attr_ex.sq_sig_all = 0;  // In every WR, it must be decided whether to generate a WC or not
  qp_init_attr_ex.cap.max_inline_data = Config::MAX_RC_INLINE_SEND;

  // TODO: Enable atomic for DM cluster
  // qp_init_attr.max_atomic_arg = 32;
  // qp_init_attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
  // qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS |
  // IBV_EXP_QP_INIT_ATTR_PD; qp_init_attr.comp_mask |=
  // IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;

  //qp_init_attr.srq = NULL;  // Shared receive queue

  // create queue pair
  if (!(qp->qp = ibv_create_qp_ex(m_res.ib_ctx, &qp_init_attr_ex))) {
    throw runtime_error("Cannot create queue pair!");
  }
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::modifyQPToInit(struct ibv_qp *qp) {
  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  struct ibv_qp_attr attr;

  //TODO XRC check, but this can probably remain the same, access_flags differ somewhat to the pingpong example

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = m_ibPort;
  attr.pkey_index = 0;
  if(qp->send_cq) {
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  } else {
    attr.qp_access_flags = 0;
  }

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    throw runtime_error("Failed modifyQPToInit!");
  }
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::modifyQPToRTR(struct ibv_qp *qp, uint32_t remote_qpn,
                                 uint16_t dlid, uint8_t *dgid) {

  //TODO XRC check, but this can probably remain the same

  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn; //pingping: send_qpn (as parameter from connectQP)
  attr.rq_psn = 0; //pingpong: is different/exchanged
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

  //TODO XRC attr.dest_qp_num = remote.recv_qpn for send_qp

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    throw runtime_error("Failed modifyQPToRTR!");
  }
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::modifyQPToRTS(struct ibv_qp *qp) {

  //TODO XRC check, but this can probably remain the same

  struct ibv_qp_attr attr;
  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 0x12;
  attr.retry_cnt = 6;
  attr.rnr_retry = 0;
  attr.sq_psn = 0; //pingpong: is recv/send_psn
  attr.max_rd_atomic = 16;

  if ((errno = ibv_modify_qp(qp, &attr, flags)) > 0) {
    throw runtime_error("Failed modifyQPToRTS!");
  }
}

//------------------------------------------------------------------------------------//

void ExReliableRDMA::createSharedReceiveQueue() {
  Logging::debug(__FILE__, __LINE__,
                 "ExReliableRDMA::createSharedReceiveQueue: Method Called");

  //TODO XRC use init_attr_ex and ibv_create_srq_ex equivalent to pingpong

  struct ibv_srq_init_attr_ex srq_init_attr_ex;
  sharedrq_t srq;
  memset(&srq_init_attr_ex, 0, sizeof(srq_init_attr_ex));

  srq_init_attr_ex.attr.max_wr = Config::RDMA_MAX_WR;
  srq_init_attr_ex.attr.max_sge = Config::RDMA_MAX_SGE;
  srq_init_attr_ex.comp_mask = IBV_SRQ_INIT_ATTR_TYPE | IBV_SRQ_INIT_ATTR_XRCD |
                               IBV_SRQ_INIT_ATTR_CQ | IBV_SRQ_INIT_ATTR_PD;
  srq_init_attr_ex.srq_type = IBV_SRQT_XRC;
  srq_init_attr_ex.xrcd = xrcd;
  //srq_init_attr_ex.cq = ; // TODO: recv_cq --- is set below
  srq_init_attr_ex.pd = m_res.pd;

  if (!(srq.recv_cq =
            ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1,
                          nullptr, nullptr, 0))) {
    throw runtime_error("Cannot create receive CQ for SRQ!");
  }
  srq_init_attr_ex.cq = srq.recv_cq;

  srq.shared_rq = ibv_create_srq_ex(m_res.ib_ctx, &srq_init_attr_ex);
  if (!srq.shared_rq) {
    throw runtime_error("Error, ibv_create_srq() failed!");
  }

  Logging::debug(__FILE__, __LINE__, "Created shared receive queue");

  //ret_srq_id = m_srqCounter;
  //m_srqs[ret_srq_id] = srq;
  m_srqs[m_srqCounter] = srq;
  m_srqCounter++;
}
 
//------------------------------------------------------------------------------------//
void ExReliableRDMA::initQP(const rdmaConnID rdmaConnID) {
  Logging::debug(
      __FILE__, __LINE__,
      "ExReliableRDMA::initQP: Method Called");
  initQPWithSuppliedID(rdmaConnID);
  //TODO what is the difference between this and ...WithSuppliedID
}
//------------------------------------------------------------------------------------//

void ExReliableRDMA::initQPWithSuppliedID(const rdmaConnID rdmaConnID) {
  Logging::debug(
      __FILE__, __LINE__,
      "ExReliableRDMA::initQPWithSuppliedID: Method Called");
  unsigned int srq_id = 0;
  struct ib_qp_t send_qp;
  struct ib_qp_t recv_qp;

  //createCQ(send_qp->send_cq, recv_qp->recv_cq);
  send_qp.send_cq = this->send_cq;
  send_qp.recv_cq = 0;
  recv_qp.recv_cq = this->recv_cq;
  recv_qp.send_cq = 0;

  // create queues
  createQP(&recv_qp, IBV_QPT_XRC_RECV);
  createQP(&send_qp, IBV_QPT_XRC_SEND);

  // create local connection data
  struct ib_conn_t localConn;
  union ibv_gid my_gid;
  memset(&my_gid, 0, sizeof my_gid);

  uint32_t srq_num; //TODO more dynamic for current srqNum
  if (ibv_get_srq_num(m_srqs[0].shared_rq, &srq_num)) {
    fprintf(stderr, "Couldn't get SRQ num\n");
    //TODO Logging::error(__FILE__, __LINE__, "Couldn't get SRQ num");
    return ;
  }

  localConn.buffer = (uint64_t)m_res.buffer;
  localConn.rc.rkey = m_res.mr->rkey;
  localConn.qp_num = send_qp.qp->qp_num;
  localConn.xrc.recv_qp_num = recv_qp.qp->qp_num;
  localConn.xrc.srqn = srq_num;
  localConn.lid = m_res.port_attr.lid;
  memcpy(localConn.gid, &my_gid, sizeof my_gid);

  // init queue pair
  modifyQPToInit(send_qp.qp);
  modifyQPToInit(recv_qp.qp);

  //TODO XRC create and init both send and recv qp using ib_create_qp_ex
  m_xrc_recv_qps.resize(rdmaConnID + 1);
  m_xrc_recv_qps[rdmaConnID] = recv_qp;

  // done
  setQP(rdmaConnID, send_qp);
  setLocalConnData(rdmaConnID, localConn);
  m_connectedQPs[srq_id].push_back(rdmaConnID);
  Logging::debug(__FILE__, __LINE__, "Created XRC queue pair");
}


////------------------------------------------------------------------------------------//
//
//void ExReliableRDMA::initQPForSRQ(size_t srq_id, rdmaConnID &retRdmaConnID) {
//  retRdmaConnID = nextConnKey();  // qp.qp->qp_num;
//  initQPForSRQWithSuppliedID(srq_id, retRdmaConnID);
//
//  //TODO XRC create and init both send and recv qp using ib_create_qp_ex
//  //TODO XRC is this method required for XRC here?
//
//}

//------------------------------------------------------------------------------------//

//void ExReliableRDMA::createQP(size_t srq_id, struct ib_qp_t &qp) {
//  //TODO XRC probably this method could be erased, how does this method differ to createQP(ib_qp_t&)?
//
//  // initialize QP attributes
//  struct ibv_qp_init_attr qp_init_attr;
//  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
//  memset(&(m_res.device_attr), 0, sizeof(m_res.device_attr));
//  // m_res.device_attr.comp_mask |= IBV_EXP_DEVICE_ATTR_EXT_ATOMIC_ARGS
//  //         | IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS;
//
//  if (ibv_query_device(m_res.ib_ctx, &(m_res.device_attr))) {
//    throw runtime_error("Error, ibv_query_device() failed");
//  }
//
//  // send queue
//  if (!(qp.send_cq = ibv_create_cq(m_res.ib_ctx, Config::RDMA_MAX_WR + 1,
//                                   nullptr, nullptr, 0))) {
//    throw runtime_error("Cannot create send CQ!");
//  }
//
//  qp.recv_cq = m_srqs[srq_id].recv_cq;
//
//  qp_init_attr.send_cq = qp.send_cq;
//  qp_init_attr.recv_cq = m_srqs[srq_id].recv_cq;
//  qp_init_attr.sq_sig_all =
//      0;  // In every WR, it must be decided whether to generate a WC or not
//  qp_init_attr.cap.max_inline_data = Config::MAX_RC_INLINE_SEND;
//
//  // TODO: Enable atomic for DM cluster
//  // qp_init_attr.max_atomic_arg = 32;
//  // qp_init_attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
//  // qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS |
//  // IBV_EXP_QP_INIT_ATTR_PD; qp_init_attr.comp_mask |=
//  // IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;
//
//  qp_init_attr.srq = m_srqs[srq_id].shared_rq;  // Shared receive queue
//  qp_init_attr.qp_type = m_qpType;
//
//  qp_init_attr.cap.max_send_wr = Config::RDMA_MAX_WR;
//  qp_init_attr.cap.max_recv_wr = Config::RDMA_MAX_WR;
//  qp_init_attr.cap.max_send_sge = Config::RDMA_MAX_SGE;
//  qp_init_attr.cap.max_recv_sge = Config::RDMA_MAX_SGE;
//
//  // create queue pair
//  if (!(qp.qp = ibv_create_qp(m_res.pd, &qp_init_attr))) {
//    throw runtime_error("Cannot create queue pair!");
//  }
//}
