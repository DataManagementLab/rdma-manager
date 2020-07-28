/**
 * @file ReliableRDMA.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */

#ifndef ReliableRDMA_H_
#define ReliableRDMA_H_

#include "../utils/Config.h"
#include <atomic>
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
  ReliableRDMA(size_t mem_size, bool huge);
  ReliableRDMA(BaseMemory *buffer);
  ~ReliableRDMA();

  void initQPWithSuppliedID(const rdmaConnID suppliedID) override;
  void initQPWithSuppliedID( struct ib_qp_t** qp ,struct ib_conn_t ** localConn) ;

  void initQP(rdmaConnID& retRdmaConnID) override;
  void connectQP(const rdmaConnID rdmaConnID) override;
  void disconnectQP(const rdmaConnID rdmaConnID) override;

  /* Function: write
   * ----------------
   * Writes data from a given array to the remote side
   * 
   * rdmaConnID:  id of the remote
   * offset:      offset on the remote side where to start writing
   * memAddr:     address of the local array that should be transfered
   * size:        how many bytes should be transfered
   * signaled:    if true the function blocks until the write request was
   *              processed by the NIC. Multiple writes can be called and 
   *              the last one should always be signaled=true.
   *              At max Config::RDMA_MAX_WR writes can be performed at once 
   *              without signaled=true.
   * 
   */
  void write(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
             size_t size, bool signaled);

  /* Function: writeImm
   * ----------------
   * Writes data from a given array to the remote side. 
   * In addition notifies receiver via the CompletionQueue with a value.
   * Therefore the remote must first post a receive() and then pollReceive() 
   * to detect the writeImm().
   * 
   * rdmaConnID:  id of the remote
   * offset:      offset on the remote side where to start writing
   * memAddr:     address of the local array that should be transfered
   * size:        how many bytes should be transfered
   * imm:         immediate value that receiver can retriev with pollReceive()
   * signaled:    if true the function blocks until the write request was
   *              processed by the NIC. Multiple writes can be called and 
   *              the last one should always be signaled=true.
   *              At max Config::RDMA_MAX_WR writes can be performed at once 
   *              without signaled=true.
   */
  void writeImm(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
             size_t size, uint32_t imm, bool signaled);

  /* Function: read
   * ----------------
   * Reads data from the remote side into a given array
   * 
   * rdmaConnID:  id of the remote
   * offset:      offset on the remote side where to start reading
   * memAddr:     address of local array where the data should be stored
   * size:        how many bytes should be transfered
   * signaled:    if true the function blocks until the read has fully been 
   *              completed. Multiple reads can be called and 
   *              the last one should always be signaled=true. 
   *              At max Config::RDMA_MAX_WR reads can be performed at once 
   *              without signaled=true.
   * 
   */
  void read(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
            size_t size, bool signaled);

  void requestRead(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, size_t size);
 
  /* Function: fetchAndAdd
   * ----------------
   * Fetches and increments one as an atomic operation
   * 
   * rdmaConnID:  id of the remote
   * offset:      offset on the remote side where to fetch and add
   * memAddr:     local address where the fetched (old) value should be stored
   * size:        size of the value that should be transfered (only 8bytes = 64bit makes sense)
   * signaled:    if true the function blocks until the fetch has fully been 
   *              completed. Multiple fetches can be called and 
   *              the last one should always be signaled=true. 
   *              At max Config::RDMA_MAX_WR fetches can be performed at once 
   *              without signaled=true.
   * 
   */                 
  void fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, size_t size, bool signaled);
  
  /* Function: fetchAndAdd
  * ----------------
  * Fetches and increments one as an atomic operation
  * 
  * rdmaConnID:  id of the remote
  * offset:      offset on the remote side where to fetch and add
  * memAddr:     local address where the fetched (old) value should be stored
  * signaled:    if true the function blocks until the fetch has fully been 
  *              completed. Multiple fetches can be called and 
  *              the last one should always be signaled=true. 
  *              At max Config::RDMA_MAX_WR fetches can be performed at once 
  *              without signaled=true.
  */
  void fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                  const void* memAddr, bool signaled){
    fetchAndAdd(rdmaConnID, offset, memAddr, sizeof(uint64_t), signaled);
  }

  /* Function: fetchAndAdd
   * ----------------
   * Fetches and adds a 64bit value as an atomic operation
   * 
   * rdmaConnID:   id of the remote
   * offset:       offset on the remote side where to fetch and add
   * memAddr:      local address where the fetched (old) value should be stored
   * value_to_add: value that should be added
   * size:         size of the value that should be transfered (only 8bytes = 64bit makes sense)
   * signaled:     if true the function blocks until the fetch has fully been 
   *               completed. Multiple fetches can be called and 
   *               the last one should always be signaled=true. 
   *               At max Config::RDMA_MAX_WR fetches can be performed at once 
   *               without signaled=true.
   */ 
  void fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, size_t value_to_add, size_t size,
                   bool signaled);

  /* Function: fetchAndAdd
   * ----------------
   * Fetches and adds a 64bit value as an atomic operation
   * 
   * rdmaConnID:    id of the remote
   * offset:        offset on the remote side where to fetch and add
   * memAddr:       local address where the fetched (old) value should be stored
   * value_to_add:  value that should be added
   * signaled:      if true the function blocks until the fetch has fully been 
   *                completed. Multiple fetches can be called and 
   *                the last one should always be signaled=true. 
   *                At max Config::RDMA_MAX_WR fetches can be performed at once 
   *                without signaled=true.
   */
  void fetchAndAdd(const rdmaConnID rdmaConnID, size_t offset,
                   const void* memAddr, int64_t value_to_add, bool signaled){
    fetchAndAdd(rdmaConnID, offset, memAddr, value_to_add, sizeof(uint64_t), signaled);
  }

  /* Function: compareAndSwap
   * ----------------
   * Compares and swaps a 64bit value as an atomic operation
   * 
   * rdmaConnID:  id of the remote
   * offset:      offset on the remote side where to fetch and add
   * memAddr:     local address where the original (old) value 
   *              should be stored. Always original value, regardless 
   *              of whether an swap has taken place or not
   * toCompare:   64bit value that should be compared with remote value
   * toSwap:      64bit that should be set on remote if comparison succeeded
   * size:        size of the value that should be transfered (only 8bytes = 64bit makes sense)
   * signaled:    if true the function blocks until the fetch has fully been 
   *              completed. Multiple fetches can be called and 
   *              the last one should always be signaled=true. 
   *              At max Config::RDMA_MAX_WR fetches can be performed at once 
   *              without signaled=true.
   */
  void compareAndSwap(const rdmaConnID rdmaConnID, size_t offset,
                      const void* memAddr, int toCompare, int toSwap,
                      size_t size, bool signaled);

  /* Function: compareAndSwap
   * ----------------
   * Compares and swaps a 64bit value as an atomic operation
   * 
   * rdmaConnID:  id of the remote
   * offset:      offset on the remote side where to fetch and add
   * memAddr:     local address where the original (old) value 
   *              should be stored. Always original value, regardless 
   *              of whether an swap has taken place or not
   * toCompare:   64bit value that should be compared with remote value
   * toSwap:      64bit that should be set on remote if comparison succeeded
   * signaled:    if true the function blocks until the fetch has fully been 
   *              completed. Multiple fetches can be called and 
   *              the last one should always be signaled=true. 
   *              At max Config::RDMA_MAX_WR fetches can be performed at once 
   *              without signaled=true.
   */
  void compareAndSwap(const rdmaConnID rdmaConnID, size_t offset,
                      const void* memAddr, int toCompare, int toSwap, bool signaled){
    compareAndSwap(rdmaConnID, offset, memAddr, toCompare, toSwap, sizeof(int64_t), signaled);
  }

  void send(const rdmaConnID rdmaConnID, const void* memAddr, size_t size,
            bool signaled) override;
  
  void receive(const rdmaConnID rdmaConnID, const void* memAddr,
               size_t size) override;
  
  int pollReceive(const rdmaConnID rdmaConnID, bool doPoll = true,uint32_t* = nullptr) override;

  void pollReceiveBatch(size_t srq_id, size_t& num_completed, bool& doPoll);
  void pollSend(const rdmaConnID rdmaConnID, bool doPoll) override;

  void* localAlloc(const size_t& size) override;
  void localFree(const void* ptr) override;
  void localFree(const size_t& offset) override;
  
  // Shared Receive Queue
  void initQPForSRQWithSuppliedID(size_t srq_id, const rdmaConnID rdmaConnID);
  void initQPForSRQ(size_t srq_id, rdmaConnID& retRdmaConnID);

  //remove these
  void receiveSRQ(size_t srq_id, const void* memAddr, size_t size);
  void pollReceiveSRQ(size_t srq_id, rdmaConnID& retrdmaConnID, bool& doPoll);
  int pollReceiveSRQ(size_t srq_id, rdmaConnID& retrdmaConnID, std::atomic<bool> & doPoll);
  int pollReceiveSRQ(size_t srq_id, rdmaConnID &retRdmaConnID, uint32_t *imm, atomic<bool> &doPoll);


  void receiveSRQ(size_t srq_id, size_t memoryIndex ,const void* memAddr, size_t size);
  int pollReceiveSRQ(size_t srq_id, rdmaConnID& retrdmaConnID, size_t& retMemoryIdx, std::atomic<bool>& doPoll);
  int pollReceiveSRQ(size_t srq_id, rdmaConnID& retrdmaConnID, size_t& retMemoryIdx, uint32_t *imm, std::atomic<bool>& doPoll);

  void createSharedReceiveQueue(size_t& ret_srq_id);

 protected:
  // RDMA operations
  inline void __attribute__((always_inline))
  remoteAccess(const rdmaConnID rdmaConnID, size_t offset, const void* memAddr,
               size_t size, bool signaled, bool wait, enum ibv_wr_opcode verb,uint32_t * imm = nullptr) {
    DebugCode(
      if (memAddr < m_buffer->pointer() || (char*)memAddr + size > (char*)m_buffer->pointer() + m_buffer->ib_mr()->length) {
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
    sge.lkey = m_buffer->ib_mr()->lkey;
    sge.length = size;
    memset(&sr, 0, sizeof(sr));
    sr.sg_list = &sge;
    sr.num_sge = 1;
    sr.opcode = verb;
    sr.next = nullptr;
    sr.send_flags = ((signaled) ? IBV_SEND_SIGNALED : 0) | (size < Config::MAX_RC_INLINE_SEND && verb == IBV_WR_RDMA_WRITE ? IBV_SEND_INLINE: 0);

    // calculate remote address using offset in local buffer
    sr.wr.rdma.remote_addr = remoteConn.buffer + offset;
    sr.wr.rdma.rkey = remoteConn.rc.rkey;

    if(imm!= nullptr)
        sr.imm_data = *imm;

    struct ibv_send_wr* bad_wr = nullptr;
    if ((errno = ibv_post_send(localQP.qp, &sr, &bad_wr))) {
      throw runtime_error("RDMA OP not successful! error: " + to_string(errno));
    }

    if (signaled && wait) {
      struct ibv_wc wc;

      do {
        wc.status = IBV_WC_SUCCESS;
        ne = ibv_poll_cq(localQP.send_cq, 1, &wc);
        if (wc.status != IBV_WC_SUCCESS) {
          throw runtime_error("RDMA completion event in CQ with error in remoteAccess()! " +
                             to_string(wc.status));
        }

#ifdef BACKOFF
        if (ne == 0) {
          __asm__("pause");
        }
#endif
      } while (ne == 0);

      if (ne < 0) {
        throw runtime_error("RDMA polling from CQ failed!");
      }
    }
  }

  virtual void destroyQPs() override;
  void createQP(struct ib_qp_t* qp) override;
  void createQP(size_t srq_id, struct ib_qp_t& qp);
  void modifyQPToInit(struct ibv_qp* qp);
  void modifyQPToRTR(struct ibv_qp* qp, uint32_t remote_qpn, uint16_t dlid,
                     uint8_t* dgid);
  void modifyQPToRTS(struct ibv_qp* qp);

  // shared receive queues
  map<size_t, sharedrq_t> m_srqs;
  size_t m_srqCounter = 0;
  map<size_t, vector<rdmaConnID>> m_connectedQPs;


};

}  // namespace rdma

#endif /* ReliableRDMA_H_ */
