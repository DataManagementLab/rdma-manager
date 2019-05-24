/**
 * @file RDMAManager.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef RDMAMANAGER_H_
#define RDMAMANAGER_H_

#include "../utils/Config.h"
#include "../proto/ProtoClient.h"

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <unordered_map>
#include <list>
#include <mutex>

namespace rdma {

enum rdma_transport_t {
    rc,
    ud
};

struct ib_resource_t {
    /* Memory region */
    void *buffer; /* memory buffer pointer */
    struct ibv_pd *pd; /* PD handle */
    struct ibv_mr *mr; /* MR handle for buf */

    /* Device attributes */
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr; /* IB port attributes */
    struct ibv_context *ib_ctx; /* device handle */
};

struct ib_qp_t {
    struct ibv_qp *qp; /* Queue pair */
    struct ibv_cq *send_cq; /* Completion Queue */
    struct ibv_cq *recv_cq;

    ib_qp_t()
            : qp(nullptr),
              send_cq(nullptr),
              recv_cq(nullptr) {
    }
};

struct ib_conn_t {
    uint64_t buffer; /*  Buffer address */
    uint64_t qp_num; /*  QP number */
    uint16_t lid; /*  LID of the IB port */
    uint8_t gid[16]; /* GID */

    struct {
        uint32_t rkey; /*  Remote memory key */
    } rc;
    struct {
        uint32_t psn; /* PSN*/
        struct ibv_ah* ah; /* Route to remote QP*/
    } ud;
};

struct ib_addr_t {
    uint64_t conn_key;

    ib_addr_t()
            : conn_key(0) {

    }

    ib_addr_t(uint64_t initConnKey)
            : conn_key(initConnKey) {
    }
};

struct rdma_mem_t {
    size_t size; /* size of memory region */
    bool free;
    size_t offset;
    bool isnull;

    rdma_mem_t(size_t initSize, bool initFree, size_t initOffset)
            : size(initSize),
              free(initFree),
              offset(initOffset),
              isnull(false) {
    }

    rdma_mem_t()
            : size(0),
              free(false),
              offset(0),
              isnull(true) {
    }
};

class RDMAManager {
 public:
    // constructors and destructor
    RDMAManager(size_t mem_size);

    virtual ~RDMAManager() {
    }

    // unicast transfer methods
    virtual bool remoteWrite(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                             size_t size, bool signaled) = 0;
    virtual bool remoteRead(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                            size_t size, bool signaled) = 0;
    virtual bool requestRead(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                             size_t size) = 0;
    virtual bool remoteFetchAndAdd(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                                   size_t size, bool signaled) = 0;
    virtual bool remoteFetchAndAdd(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr, size_t value_to_add,
                                       size_t size, bool signaled) = 0;
    virtual bool remoteCompareAndSwap(struct ib_addr_t& ibAddr, size_t offset, const void* memAddr,
                                      int toCompare, int toSwap, size_t size, bool signaled) = 0;
    virtual bool send(struct ib_addr_t& ibAddr, const void* memAddr, size_t size,
                      bool signaled) = 0;
    virtual bool receive(struct ib_addr_t& ibAddr, const void* memAddr, size_t size) = 0;
    virtual bool pollReceive(struct ib_addr_t& ibAddr, bool doPoll = true) = 0;
    virtual bool pollReceive(struct ib_addr_t& ibAddr, uint32_t& ret_qp_num) {
        (void) ibAddr;
        (void) ret_qp_num;
        return false;
    }

    //srq
    virtual bool receive(size_t srq_id, const void* memAddr, size_t size) {
        (void) srq_id;
        (void) memAddr;
        (void) size;
        return false;
    }
    ;
    virtual bool pollReceive(size_t srq_id, ib_addr_t& ret_qp_num) {
        (void) srq_id;
        (void) ret_qp_num;
        return false;
    }
    ;
    virtual bool createSharedReceiveQueue(size_t& ret_srq_id) {
        (void) ret_srq_id;
        return false;
    }
    ;
    //srq

    virtual bool pollSend(struct ib_addr_t& ibAddr, bool doPoll = true) = 0;

    // unicast connection management
    virtual bool initQP(struct ib_addr_t& reIbAddr, bool isMgmtQP = false) = 0;
    virtual bool initQP(size_t srq_id, struct ib_addr_t& reIbAddr) {
        (void) reIbAddr;
        (void) srq_id;
        return false;
    };

    virtual bool connectQP(struct ib_addr_t& ibAddr) = 0;

    uint64_t getQPNum(const ib_addr_t& ibAddr) {
        return m_qps[ibAddr.conn_key].qp->qp_num;
    }

    ib_conn_t getLocalConnData(struct ib_addr_t& ibAddr) {
        return m_lconns[ibAddr.conn_key];
    }

    ib_conn_t getRemoteConnData(struct ib_addr_t& ibAddr) {
        return m_rconns[ibAddr.conn_key];
    }

    void setRemoteConnData(ib_addr_t& ibAddr, ib_conn_t& conn) {
        size_t connKey = ibAddr.conn_key;
        if (m_rconns.size() < connKey + 1) {
            m_rconns.resize(connKey + 1);
        }
        m_rconns[connKey] = conn;
    }

    // multicast transfer methods
    virtual bool joinMCastGroup(string mCastAddress, struct ib_addr_t& retIbAddr) = 0;
    virtual bool leaveMCastGroup(struct ib_addr_t ibAddr) = 0;
    virtual bool sendMCast(struct ib_addr_t ibAddr, const void* memAddr, size_t size,
                           bool signaled) = 0;
    virtual bool receiveMCast(struct ib_addr_t ibAddr, const void* memAddr, size_t size) = 0;
    virtual bool pollReceiveMCast(struct ib_addr_t ibAddr) = 0;

    // memory management
    virtual void* localAlloc(const size_t& size) = 0;
    virtual bool localFree(const void* ptr) = 0;
    virtual bool localFree(const size_t& offset) = 0;
    virtual rdma_mem_t remoteAlloc(const size_t& size) = 0;
    virtual bool remoteFree(const size_t& offset) = 0;

    void* getBuffer() {
        return m_res.buffer;
    }

    const list<rdma_mem_t> getFreeMemList() const {
        return m_rdmaMem;
    }

    void* getOffsetToPtr(size_t offset) {
        //check if already allocated

            return (void*) ((char*) m_res.buffer + offset);
    }

    size_t getBufferSize() {
        return m_memSize;
    }

    void printBuffer() {
        auto listIter = m_rdmaMem.begin();
        for (; listIter != m_rdmaMem.end(); ++listIter) {
            Logging::debug(
                    __FILE__,
                    __LINE__,
                    "offset=" + to_string((*listIter).offset) + "," + "size="
                            + to_string((*listIter).size) + "," + "free="
                            + to_string((*listIter).free));
        }
        Logging::debug(__FILE__, __LINE__, "---------");
    }

    inline uint64_t getConnKey(const uint64_t& qp_num) {
        return m_qpNum2connKey[qp_num];
    }

    /*
     * Restore Memory from snapshot
     * memory needs to be in a clean state
     * ToDO: faster implementation
     */
    inline bool restoreMemStateSnapShot(const unordered_map<size_t, rdma_mem_t>& usedMemory,
                                        vector<size_t>& sortedOffsets) {

        auto& off = sortedOffsets;
        Logging::debug(__FILE__, __LINE__, "restoreMemStateSnapShot: Snapshot restoring");

        m_usedRdmaMem.insert(usedMemory.begin(), usedMemory.end());
        // restore free state
        m_rdmaMem.clear();
        // loop counter
        size_t i = 0;
        // check begin
        if (off.front() != 0) {
            rdma_mem_t firstFree(off.front(), true, 0);
            m_rdmaMem.push_back(firstFree);
            i = 1;
        }

        // check all elements between 0/1 and last ele
        for (; i < off.size() - 1; i++) {
            auto mem = m_usedRdmaMem.at(off[i]);
            if ((off[i + 1] - (off[i] + mem.size)) > 0) {
                rdma_mem_t free(off[i + 1] - (off[i] + mem.size), true, off[i] + mem.size);
                m_rdmaMem.push_back(free);
            }
        }

        //check  last element
        {
            auto mem = m_usedRdmaMem.at(off.back());
            if ((off.back() + mem.size) != Config::RDMA_MEMSIZE) {
                rdma_mem_t last(Config::RDMA_MEMSIZE - (off.back() + mem.size), true,
                                off.back() + mem.size);
                m_rdmaMem.push_back(last);
            }
        }

        return true;
    }

 protected:
    void destroyManager();
    virtual void destroyQPs()=0;

// memory management
    bool createBuffer();

    inline bool mergeFreeMem(list<rdma_mem_t>::iterator& iter) {
        size_t freeSpace = (*iter).size;
        size_t offset = (*iter).offset;
        size_t size = (*iter).size;

        // start with the prev
        if (iter != m_rdmaMem.begin()) {
            --iter;
            if (iter->offset + iter->size == offset) {
                //increase mem of prev
                freeSpace += iter->size;
                (*iter).size = freeSpace;

                //delete hand-in el
                iter++;
                iter = m_rdmaMem.erase(iter);
                iter--;
            } else {
                //adjust iter to point to hand-in el
                iter++;
            }

        }
        // now check following
        ++iter;
        if (iter != m_rdmaMem.end()) {
            if (offset + size == iter->offset) {
                freeSpace += iter->size;

                //delete following
                iter = m_rdmaMem.erase(iter);

                //go to previous and extend
                --iter;
                (*iter).size = freeSpace;
            }
        }
        Logging::debug(
                __FILE__,
                __LINE__,
                "Merged consecutive free RDMA memory regions, total free space: "
                        + to_string(freeSpace));
        return true;
    }

    inline rdma_mem_t internalAlloc(const size_t& size) {
        auto listIter = m_rdmaMem.begin();
        for (; listIter != m_rdmaMem.end(); ++listIter) {
            rdma_mem_t memRes = *listIter;
            if (memRes.free && memRes.size >= size) {
                rdma_mem_t memResUsed(size, false, memRes.offset);
                m_usedRdmaMem[memRes.offset] = memResUsed;

                if (memRes.size > size) {
                    rdma_mem_t memResFree(memRes.size - size, true, memRes.offset + size);
                    m_rdmaMem.insert(listIter, memResFree);
                }
                m_rdmaMem.erase(listIter);
                //printMem();
                return memResUsed;
            }
        }
        //printMem();
        return rdma_mem_t();  //nullptr
    }

    inline bool internalFree(const size_t& offset) {
        size_t lastOffset = 0;
        rdma_mem_t memResFree = m_usedRdmaMem[offset];
        m_usedRdmaMem.erase(offset);

        // lookup the memory region that was assigned to this pointer
        auto listIter = m_rdmaMem.begin();
        if (listIter != m_rdmaMem.end()) {
            for (; listIter != m_rdmaMem.end(); listIter++) {
                rdma_mem_t& memRes = *(listIter);
                if (lastOffset <= offset && offset < memRes.offset) {
                    memResFree.free = true;
                    m_rdmaMem.insert(listIter, memResFree);
                    listIter--;
                    Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");
                    //printMem();
                    mergeFreeMem(listIter);
                    //printMem();

                    return true;

                }
                lastOffset += memRes.offset;
            }
        } else {
            memResFree.free = true;
            m_rdmaMem.insert(listIter, memResFree);
            Logging::debug(__FILE__, __LINE__, "Freed reserved local memory");
            //printMem();
            return true;

        }
        //printMem();
        return false;
    }

// RDMA operations
    inline uint64_t nextConnKey() {
        return m_lastConnKey++;
    }

    inline void setQP(ib_addr_t& ibAddr, ib_qp_t& qp) {
        size_t connKey = ibAddr.conn_key;
        if (m_qps.size() < connKey + 1) {
            m_qps.resize(connKey + 1);
        }
        m_qps[connKey] = qp;
        m_qpNum2connKey[qp.qp->qp_num] = connKey;
    }

    inline void setLocalConnData(ib_addr_t& ibAddr, ib_conn_t& conn) {
        size_t connKey = ibAddr.conn_key;
        if (m_lconns.size() < connKey + 1) {
            m_lconns.resize(connKey + 1);
        }
        m_lconns[connKey] = conn;
    }

    bool createCQ(ibv_cq*& send_cq, ibv_cq*& rcv_cq);
    bool destroyCQ(ibv_cq*& send_cq, ibv_cq*& rcv_cq);
    virtual bool createQP(struct ib_qp_t *qp) = 0;

    ibv_qp_type m_qpType;
    size_t m_memSize;
    int m_numaRegion;
    int m_rdmaDevice;
    int m_ibPort;
    int m_gidIdx;

    struct ib_resource_t m_res;

//maps ibaddr.conn_keys to connection infos
    size_t m_lastConnKey;
    vector<ib_qp_t> m_qps;  //conn_key is the index of the vector
    vector<ib_conn_t> m_rconns;
    vector<ib_conn_t> m_lconns;

    unordered_map<uint64_t, bool> m_connected;
    unordered_map<uint64_t, uint64_t> m_qpNum2connKey;

    list<rdma_mem_t> m_rdmaMem;
    unordered_map<size_t, rdma_mem_t> m_usedRdmaMem;

    static rdma_mem_t s_nillmem;
}
;

}

#endif /* RDMAMANAGER_H_ */
