

#ifndef SRC_DB_UTILS_RPCVOIDHANDLER_H_
#define SRC_DB_UTILS_RPCVOIDHANDLER_H_



#include "../utils/Config.h"
#include "RPCMemory.h"
#include "../thread/Thread.h"



#include "../rdma/RDMAServer.h"



namespace rdma
{

    template <class RDMA_API_T>
            class RPCVoidHandlerBase : public Thread{
            public:
                virtual bool startHandler() = 0;
                virtual void stopHandler() = 0;
            };

    //todo remove
    template<class RDMA_API_T>
    class RPCVoidHandlerThreadOld : public RPCVoidHandlerBase<RDMA_API_T>
    {

    public:
        RPCVoidHandlerThreadOld(RDMAServer<RDMA_API_T> *rdmaServer, size_t srqID,size_t msgSize,
                         size_t maxNumberMsgs,char* rpcbuffer
                          )
                : m_rdmaServer(rdmaServer),
                  m_srqID(srqID),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory(rpcbuffer, msgSize,maxNumberMsgs),
                  m_freeInClass(false)

        {

            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };

        //constructor without rpcbuffer
        RPCVoidHandlerThreadOld(RDMAServer<RDMA_API_T> *rdmaServer, size_t srqID,size_t msgSize,
                         size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_srqID(srqID),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory((char*)m_rdmaServer->localAlloc(msgSize * maxNumberMsgs), msgSize,maxNumberMsgs),
                  m_freeInClass(true)

        {

            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };

        //constructor without rpcbuffer and without srqID
        RPCVoidHandlerThreadOld(RDMAServer<RDMA_API_T> *rdmaServer,size_t msgSize,
                         size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory((char*)m_rdmaServer->localAlloc(msgSize * maxNumberMsgs), msgSize,maxNumberMsgs),
                  m_freeInClass(true)

        {

            rdmaServer->createSharedReceiveQueue(m_srqID);
            rdmaServer->activateSRQ(m_srqID);
            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };



        ~RPCVoidHandlerThreadOld(){

            //if rpcbuffer is not passed it is created here and needs to be cleaned up
            if(m_freeInClass){
                m_rdmaServer->localFree(m_rpcMemory.bufferAdd());
            }

            m_rdmaServer->localFree(m_intermediateRspBufferVoid);

        };

        size_t getMsgSize(){
            return m_msgSize;
        }



        bool startHandler(){
            if(m_processing){
                return true;
            }
            Thread::start();

            stringstream ss;
            while (!m_processing) {
                if (Thread::killed()) {
                    ss << "RPC handler Thread" << " starting failed  \n";
                    Logging::error(__FILE__, __LINE__, ss.str());
                    return false;
                }
                ss << "RPC handler Thread" << " starting done  \n";
                Logging::debug(__FILE__, __LINE__, ss.str());
                //is this needed
                usleep(Config::RDMA_SLEEP_INTERVAL);
            }
            return  true;

        };


        void stopHandler(){
            stringstream ss;

            if (m_processing) {
                Thread::stop();

                m_poll = false;

                Thread::join();
                m_poll = true;
            }
            ss << "RPC handler Thread" << " stopping done \n";
            Logging::debug(__FILE__, __LINE__, ss.str());

        };


        //init receive calls on rpcMemory
        bool initMemory()
        {   
            //std::cout << "Init receives " << std::endl;
            for (uint32_t i = 0; i < m_maxNumberMsgs; i++)
            {
                auto ptr = m_rpcMemory.getNext();
                m_rdmaServer->receiveSRQ(m_srqID, (void *)ptr, m_msgSize);
                Logging::debug(__FILE__, __LINE__, "initMemory: POTS RECV: " + to_string(i));
            }
            return true;
        }

        void  run() override {
            m_processing = true;
            while (!Thread::killed()) {

                NodeID ibAddr;
                int ret = m_rdmaServer->pollReceiveSRQ(m_srqID, ibAddr,m_poll);
                if(ret){
                    auto message =  m_rpcMemory.getNext();

                    handleRDMARPCVoid(message, ibAddr);

                    m_rdmaServer->receiveSRQ(m_srqID, (void *)message, m_msgSize);
                }/*else{
                    cout << "ret null should not happen except on shutdown" << endl;
                }*/

            }
            m_processing = false;
        }

        //This Message needs to be implemented in subclass to handle the messages
        virtual void handleRDMARPCVoid(void *message, NodeID &returnAdd) =0;

    protected:


        RDMAServer<RDMA_API_T> *m_rdmaServer;


        size_t m_srqID;


        void *m_intermediateRspBufferVoid;



        const size_t m_msgSize;
        uint32_t m_maxNumberMsgs;

        std::atomic<bool> m_processing {false};

        std::atomic<bool> m_poll {true};

        RPCMemory m_rpcMemory;

        const bool m_freeInClass;
    };



    template<class RDMA_API_T>
    class RPCVoidHandlerThread : public RPCVoidHandlerBase<RDMA_API_T>
    {

    public:
        RPCVoidHandlerThread(RDMAServer<RDMA_API_T> *rdmaServer, size_t srqID,size_t msgSize,
                             size_t maxNumberMsgs,char* rpcbuffer
        )
                : m_rdmaServer(rdmaServer),
                  m_srqID(srqID),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcBuffer(rpcbuffer),
                  m_freeInClass(false)

        {

            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };

        //constructor without rpcbuffer
        RPCVoidHandlerThread(RDMAServer<RDMA_API_T> *rdmaServer, size_t srqID,size_t msgSize,
                             size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_srqID(srqID),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcBuffer((char*)m_rdmaServer->localAlloc(msgSize * maxNumberMsgs)),
                  m_freeInClass(true)

        {

            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };

        //constructor without rpcbuffer and without srqID
        RPCVoidHandlerThread(RDMAServer<RDMA_API_T> *rdmaServer,size_t msgSize,
                             size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcBuffer((char*)m_rdmaServer->localAlloc(msgSize * maxNumberMsgs)),
                  m_freeInClass(true)

        {

            rdmaServer->createSharedReceiveQueue(m_srqID);
            rdmaServer->activateSRQ(m_srqID);
            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };



        ~RPCVoidHandlerThread(){

            //if rpcbuffer is not passed it is created here and needs to be cleaned up
            if(m_freeInClass){
                m_rdmaServer->localFree(m_rpcBuffer);
            }

            m_rdmaServer->localFree(m_intermediateRspBufferVoid);

        };

        size_t getMsgSize(){
            return m_msgSize;
        }



        virtual bool startHandler(){
            if(m_processing){
                return true;
            }
            Thread::start();

            stringstream ss;
            while (!m_processing) {
                if (Thread::killed()) {
                    ss << "RPC handler Thread" << " starting failed  \n";
                    Logging::error(__FILE__, __LINE__, ss.str());
                    return false;
                }
                ss << "RPC handler Thread" << " starting done  \n";
                Logging::debug(__FILE__, __LINE__, ss.str());
                //is this needed
                usleep(Config::RDMA_SLEEP_INTERVAL);
            }
            return  true;

        };


        virtual void stopHandler(){
            stringstream ss;

            if (m_processing) {
                Thread::stop();

                m_poll = false;

                Thread::join();
                m_poll = true;
            }
            ss << "RPC handler Thread" << " stopping done \n";
            Logging::debug(__FILE__, __LINE__, ss.str());

        };


        //init receive calls on rpcMemory
        bool initMemory()
        {
            //std::cout << "Init receives " << std::endl;
            for (std::size_t i = 0; i < m_maxNumberMsgs; i++)
            {
                auto ptr = m_rpcBuffer + i * m_msgSize;

                m_rdmaServer->receiveSRQ(m_srqID,i, (void *)ptr, m_msgSize);
                Logging::debug(__FILE__, __LINE__, "initMemory: POTS RECV: " + to_string(i));
            }
            return true;
        }

        void  run() {
            m_processing = true;
            while (!Thread::killed()) {

                NodeID nodeId;
                std::size_t memIndex;
                int ret = m_rdmaServer->pollReceiveSRQ(m_srqID, nodeId,memIndex,m_poll);
                if(ret){
                    auto message = m_rpcBuffer + memIndex * m_msgSize;

                    handleRDMARPCVoid(message, nodeId);

                    m_rdmaServer->receiveSRQ(m_srqID,memIndex, (void *)message, m_msgSize);
                }/*else{
                    cout << "ret null should not happen except on shutdown" << endl;
                }*/

            }
            m_processing = false;
        }

        //This Message needs to be implemented in subclass to handle the messages
        virtual void handleRDMARPCVoid(void *message, NodeID &returnAdd) =0;

    protected:


        RDMAServer<RDMA_API_T> *m_rdmaServer;


        size_t m_srqID;


        void *m_intermediateRspBufferVoid;



        const size_t m_msgSize;
        uint32_t m_maxNumberMsgs;

        char *m_rpcBuffer;

        std::atomic<bool> m_processing {false};

        std::atomic<bool> m_poll {true};



        const bool m_freeInClass;
    };

   

} /* namespace rdma */

#endif /* SRC_DB_UTILS_RPCVOIDHANDLER_H_ */