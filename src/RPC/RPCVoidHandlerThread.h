

#ifndef SRC_DB_UTILS_RPCVOIDHANDLER_H_
#define SRC_DB_UTILS_RPCVOIDHANDLER_H_



#include "../utils/Config.h"
#include "RPCMemory.h"
#include "../thread/Thread.h"



#include "../rdma/RDMAServer.h"



namespace rdma
{
    class RPCVoidHandlerThread : public Thread
    {

    public:
        RPCVoidHandlerThread(RDMAServer *rdmaServer, size_t srqID,size_t msgSize,
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
        RPCVoidHandlerThread(RDMAServer *rdmaServer, size_t srqID,size_t msgSize,
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
        RPCVoidHandlerThread(RDMAServer *rdmaServer,size_t msgSize,
                         size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_msgSize(msgSize),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory((char*)m_rdmaServer->localAlloc(msgSize * maxNumberMsgs), msgSize,maxNumberMsgs),
                  m_freeInClass(true)

        {

            rdmaServer->createSRQ(m_srqID);
            rdmaServer->activateSRQ(m_srqID);
            m_intermediateRspBufferVoid = m_rdmaServer->localAlloc(m_msgSize);
            initMemory();


        };



        ~RPCVoidHandlerThread(){

            //if rpcbuffer is not passed it is created here and needs to be cleaned up
            if(m_freeInClass){
                m_rdmaServer->localFree(m_rpcMemory.bufferAdd());
            }

            m_rdmaServer->localFree(m_intermediateRspBufferVoid);

        };

        size_t getMsgSize(){
            return m_msgSize;
        }


        //todo umbenenen
        virtual bool startHandler(){
            start();

            stringstream ss;
            while (!m_processing) {
                if (killed()) {
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

        //todo umbenenen
        virtual void stopHandler(){
            stringstream ss;

            if (m_processing) {
                m_processing = false;

                stop();



                join();
            }
            ss << "RPC handler Thread" << " stopping done \n";
            Logging::debug(__FILE__, __LINE__, ss.str());

        };


        //init receive calls on rpcMemory
        bool initMemory()
        {
            for (uint32_t i = 0; i < m_maxNumberMsgs; i++)
            {
                auto ptr = m_rpcMemory.getNext();
                m_rdmaServer->receive(m_srqID, (void *)ptr, m_msgSize);
                Logging::debug(__FILE__, __LINE__, "initMemory: POTS RECV: " + to_string(i));
            }
            return true;
        }

        void  run() {
            m_processing = true;
            while (m_processing && !killed()) {

                //todo nodeId
                ib_addr_t ibAddr;
                auto ret = m_rdmaServer->pollReceive(m_srqID, ibAddr,m_processing);

                if (ret)
                {
                    auto message =  m_rpcMemory.getNext();
                    handleRDMARPCVoid(message, ibAddr);
                    m_rdmaServer->receive(m_srqID, (void *)message, m_msgSize);

                }


            }
        }

        //This Message needs to be implemented in subclass to handle the messages
        void virtual handleRDMARPCVoid(void *message, ib_addr_t &returnAdd) =0;

    protected:


        RDMAServer *m_rdmaServer;


        size_t m_srqID;


        void *m_intermediateRspBufferVoid;



        const size_t m_msgSize;
        uint32_t m_maxNumberMsgs;

        bool m_processing;

        RPCMemory m_rpcMemory;

        const bool m_freeInClass;
    };

   

} /* namespace rdma */

#endif /* SRC_DB_UTILS_RPCVOIDHANDLER_H_ */