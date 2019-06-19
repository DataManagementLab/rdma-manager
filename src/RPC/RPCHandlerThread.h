

#ifndef SRC_DB_UTILS_RPCHANDLER_H_
#define SRC_DB_UTILS_RPCHANDLER_H_



#include "../utils/Config.h"
#include "RPCMemory.h"
#include "../thread/Thread.h"



#include "../rdma/RDMAServer.h"



namespace rdma
{
    template <class MessageType>
    class RPCHandlerThread : public Thread
    {

    public:
        RPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs,char* rpcbuffer
                          )
                : m_rdmaServer(rdmaServer),
                  m_srqID(srqID),
                  m_msgSize(sizeof(MessageType)),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory(rpcbuffer, sizeof(MessageType),maxNumberMsgs),
                  m_freeInClass(false)

        {

            m_intermediateRspBuffer = (MessageType *)m_rdmaServer->localAlloc(sizeof(MessageType));
            initMemory();


        };

        //constructor without rpcbuffer
        RPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_srqID(srqID),
                  m_msgSize(sizeof(MessageType)),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory((char*)m_rdmaServer->localAlloc(sizeof(MessageType) * maxNumberMsgs), sizeof(MessageType),maxNumberMsgs),
                  m_freeInClass(true)

        {

            m_intermediateRspBuffer = (MessageType *)m_rdmaServer->localAlloc(sizeof(MessageType));
            initMemory();


        };

        //constructor without rpcbuffer and without srqID
        RPCHandlerThread(RDMAServer *rdmaServer,
                         size_t maxNumberMsgs
        )
                : m_rdmaServer(rdmaServer),
                  m_msgSize(sizeof(MessageType)),
                  m_maxNumberMsgs(maxNumberMsgs),
                  m_rpcMemory((char*)m_rdmaServer->localAlloc(sizeof(MessageType) * maxNumberMsgs), sizeof(MessageType),maxNumberMsgs),
                  m_freeInClass(true)

        {

            rdmaServer->createSRQ(m_srqID);
            rdmaServer->activateSRQ(m_srqID);
            m_intermediateRspBuffer = (MessageType *)m_rdmaServer->localAlloc(sizeof(MessageType));
            initMemory();


        };



        ~RPCHandlerThread(){

            //if rpcbuffer is not passed it is created here and needs to be cleaned up
            if(m_freeInClass){
                m_rdmaServer->localFree(m_rpcMemory.bufferAdd());
            }

            m_rdmaServer->localFree(m_intermediateRspBuffer);

        };



        virtual bool startServer(){
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

        virtual void stopServer(){
            stringstream ss;

            if (m_processing) {
                m_processing = false;

                stop();


                //locks right now
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
                m_rdmaServer->receive(m_srqID, (void *)ptr, sizeof(MessageType));
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
                    auto message = (MessageType *) m_rpcMemory.getNext();
                    handleRDMARPC(message, ibAddr);
                    m_rdmaServer->receive(m_srqID, (void *)message, sizeof(MessageType));

                }


            }
        }

        //This Message needs to be implemented in subclass to handle the messages
        void virtual handleRDMARPC(MessageType *message, ib_addr_t &returnAdd) =0;

    protected:


        RDMAServer *m_rdmaServer;


        size_t m_srqID;


        MessageType *m_intermediateRspBuffer;



        const size_t m_msgSize;
        uint32_t m_maxNumberMsgs;

        bool m_processing;

        RPCMemory m_rpcMemory;

        const bool m_freeInClass;
    };

   

} /* namespace rdma */

#endif /* SRC_DB_UTILS_RPCHANDLER_H_ */