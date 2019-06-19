

#ifndef SRC_DB_UTILS_RPCHANDLER_H_
#define SRC_DB_UTILS_RPCHANDLER_H_



#include "../utils/Config.h"
#include "RPCMemory.h"
#include "../thread/Thread.h"
#include "RPCVoidHandlerThread.h"



#include "../rdma/RDMAServer.h"



namespace rdma
{
    //templated RPCHandler Class
    template <class MessageType>
    class RPCHandlerThread : public RPCVoidHandlerThread
    {

    public:
        RPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs,char* rpcbuffer
                          )
                :RPCVoidHandlerThread(rdmaServer,srqID,sizeof(MessageType),maxNumberMsgs,rpcbuffer)

        {
            m_intermediateRspBuffer = static_cast<MessageType*>(m_intermediateRspBufferVoid);


        };

        //constructor without rpcbuffer
        RPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs
        )
                :RPCVoidHandlerThread(rdmaServer,srqID,sizeof(MessageType),maxNumberMsgs)

        {

            m_intermediateRspBuffer = static_cast<MessageType*>(m_intermediateRspBufferVoid);


        };

        //constructor without rpcbuffer and without srqID
        RPCHandlerThread(RDMAServer *rdmaServer,
                         size_t maxNumberMsgs
        )
                :
                  RPCVoidHandlerThread(rdmaServer,sizeof(MessageType),maxNumberMsgs)

        {
            m_intermediateRspBuffer = static_cast<MessageType*>(m_intermediateRspBufferVoid);

        };


        ~RPCHandlerThread(){


        };




        //This Message needs to be implemented in subclass to handle the messages
        void  handleRDMARPCVoid(void *message, ib_addr_t &returnAdd){
            handleRDMARPC(static_cast<MessageType*>(message),returnAdd);
        }

        void virtual handleRDMARPC(MessageType* message,ib_addr_t & returnAdd) =0;


        protected:

        MessageType *m_intermediateRspBuffer;



    };




   

} /* namespace rdma */

#endif /* SRC_DB_UTILS_RPCHANDLER_H_ */