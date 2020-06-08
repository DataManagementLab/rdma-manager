

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
    template <template<class> class VoidHandlerT,class MessageType,typename RDMA_API_T>
    class RPCHandlerThreadTemp : public VoidHandlerT<RDMA_API_T>
    {

    public:
        RPCHandlerThreadTemp(RDMAServer<ReliableRDMA> *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs,char* rpcbuffer
                          )
                :VoidHandlerT<RDMA_API_T>(rdmaServer,srqID,sizeof(MessageType),maxNumberMsgs,rpcbuffer)

        {
            m_intermediateRspBuffer = static_cast<MessageType*>(VoidHandlerT<RDMA_API_T>::m_intermediateRspBufferVoid);


        };

        //constructor without rpcbuffer
        RPCHandlerThreadTemp(RDMAServer<RDMA_API_T> *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs
        )
                :VoidHandlerT<RDMA_API_T>(rdmaServer,srqID,sizeof(MessageType),maxNumberMsgs)

        {

            m_intermediateRspBuffer = static_cast<MessageType*>(VoidHandlerT<RDMA_API_T>::m_intermediateRspBufferVoid);


        };

        //constructor without rpcbuffer and without srqID
        RPCHandlerThreadTemp(RDMAServer<RDMA_API_T> *rdmaServer,
                         size_t maxNumberMsgs
        )
                :VoidHandlerT<RDMA_API_T>(rdmaServer,sizeof(MessageType),maxNumberMsgs)

        {
            m_intermediateRspBuffer = static_cast<MessageType*>(VoidHandlerT<RDMA_API_T>::m_intermediateRspBufferVoid);

        };

        ~RPCHandlerThreadTemp(){


        };

        void  handleRDMARPCVoid(void *message, NodeID &returnAdd){
            handleRDMARPC(static_cast<MessageType*>(message),returnAdd);
        }

        //This Message needs to be implemented in subclass to handle the messages
        virtual void  handleRDMARPC(MessageType* message,NodeID & returnAdd) =0;


        protected:

        MessageType *m_intermediateRspBuffer;


    };



    template <class MessageType,typename RDMA_API_T>
    class RPCHandlerThread : public RPCHandlerThreadTemp<RPCVoidHandlerThread,MessageType,RDMA_API_T>
            {
            using RPCHandlerThreadTemp<RPCVoidHandlerThread,MessageType,RDMA_API_T>::RPCHandlerThreadTemp;
            };

    //todo remove
    template <class MessageType,typename RDMA_API_T>
    class RPCHandlerThreadOld : public RPCHandlerThreadTemp<RPCVoidHandlerThreadOld,MessageType,RDMA_API_T>
    {
        using RPCHandlerThreadTemp<RPCVoidHandlerThreadOld,MessageType,RDMA_API_T>::RPCHandlerThreadTemp;
    };
   

} /* namespace rdma */

#endif /* SRC_DB_UTILS_RPCHANDLER_H_ */