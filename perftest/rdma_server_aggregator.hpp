#include "aggregator.hpp"

template<class T>
class RDMAServerConnAgg : public Aggregator {
  public:
    RDMAServerConnAgg(T* server)
      :server(server) {}
    long read(){ return server->getNumQPs(); }
    std::string getName() { return "qps"; }
    std::string getUnit() { return ""; }
  private:
    T* server;
};
