import config
from distexprunner import *
from general_memory_experiment import memory_experiment

nodes = ['dm-node0' + str(i) for i in range(1,6)]
servers = nodes + [n+'-1' for n in nodes]
rdma_clients = servers
rdma_servers = ['dm-node02']


server_list = ServerList(
    [Server(s, config.servers[s]['ip'], config.RUNNER_PORT) for s in servers],
    working_directory='../../build/bin'
)

parameter_grid = ParameterGrid(
    size = [256],
    transport = ['xrc', 'rc'],
    threads = [28, 56, 102],
    iterations = [1000000]
)


@reg_exp(servers=server_list, params=parameter_grid)
def m_to_one(servers, size, transport, threads, iterations):
    memory_experiment(servers, rdma_servers, rdma_clients, "../../results/m1/", size, transport, threads, iterations)
