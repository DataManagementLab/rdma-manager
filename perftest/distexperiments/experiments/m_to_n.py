import config
from distexprunner import *
from general_memory_experiment import memory_experiment

nodes = ['dm-node0' + str(i) for i in range(1,6)]
servers = nodes + [n+'-1' for n in nodes]
rdma_servers = servers
rdma_clients = servers


server_list = ServerList(
    [Server(s, config.servers[s]['ip'], config.RUNNER_PORT) for s in servers],
    working_directory='../../build/bin'
)

parameter_grid = ParameterGrid(
    size = [256],
    transport = ['xrc', 'rc'],
    threads = [1, 2, 4, 6, 7, 8],
    iterations = [50000000],
    server_count = [1, 4]
)


@reg_exp(servers=server_list, params=parameter_grid)
def m_to_n(servers, size, transport, threads, iterations, server_count):
    memory_experiment(servers, rdma_servers, rdma_clients, "../../results/mn/", size, transport, threads, iterations, server_count)
