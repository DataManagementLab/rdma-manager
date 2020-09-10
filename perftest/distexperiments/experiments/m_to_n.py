import config
from distexprunner import *
from general_memory_experiment import memory_experiment

servers = ['dm-node07', 'dm-node08']
rdma_servers = ['dm-node07', 'dm-node08']
rdma_clients = ['dm-node07', 'dm-node08']


server_list = ServerList(
    [Server(s, config.servers[s]['ip'], config.RUNNER_PORT) for s in servers],
    working_directory='../../build/bin'
)

parameter_grid = ParameterGrid(
    size = [2048],
    transport = ['xrc'],
)


@reg_exp(servers=server_list, params=parameter_grid)
def m_to_n(servers, size, transport):
    memory_experiment(servers, rdma_servers, rdma_clients, size, transport)