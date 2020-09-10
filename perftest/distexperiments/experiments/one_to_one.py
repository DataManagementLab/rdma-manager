import config
from distexprunner import *

servers = ['dm-node07', 'dm-node08']
rdma_servers = ['dm-node07']
rdma_clients = ['dm-node08']


server_list = ServerList(
    [Server(s, config.servers[s]['ip'], config.RUNNER_PORT) for s in servers],
    working_directory='../../build/bin'
)

parameter_grid = ParameterGrid(
    size = [2048],
    transport = ['xrc'],
)

@reg_exp(servers=server_list, params=parameter_grid)
def one_to_one(servers, size, transport):
    server_number = 12 if transport=='xrc' else 2
    client_number = 11 if transport=='xrc' else 1
    rdma_server_ip_list = ""
    for s in rdma_servers:
        rdma_server_ip_list += config.servers[s]['ib_ip'] + ","

    server_cmd = f'./perf_test -n {server_number} -s {rdma_server_ip_list}'
    client_cmd = f'./perf_test -n {client_number} -s {rdma_server_ip_list} -d {size}'
    print("Server cmd: " + server_cmd)
    print("Client cmd: " + client_cmd)

    server_procs = [servers[server_id].run_cmd(server_cmd) for server_id in rdma_servers]

    client_procs = []
    sentries = []
    for client in rdma_clients:
        sentry = SubstrMatcher('Press Enter to run Benchmark!')
        sentries.append(sentry)
        proc = servers[client].run_cmd(client_cmd, stdout=sentry)
        client_procs.append(proc)

    for sentry in sentries:
        sentry.wait()

    print("All clients are ready")

    for cmd in client_procs:
        cmd.stdin('\n')

    for cmd in client_procs:
        cmd.wait()