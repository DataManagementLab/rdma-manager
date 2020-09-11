import config
from distexprunner import *

def memory_experiment(servers, rdma_servers, rdma_clients, size, transport, threads=1):
    server_number = 12 if transport=='xrc' else 2
    client_number = 11 if transport=='xrc' else 1
    rdma_server_ip_list = ""
    for s in rdma_servers:
        rdma_server_ip_list += config.servers[s]['ib_ip'] + ","
    
    # TODO: this needs to be modified to be unique per process on a host
    logfile = f'/tmp/tr{transport}-s{size}-t{threads}.csv'

    server_cmd = f'./perf_test -n {server_number} -s {rdma_server_ip_list} -l {logfile}'
    client_cmd = f'./perf_test -n {client_number} -s {rdma_server_ip_list} -d {size} -t {threads} -l {logfile}'
    print("Server cmd: " + server_cmd)
    print("Client cmd: " + client_cmd)

    server_procs = [servers[server_id].run_cmd(server_cmd) for server_id in rdma_servers]

    client_procs = []
    sentries = []
    for client in rdma_clients:
        sentry = SubstrMatcher('Press Enter to run Benchmark!')
        sentries.append(sentry)
        outputs = [sentry, Console(fmt=f'{client.id}: %s'), File(logfile)]
        proc = servers[client].run_cmd(client_cmd, stdout=outputs)
        client_procs.append(proc)

    for sentry in sentries:
        sentry.wait()

    print("All clients are ready")

    for cmd in client_procs:
        cmd.stdin('\n')

    for cmd in client_procs:
        cmd.wait()
