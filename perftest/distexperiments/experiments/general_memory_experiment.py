import config
import time
from distexprunner import *

def memory_experiment(servers, rdma_servers, rdma_clients, size, transport, threads=1):
    server_number = 12 if transport=='xrc' else 2
    client_number = 11 if transport=='xrc' else 1
    rdma_server_ip_list = ""
    for s in rdma_servers:
        rdma_server_ip_list += config.servers[s]['ib_ip'] + ","
    
    # TODO: this needs to be modified to be unique per process on a host
    logfile = f'/tmp/tr{transport}-s{size}-t{threads}.csv'

    server_cmd = f'./perf_test -n {server_number} -s {rdma_server_ip_list} -f {logfile}'
    client_cmd = f'./perf_test -n {client_number} -s {rdma_server_ip_list} -d {size} -t {threads} -i 10000 -f {logfile}'
    print("Server cmd: " + server_cmd)
    print("Client cmd: " + client_cmd)

    server_procs = []
    for server_id in rdma_servers:
        args = ""
        if "-1" in server_id:
            args += " -q 1 -e ib1"

        proc = servers[server_id].run_cmd(server_cmd + args,
            stdout=[Console(fmt=f'{server_id}: %s'), File(logfile)])
        server_procs.append(proc)

    client_procs = []
    sentries = []
    for client in rdma_clients:
        args = ""
        if "-1" in client:
            args += " -q 1 -e ib1"

        sentry = SubstrMatcher('Press Enter to run Benchmark!')
        sentries.append(sentry)
        outputs = [sentry, Console(fmt=f'{client}: %s'), File(logfile)]
        proc = servers[client].run_cmd(client_cmd + args, stdout=outputs)
        client_procs.append(proc)

    for sentry in sentries:
        sentry.wait()

    print("All clients are ready")

    for cmd in client_procs:
        cmd.stdin('\n')

    for cmd in client_procs:
        cmd.wait()

    for cmd in server_procs:
        cmd.stdin('stop\n')
    
    time.sleep(2)
    for cmd in server_procs:
        if cmd.wait(block=False) is None:
            cmd.kill()
            # TODO cmd.kill sends SIGKILL, we cannot handle
            cmd.wait()
