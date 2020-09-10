import config
from distexprunner import *


compile_servers = ['dm-node07', 'dm-node08']

server_list = ServerList(
    [Server(s, config.servers[s]['ip'], config.RUNNER_PORT) for s in compile_servers],
    working_directory='../..'
)

@reg_exp(servers=server_list)
def compile(servers):
    cmake_cmd = f'mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..'
    procs = [s.run_cmd(cmake_cmd) for s in servers]
    assert(all(p.wait() == 0 for p in procs))


    make_cmd = f'cd build && make -j'
    procs = [s.run_cmd(make_cmd) for s in servers]
    assert(all(p.wait() == 0 for p in procs))
