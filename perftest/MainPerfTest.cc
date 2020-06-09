#include <stdio.h>

#include <gflags/gflags.h>

DEFINE_bool(server, false, "Act as server for a client to test performance");
DEFINE_bool(gpu, false, "Use GPU memory instead of main memory");
DEFINE_uint64(size, 4096, "Memory size in bytes");
DEFINE_string(nids, "172.18.94.20", "NodeIDSequencer address");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    return 0;
}