#include "distributed_sys.h"

#include <mpi.h>
#include <assert.h>

#include <iostream>
#include <type_traits>

DistributedSys * DistributedSys::instance_ = nullptr;

DistributedSys::DistributedSys() {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id_);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes_);
    char host_name[128];
    int host_name_len;
    MPI_Get_processor_name(host_name, &host_name_len);
    host_name[host_name_len] = 0;
    printf("Nodename: %s\n", host_name);
}

void DistributedSys::init_distributed_sys() {
    assert(instance_ == nullptr);
    instance_ = new DistributedSys();
}

void DistributedSys::finalize_distributed_sys() {
    if (instance_ != nullptr) {
        MPI_Barrier(MPI_COMM_WORLD);
        //printf("Node %d going to finalize the MPI.\n",
        //        instance_->node_id_);
	    MPI_Finalize();
    }
}

DistributedSys * DistributedSys::get_instance() {
    if (instance_ == nullptr) {
	    init_distributed_sys();
    }
    return instance_;
}
