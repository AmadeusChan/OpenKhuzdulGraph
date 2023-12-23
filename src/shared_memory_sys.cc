#include "shared_memory_sys.h"

#include <assert.h>
#include <numa.h>
#include <omp.h>
#include <string.h>

#include <string>

SharedMemorySys * SharedMemorySys::instance_ = nullptr;

SharedMemorySys::SharedMemorySys(int num_threads = -1) {
	assert(numa_available() != -1);
	assert(sizeof(unsigned long) == 8);

    if (num_threads == -1) {
        num_threads_ = numa_num_configured_cpus();
    } else {
        num_threads_ = num_threads;
    }
	num_sockets_ = numa_num_configured_nodes(); //  
    if (num_threads_ < num_sockets_) {
        num_sockets_ = num_threads_;
    }
	num_threads_per_socket_ = num_threads_ / num_sockets_;

    printf("Number of CPU cores: %d, number of sockets: %d\n",
            num_threads_, num_sockets_);

	char nodestring[num_sockets_ * 2 + 1];
	memset(nodestring, 0, sizeof(nodestring));
	nodestring[0] = '0';
	for (int s_i = 1; s_i < num_sockets_; ++ s_i) {
		nodestring[s_i * 2 - 1] = ',';
		nodestring[s_i * 2] = '0' + s_i;
	}
	struct bitmask * nodemask = numa_parse_nodestring(nodestring);
	numa_set_interleave_mask(nodemask);

	omp_set_dynamic(0);
	omp_set_num_threads(num_threads_);
#pragma omp parallel for 
	for (int t_i = 0; t_i < num_threads_; ++ t_i) {
		int s_i = get_socket_id(t_i);
		assert(numa_run_on_node(s_i) == 0);
	}
}

void SharedMemorySys::init_shared_memory_sys(int num_threads) {
	assert(instance_ == nullptr);
	instance_ = new SharedMemorySys(num_threads);
}

SharedMemorySys* SharedMemorySys::get_instance() {
	if (instance_ == nullptr) {
		instance_ = new SharedMemorySys();
	}
	return instance_;
}


