#ifndef SHARED_MEMORY_SYS_H
#define SHARED_MEMORY_SYS_H

#include <omp.h>

#define MAX_NUM_THREADS 64
#define MAX_NUM_SOCKETS 8

class SharedMemorySys {
	private:
		static SharedMemorySys * instance_;

		int num_threads_;
		int num_sockets_;
		int num_threads_per_socket_;

		SharedMemorySys(int num_threads);
	public:
		static const int num_communication_threads = 2;

		static void init_shared_memory_sys(int num_threads = -1);
		static SharedMemorySys * get_instance();

		inline int get_socket_id(int thread_id) {
			return thread_id / num_threads_per_socket_;
		}
		inline int get_socket_offset(int thread_id) {
			return thread_id % num_threads_per_socket_;
		}
		inline int get_num_sockets() {
			return num_sockets_;
		}
		inline int get_num_threads() {
			return num_threads_;
		}
		inline int get_num_threads_per_socket() {
			return num_threads_per_socket_;
		}

		inline int get_current_thread_id() {
			return omp_get_thread_num();
		}
		inline int get_current_socket_id() {
			return get_socket_id(omp_get_thread_num());
		}
		inline int get_current_socket_offset() {
			return get_socket_offset(omp_get_thread_num());
		}
};

#endif
