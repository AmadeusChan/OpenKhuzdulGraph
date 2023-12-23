#ifndef UTILITIES_H
#define UTILITIES_H

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <errno.h> 
#include <string.h>
#include <sys/resource.h>

#include <string>
#include <mutex>

#include "types.h"

#define MACHINE_ID_LEN 32

inline long file_size(const std::string &file_name) {
	struct stat st;
	assert(stat(file_name.c_str(), &st) == 0);
	return st.st_size;
}

inline bool file_exits(const std::string &file_name) {
	struct stat st;
	return stat(file_name.c_str(), &st)==0; 
}

void write_file(int f, uint8_t * ptr, long size);
void read_file(int f, uint8_t * ptr, long size);

void process_mem_usage(double& vm_usage, double& resident_set);
void print_mem_usage();
double get_mem_usage();

inline uint64_t get_cycle() {
    uint64_t var;
    uint32_t hi, lo;
    __asm volatile (
	    "mfence\n\t"
	    "lfence\n\t"
	    "rdtsc": "=a" (lo), "=d" (hi)
	    );
    var = ((uint64_t)hi << 32) | lo;
    return var;
}

// buff should have at least MACHINE_ID_LEN + 1 chars
inline void get_machine_id(char * buff) {
    if (! file_exits("/etc/machine-id")) {
        fprintf(stderr, "Cannot find the machine id file: /etc/machine-id\n");
        exit(-1);
    }

    FILE * f = fopen("/etc/machine-id", "r");
    assert(f != NULL);
    fgets(buff, MACHINE_ID_LEN + 1, f);
    assert(fclose(f) == 0);
    assert(strlen(buff) == MACHINE_ID_LEN);
}

inline VertexId trim_v_list(VertexId * v_list, VertexId num_v, VertexId upper_bound) {
    if (num_v == 0 || v_list[0] >= upper_bound) return 0;
    if (v_list[num_v - 1] < upper_bound) return num_v;
    VertexId left = 0, right = num_v;
    while (right - left > 1) {
        VertexId mid = (left + right) >> 1;
        left = v_list[mid] < upper_bound ? mid: left;
        right = v_list[mid] < upper_bound ? right: mid;
    }
    return left + 1;
}

inline VertexId trim_v_list_lower_bound(VertexId * v_list, VertexId num_v, VertexId lower_bound) {
    if (num_v == 0 || v_list[num_v - 1] <= lower_bound) return num_v;
    if (v_list[0] > lower_bound) return 0;
    VertexId left = 0, right = num_v;
    while (right - left > 1) {
        VertexId mid = (left + right) >> 1;
        left = v_list[mid] <= lower_bound ? mid: left;
        right = v_list[mid] <= lower_bound ? right: mid;
    }
    return left + 1;
}

inline void configure_stack_size(size_t size) {
    const rlim_t stack_size = size;
    struct rlimit rl;
    int ret = getrlimit(RLIMIT_STACK, &rl);
    assert(ret == 0);
    if (rl.rlim_cur < stack_size) {
        rl.rlim_cur = stack_size;
        ret = setrlimit(RLIMIT_STACK, &rl);
        assert(ret == 0);
    }
}

//// get the CPU time of the current process
//// unit: ms
//inline double get_cpu_time() {
//    pid_t pid = getpid();
//    char buff[256];
//
//    std::string command = "ps -o time -p " + std::to_string(pid);
//    FILE * pipe = popen(command.c_str(), "r");
//    assert(pipe);
//
//    fscanf(pipe, "%s", buff);
//    buff[255] = '\0';
//    assert(strcmp(buff, "TIME") == 0);
//
//    fscanf(pipe, "%s", buff);
//    printf("************* CPU TIME: '%s'\n", buff);
//
//    pclose(pipe);
//    return 0;
//}

class CPUUtilizationMeasurer {
    private:
        unsigned long long start_cpu_time;
        unsigned long long start_idle_time;
        unsigned long long end_cpu_time;
        unsigned long long end_idle_time;

        void get_times(unsigned long long &cpu_time, unsigned long long &idle_time) {
            FILE * file = fopen("/proc/stat", "r");
            assert(file != NULL);

            unsigned long long user, nice, system, idle, iowait, irq, softirq;
            fscanf(file, "cpu %llu %llu %llu %llu %llu %llu %llu", 
                &user, &nice, &system, &idle, &iowait, &irq, &softirq
            );
            printf("%llu %llu %llu %llu %llu %llu %llu\n",
                user, nice, system, idle, iowait, irq, softirq);
            cpu_time = user + nice + system + idle + iowait + irq + softirq;
            idle_time = idle;

            assert(! fclose(file));
        }

    public:
        void start_measurement() {
            get_times(start_cpu_time, start_idle_time);
        }

        void end_measurement() {
            get_times(end_cpu_time, end_idle_time);
        }

        double get_utilization() {
            unsigned long long cpu_time = end_cpu_time - start_cpu_time;
            unsigned long long idle_time = end_idle_time - start_idle_time;
            return (1. - 1. * idle_time / cpu_time) * 100.;
        }
};

struct PermutationList {
    int num_permutations;
    int length;
    int * permutations;
};

class PermutationGenerator {
    private:
        static const int max_permutation_length = 12;
        static PermutationGenerator * instance;
        static std::mutex instance_mutex;

        volatile bool is_cached_[max_permutation_length];
        PermutationList cached_permutations_[max_permutation_length];
        std::mutex mutex_[max_permutation_length];

        PermutationGenerator() {
            for (int i = 0; i < max_permutation_length; ++ i) {
                is_cached_[i] = false;
                cached_permutations_[i].num_permutations = 0;
                cached_permutations_[i].length = i;
            }
        }
        void dfs(int curr_pos, int length, int &num_found_permutations, bool* is_visited, int* curr_permutation, int* permutation_buffer) {
            if (curr_pos == length) {
                memcpy(permutation_buffer + length * num_found_permutations, curr_permutation, sizeof(int) * length);
                num_found_permutations ++;
                return ;
            }
            for (int i = 0; i < length; ++ i) {
                if (! is_visited[i]) {
                    is_visited[i] = true;
                    curr_permutation[curr_pos] = i;
                    dfs(curr_pos + 1, length, num_found_permutations, is_visited, curr_permutation, permutation_buffer);
                    is_visited[i] = false;
                }
            }
        }
        void get_permutations_by_dfs(int length) {
            bool is_visited[length];
            memset(is_visited, 0, sizeof(is_visited));
            int curr_permutation[length];
            int num_permutations = 1;
            for (int i = 1; i <= length; ++ i) {
                num_permutations *= i;
            }
            int * permutation_buffer = new int [num_permutations * length];
            int num_found_permutations = 0;
            dfs(0, length, num_found_permutations, is_visited, curr_permutation, permutation_buffer);
            assert(num_found_permutations == num_permutations);
            cached_permutations_[length].num_permutations = num_permutations;
            cached_permutations_[length].permutations = permutation_buffer;
        }
    public:
        ~PermutationGenerator() {
            for (int i = 0; i < max_permutation_length; ++ i) {
                if (is_cached_[i]) {
                    delete [] cached_permutations_[i].permutations;
                }
            }
        }
        static inline PermutationGenerator* get_instance() {
            if (instance == nullptr) {
                instance_mutex.lock(); // to make sure that only one thread will enter this section
                if (instance == nullptr) {
                    instance = new PermutationGenerator();
                }
                instance_mutex.unlock();
            }
            return instance;
        }
        inline PermutationList get_permutations(int length) {
            assert(length > 0 && length < max_permutation_length);
            if (is_cached_[length] == true) {
                return cached_permutations_[length];
            }
            mutex_[length].lock();
            if (is_cached_[length] == true) {
                mutex_[length].unlock();
                return cached_permutations_[length];
            } else {
                get_permutations_by_dfs(length);
                is_cached_[length] = true;
                mutex_[length].unlock();
                return cached_permutations_[length];
            }
        }
};

#endif
