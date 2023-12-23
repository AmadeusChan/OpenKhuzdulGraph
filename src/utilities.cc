#include "utilities.h"

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <errno.h> 
#include <stdint.h>

#include <string>
#include <fstream>

void write_file(int f, uint8_t * ptr, long size) {
	long total_write_bytes = 0;
	long write_bytes;
	while (total_write_bytes < size) {
		write_bytes = write(f, ptr + total_write_bytes, size - total_write_bytes);
		assert(write_bytes >= 0);
		total_write_bytes += write_bytes;
	}
	assert(total_write_bytes == size);
}

void read_file(int f, uint8_t * ptr, long size) {
	long total_read_bytes = 0;
	long read_bytes;
	while (total_read_bytes < size) {
		read_bytes = read(f, ptr + total_read_bytes, size - total_read_bytes);
		assert(read_bytes >= 0);
		total_read_bytes += read_bytes;
	}
	assert(total_read_bytes == size);
}

// from https://gist.github.com/thirdwing/da4621eb163a886a03c5
void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

void print_mem_usage() {
    double vm, rss;
    process_mem_usage(vm, rss);
    printf("Memory Usage: Virtual Memory: %.4f (GB), Res: %.4f (GB)\n", vm / 1024. / 1024., rss / 1024. / 1024.);
}

// return the resident physical memory usage
// unit: MB
double get_mem_usage() {
    double vm, rss;
    process_mem_usage(vm, rss);
    return rss / 1024.;
}

PermutationGenerator* PermutationGenerator::instance = nullptr;
std::mutex PermutationGenerator::instance_mutex;



