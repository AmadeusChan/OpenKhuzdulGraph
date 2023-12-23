#include "memory_pool.h"

#include <numa.h>
#include <sys/mman.h>

#include "dwarves_debug.h"
#include "shared_memory_sys.h"

MemoryPool::MemoryPool(VertexId max_degree, VertexId pattern_size):
	max_degree_(max_degree), pattern_size_(pattern_size) {
		//Debug::get_instance()->enter_function("MemoryPool::MemoryPool");
		//Debug::get_instance()->log("max_degree: ", max_degree, " pattern_size: ", pattern_size);
		socket_id_ = SharedMemorySys::get_instance()->get_current_socket_id();
		//max_count_ = max_degree * pattern_size * 2; 
		max_count_ = max_degree * pattern_size * (pattern_size - 1);  // TODO may not precise
		//pool_ = (VertexId*) numa_alloc_onnode(sizeof(VertexId) * max_count_, socket_id_);
		pool_ = new VertexId[max_count_];
		//madvise(pool_, sizeof(VertexId) * max_count_, MADV_HUGEPAGE);
		free_stack_top_ = 0;
		free_stack_ = new VertexId[max_count_];
		//madvise(free_stack_, sizeof(VertexId) * max_count_, MADV_HUGEPAGE);
		//free_stack_ = (VertexId*) numa_alloc_onnode(sizeof(VertexId) * max_count_, socket_id_);
		for (VertexId i = 0; i < pattern_size * 2; ++ i) {
			free_stack_[free_stack_top_ ++] = i * max_degree_;
		}
		//Debug::get_instance()->leave_function("MemoryPool::MemoryPool");
	}

MemoryPool::~MemoryPool() {
	//numa_free((void*) pool_, sizeof(VertexId) * max_count_);
	delete [] pool_;
	delete [] free_stack_;
	//numa_free((void*) free_stack_, sizeof(VertexId) * max_count_);
}

DwarvesMemoryPool::DwarvesMemoryPool(uint64_t block_cnt, uint64_t block_size): 
	block_cnt_(block_cnt), block_size_(block_size) {
		socket_id_ = SharedMemorySys::get_instance()->get_current_socket_id();
		uint64_t total_cnt = block_cnt * block_size;
		pool_ = new uint8_t [total_cnt];
		//mlock(pool_, total_cnt);
		free_stack_top_ = 0;
		free_stack_ = new uint64_t [block_cnt];
		for (uint64_t i = block_cnt; i > 0; -- i) {
			free_stack_[free_stack_top_ ++] = block_size * (i - 1);
		}
	}

DwarvesMemoryPool::~DwarvesMemoryPool() {
	delete [] pool_;
	delete [] free_stack_;
}
