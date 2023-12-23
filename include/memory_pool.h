#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <assert.h>
#include <stdint.h>

#include <utility>

#include "types.h"

// memory upper bound: |max-degree| x |pattern-size|
class MemoryPool {
	private:
		VertexId * pool_; // could be optimized into numa-aware version
		VertexId max_count_;
		VertexId max_degree_;
		VertexId pattern_size_;
		VertexId * free_stack_;
		VertexId free_stack_top_;
		int socket_id_;
	public:
		MemoryPool(VertexId max_degree, VertexId pattern_size);  // this function is not thread-safe
		~MemoryPool();

		inline std::pair<VertexId*, VertexId> get_new_space() {
			assert(free_stack_top_ > 0);
			VertexId offset = free_stack_[-- free_stack_top_];
			return std::make_pair(pool_ + offset, offset);
		}
		inline void free_space(VertexId offset) {
			free_stack_[free_stack_top_ ++] = offset;
		}
};

class DwarvesMemoryPool {
	private:
		uint8_t * pool_;
		uint64_t block_cnt_;
		uint64_t block_size_; // in bytes
		uint64_t * free_stack_;
		uint64_t free_stack_top_;
		int socket_id_;
	public:
		DwarvesMemoryPool(uint64_t block_cnt, uint64_t block_size);
		~DwarvesMemoryPool();

		inline std::pair<uint8_t*, uint64_t> alloc_new_space() {
			assert(free_stack_top_ > 0);
			uint64_t offset = free_stack_[-- free_stack_top_];
			return std::make_pair(pool_ + offset, offset);
		}
		inline void dealloc_space(uint64_t offset) {
			free_stack_[free_stack_top_ ++] = offset;
		}
};

#endif
