#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>

#define WORD_OFFSET(i) ((i) >> 6)
#define BIT_OFFSET(i) ((i) & 0x3f)

// borrowed from GeminiGraph
class Bitmap {
	private:
		size_t bitmap_size_;
		uint64_t * data_;
	public:
		Bitmap(size_t bitmap_size): bitmap_size_(bitmap_size) {
			data_ = new uint64_t [WORD_OFFSET(bitmap_size) + 1];
			clear();
		}
		~Bitmap() {
			delete [] data_;
		}

		inline void clear() {
			size_t size_in_word = WORD_OFFSET(bitmap_size_);
#pragma omp parallel for 
			for (size_t i = 0; i <= size_in_word; ++ i) {
				data_[i] = 0;
			}
		}
		inline void fill() {
			size_t size_in_word = WORD_OFFSET(bitmap_size_);
#pragma omp parallel for 
			for (size_t i = 0; i <= size_in_word; ++ i) {
				data_[i] = (uint64_t) -1;
			}
		}
		inline uint64_t get_bit(size_t idx) {
			return data_[WORD_OFFSET(idx)] & (((uint64_t) 1) << BIT_OFFSET(idx));
		}
		inline void set_bit(size_t idx) {
			__sync_fetch_and_or(&data_[WORD_OFFSET(idx)], (((uint64_t) 1) << BIT_OFFSET(idx)));
		}
		inline void reset_bit(size_t idx) {
			__sync_fetch_and_and(&data_[WORD_OFFSET(idx)], (~(1ul<<BIT_OFFSET(idx))));
		}
		inline uint64_t fetch_and_set_bit(size_t idx) {
			return __sync_fetch_and_or(&data_[WORD_OFFSET(idx)], (((uint64_t) 1) << BIT_OFFSET(idx))) & (((uint64_t) 1) << BIT_OFFSET(idx));
		}
		// enable could be either 0 or 1
		inline void set_bit(size_t idx, uint64_t enable) {
			__sync_fetch_and_or(
					&data_[WORD_OFFSET(idx)],
					((enable) << BIT_OFFSET(idx))
					);
		}
		inline uint64_t fetch_and_set_bit(size_t idx, uint64_t enable) {
			return __sync_fetch_and_or(
					&data_[WORD_OFFSET(idx)],
					((enable) << BIT_OFFSET(idx))
					) & (((uint64_t) 1) << BIT_OFFSET(idx));
		}
};

#endif
