#ifndef VERTEX_SET_H
#define VERTEX_SET_H

#include <functional>
#include <iostream>
#include <string>

#include "types.h"
#include "memory_pool.h"
#include "dwarves_debug.h"

void print_num_comparision();

struct VertexSet {
		VertexId * vertices_list_;
		VertexId num_vertices_;

		VertexSet() {}
		VertexSet(VertexId * vertices_list, VertexId num_vertices);

		inline VertexId * get_vertices_list() {
			return vertices_list_;
		}
		inline VertexId get_num_vertices() const {
			return num_vertices_;
		}
		inline VertexId get_vertex(VertexId idx) {
			return vertices_list_[idx];
		}
		inline void set_vertices_list(VertexId * vertices_list) {
			vertices_list_ = vertices_list;
		}
		inline void set_num_vertices(VertexId num_vertices) {
			num_vertices_ = num_vertices;
		}
		inline void set_vertex(VertexId idx, VertexId vtx) {
			vertices_list_[idx] = vtx;
		}

		void intersect_with(VertexSet * op, VertexSet * result);
		void subtracted_by(VertexSet * op, VertexSet * result);
        VertexId intersect_and_count(VertexSet * op);

        void bounded_intersect_with(VertexSet * op, VertexSet * result, VertexId upper_bound);
        void bounded_subtracted_by(VertexSet * op, VertexSet * result, VertexId upper_bound);

		friend std::ostream& operator << (std::ostream &os, const VertexSet &vertex_set) {
			os << "vertex set: ";
			os << "(size: " << vertex_set.num_vertices_ << ") ";
			for (VertexId v_i = 0; v_i < vertex_set.num_vertices_; ++ v_i) {
				os << vertex_set.vertices_list_[v_i] << " ";
			}
			return os;
		}
};

#endif

