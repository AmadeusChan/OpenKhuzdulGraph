#include "vertex_set.h"

#include <string.h>

#include <functional>
#include <algorithm>

#include "memory_pool.h"
#include "dwarves_debug.h"

EdgeId num_comparision = 0;

void print_num_comparision() {
	Debug::get_instance()->log("Num comparision = ", num_comparision);
}

VertexSet::VertexSet(VertexId * vertices_list, VertexId num_vertices): vertices_list_(vertices_list), num_vertices_(num_vertices) {
}

void VertexSet::intersect_with(VertexSet * op, VertexSet * result) {
	if (num_vertices_ == 0 || op->num_vertices_ == 0) {
		result->num_vertices_ = 0;
		return ;
	}
	//num_comparision += num_vertices_;
	//num_comparision += op->num_vertices_;

	VertexId idx_in_op = 0;
	VertexId num_vertices_in_result = 0;
	VertexId * curr_pos_in_result_list = result->vertices_list_;
	VertexId * curr_pos_in_local_list = vertices_list_;
	VertexId * curr_pos_in_op_list = op->vertices_list_;
	VertexId op_num_vertices = op->num_vertices_;
	for (register VertexId v_i = num_vertices_; v_i; -- v_i) { // faster than 0...num_vertices-1
		VertexId v = *(curr_pos_in_local_list ++);
		while (idx_in_op < op_num_vertices && *curr_pos_in_op_list < v) {
			idx_in_op += 1;
			curr_pos_in_op_list += 1;
		}
		if (idx_in_op < op_num_vertices && *curr_pos_in_op_list == v) {
			*curr_pos_in_result_list = v;
			++ num_vertices_in_result;
			++ curr_pos_in_result_list;
			++ curr_pos_in_op_list;
			++ idx_in_op;
		} 
	}
	result->num_vertices_ = num_vertices_in_result;
}

void VertexSet::subtracted_by(VertexSet * op, VertexSet * result) {
	if (num_vertices_ == 0) {
		result->num_vertices_ = 0;
		return ;
	}

	VertexId idx_in_op = 0;
	VertexId num_vertices_in_result = 0;
	VertexId * curr_pos_in_result_list = result->vertices_list_;
	VertexId * curr_pos_in_local_list = vertices_list_;
	VertexId * curr_pos_in_op_list = op->vertices_list_;
	VertexId op_num_vertices = op->num_vertices_;
	for (VertexId v_i = num_vertices_; v_i; -- v_i) {
		VertexId v = *(curr_pos_in_local_list ++);
		bool found_in_op = false;
		while (idx_in_op < op_num_vertices) {
			if (*curr_pos_in_op_list == v) {
				found_in_op = true;
				++ curr_pos_in_op_list;
				++ idx_in_op;
				break;
			}
			if (*curr_pos_in_op_list > v) {
				break;
			}
			++ idx_in_op;
			++ curr_pos_in_op_list;
		}
		if (! found_in_op) {
			*curr_pos_in_result_list = v;
			++ curr_pos_in_result_list;
			++ num_vertices_in_result;
		}
	}
	result->num_vertices_ = num_vertices_in_result;
}

VertexId VertexSet::intersect_and_count(VertexSet * op) {
	if (num_vertices_ == 0 || op->num_vertices_ == 0) {
        return 0;
	}

	VertexId idx_in_op = 0;
	VertexId num_vertices_in_result = 0;
	VertexId * curr_pos_in_local_list = vertices_list_;
	VertexId * curr_pos_in_op_list = op->vertices_list_;
	VertexId op_num_vertices = op->num_vertices_;
	for (register VertexId v_i = num_vertices_; v_i; -- v_i) { // faster than 0...num_vertices-1
		VertexId v = *(curr_pos_in_local_list ++);
		while (idx_in_op < op_num_vertices && *curr_pos_in_op_list < v) {
			idx_in_op += 1;
			curr_pos_in_op_list += 1;
		}
		if (idx_in_op < op_num_vertices && *curr_pos_in_op_list == v) {
			++ num_vertices_in_result;
			++ curr_pos_in_op_list;
			++ idx_in_op;
		} 
	}
	return num_vertices_in_result;
}

void VertexSet::bounded_intersect_with(VertexSet * op, VertexSet * result, VertexId upper_bound) {
	if (num_vertices_ == 0 || op->num_vertices_ == 0) {
		result->num_vertices_ = 0;
		return ;
	}

	VertexId idx_in_op = 0;
	VertexId num_vertices_in_result = 0;
	VertexId * curr_pos_in_result_list = result->vertices_list_;
	VertexId * curr_pos_in_local_list = vertices_list_;
	VertexId * curr_pos_in_op_list = op->vertices_list_;
	VertexId op_num_vertices = op->num_vertices_;

	for (register VertexId v_i = num_vertices_; v_i; -- v_i) { // faster than 0...num_vertices-1
		VertexId v = *(curr_pos_in_local_list ++);
        if (v >= upper_bound) {
            break;
        }
		while (idx_in_op < op_num_vertices && *curr_pos_in_op_list < v) {
			idx_in_op += 1;
			curr_pos_in_op_list += 1;
		}
		if (idx_in_op < op_num_vertices && *curr_pos_in_op_list == v) {
			*curr_pos_in_result_list = v;
			++ num_vertices_in_result;
			++ curr_pos_in_result_list;
			++ curr_pos_in_op_list;
			++ idx_in_op;
		} 
	}
	result->num_vertices_ = num_vertices_in_result;
}

void VertexSet::bounded_subtracted_by(VertexSet * op, VertexSet * result, VertexId upper_bound) {
	if (num_vertices_ == 0) {
		result->num_vertices_ = 0;
		return ;
	}

	VertexId idx_in_op = 0;
	VertexId num_vertices_in_result = 0;
	VertexId * curr_pos_in_result_list = result->vertices_list_;
	VertexId * curr_pos_in_local_list = vertices_list_;
	VertexId * curr_pos_in_op_list = op->vertices_list_;
	VertexId op_num_vertices = op->num_vertices_;
	for (VertexId v_i = num_vertices_; v_i; -- v_i) {
		VertexId v = *(curr_pos_in_local_list ++);
        if (v >= upper_bound) {
            break;
        }
		bool found_in_op = false;
		while (idx_in_op < op_num_vertices) {
			if (*curr_pos_in_op_list == v) {
				found_in_op = true;
				++ curr_pos_in_op_list;
				++ idx_in_op;
				break;
			}
			if (*curr_pos_in_op_list > v) {
				break;
			}
			++ idx_in_op;
			++ curr_pos_in_op_list;
		}
		if (! found_in_op) {
			*curr_pos_in_result_list = v;
			++ curr_pos_in_result_list;
			++ num_vertices_in_result;
		}
	}
	result->num_vertices_ = num_vertices_in_result;
}


