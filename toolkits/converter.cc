#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>

#include <string>
#include <algorithm>
#include <iostream>

#include "graph.h"
#include "shared_memory_sys.h"
#include "utilities.h"
#include "graph_loader.h"
#include "dwarves_debug.h"

// we assume that all node label && edge label is a 32-bit unsigned integer
// using this toolkit, you are able to convert the graph dataset into Dwarves graph format (*.dgraph)

VertexId num_vertices;
EdgeId num_edges;
int has_node_label;
int has_edge_label;
int is_directed;

template<typename EdgeLabel>
bool cmp(const EdgeStruct<EdgeLabel> &a, const EdgeStruct<EdgeLabel> &b) {
	if (a.src < b.src) {
		return true;
	} else if (a.src > b.src) {
		return false;
	}
	return a.dst < b.dst;
}

template<typename EdgeLabel> 
void simple_check_edge_list(EdgeStruct<EdgeLabel> * edge_list, EdgeId num_edges) { // check that if there is some self-loop in the given edge list
#pragma omp parallel for 
	for (VertexId v_i = 0; v_i < num_edges; ++ v_i) {
		if (edge_list[v_i].src == edge_list[v_i].dst) {
			printf("** %u: %u %u\n", v_i, edge_list[v_i].src, edge_list[v_i].dst);
		}
		assert(edge_list[v_i].src != edge_list[v_i].dst);
	}
}

template<typename EdgeLabel> 
EdgeStruct<EdgeLabel>* duplicate_edge_list(EdgeStruct<EdgeLabel> * edge_list, EdgeId num_edges) {
	EdgeStruct<EdgeLabel> * new_edge_list = new EdgeStruct<EdgeLabel> [num_edges * 2];
#pragma omp parallel for
	for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
		new_edge_list[e_i] = edge_list[e_i];
		EdgeId e_j = num_edges + e_i;
		new_edge_list[e_j].src = edge_list[e_i].dst;
		new_edge_list[e_j].dst = edge_list[e_i].src;
		if (! std::is_same<EdgeLabel, Empty>::value) {
			new_edge_list[e_j].label = edge_list[e_i].label;
		}
	}
	delete [] edge_list;
	return new_edge_list;
}

template<typename EdgeLabel>
EdgeStruct<EdgeLabel>* sort_dst(EdgeStruct<EdgeLabel> * edge_list, EdgeId num_edges) {
	Debug::get_instance()->enter_function("sort_dst");

	EdgeStruct<EdgeLabel> * sorted_edge_list = new EdgeStruct<EdgeLabel> [num_edges]; 
	EdgeId * pre = new EdgeId[num_vertices + 1];
#pragma omp parallel for 
	for (VertexId v_i = 0; v_i <= num_vertices; ++ v_i) {
		pre[v_i] = 0;
	}
#pragma omp parallel for 
	for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
		EdgeId dst = edge_list[e_i].dst;
		__sync_fetch_and_add(&pre[dst + 1], 1);
	}
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		pre[v_i + 1] += pre[v_i];
	}
#pragma omp parallel for 
	for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
		VertexId dst = edge_list[e_i].dst;
		assert(dst >= 0 && dst < num_vertices);
		EdgeId pos = __sync_fetch_and_add(&pre[dst], 1);
		sorted_edge_list[pos] = edge_list[e_i];
	}
	delete [] edge_list;
	delete [] pre;

	Debug::get_instance()->leave_function("sort_dst");
	return sorted_edge_list;
}

template<typename EdgeLabel>
EdgeStruct<EdgeLabel>* sort_src(EdgeStruct<EdgeLabel> * edge_list, EdgeId num_edges) {
	Debug::get_instance()->enter_function("sort_src");
	
	EdgeStruct<EdgeLabel> * sorted_edge_list = new EdgeStruct<EdgeLabel> [num_edges];
	EdgeId * pre = new EdgeId [num_vertices + 1];

#pragma omp parallel for 
	for (VertexId v_i = 0; v_i <= num_vertices; ++ v_i) {
		pre[v_i] = 0;
	}
#pragma omp parallel for 
	for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
		EdgeId src = edge_list[e_i].src;
		__sync_fetch_and_add(&pre[src + 1], 1);
	}
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		pre[v_i + 1] += pre[v_i];
	}
	for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
		VertexId src = edge_list[e_i].src;
		assert(src >= 0 && src < num_vertices);
		sorted_edge_list[pre[src]] = edge_list[e_i];
		++ pre[src];
	}

	delete [] pre;
	delete [] edge_list;

	Debug::get_instance()->leave_function("sort_src");
	return sorted_edge_list;
}

template<typename EdgeLabel>
void print_edge_list(EdgeStruct<EdgeLabel> * edge_list, EdgeId num_edges) {
	for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
		printf("%lu: %u %u\n", e_i, edge_list[e_i].src, edge_list[e_i].dst);
	}
}

template<typename EdgeLabel> 
void check_if_sorted_correctly(EdgeStruct<EdgeLabel> * edge_list, EdgeId num_edges) {
#pragma omp parallel for 
	for (EdgeId e_i = 1; e_i < num_edges; ++ e_i) {
		assert(edge_list[e_i - 1].src <= edge_list[e_i].src);
		if (edge_list[e_i - 1].src == edge_list[e_i].src) {
			assert(edge_list[e_i - 1].dst < edge_list[e_i].dst);
		}
	}
}

int main(int argc, char ** argv) {
	// there shall not be any duplicated edges or self-loops in the input graph
	if (argc != 7) {
		fprintf(stderr, "usage: ./converter [input file] [output file] [#vertices] [has node label?1:0] [has edge label?1:0] [is directed?1:0]\n");
		exit(-1);
	}

	SharedMemorySys::init_shared_memory_sys(); // to initialize openMP && NUMA

	std::string input_path = argv[1];
	std::string output_path = argv[2];
	num_vertices = atoi(argv[3]);
	has_node_label = atoi(argv[4]);
	has_edge_label = atoi(argv[5]);
	is_directed = atoi(argv[6]);

	assert(sizeof(EdgeStruct<Empty>) == 2 * sizeof(VertexId));
	assert(sizeof(EdgeStruct<VertexId>) == 3 * sizeof(VertexId));
	
	long edge_list_size = file_size(input_path + ".biedgelist");
	if (has_edge_label) {
		num_edges = edge_list_size / sizeof(EdgeStruct<VertexId>);
		assert(edge_list_size % sizeof(EdgeStruct<VertexId>) == 0);
	} else {
		num_edges = edge_list_size / sizeof(EdgeStruct<Empty>);
		assert(edge_list_size % sizeof(EdgeStruct<Empty>) == 0);
	}

	GraphMetaDataStructOnDisk meta_data;
	meta_data.num_vertices = num_vertices;
	meta_data.num_edges = num_edges;
	if (is_directed == 0) {
		meta_data.num_edges *= 2;
	}
	//meta_data.is_directed = is_directed;
	meta_data.has_node_label = has_node_label;
	meta_data.has_edge_label = has_edge_label;

	std::string output_file = output_path + ".dgraph";
	Debug::get_instance()->log("output file: ", output_file);
	int f = open(output_file.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
	assert(f != -1);

	// write the meta data to the file system
	write_file(f, (uint8_t *) &meta_data, sizeof(GraphMetaDataStructOnDisk));

	if (has_node_label) {
		// write node label data
		std::string input_node_label = input_path + ".binodelabel";
		assert((unsigned long)(file_size(input_node_label)) == sizeof(VertexId) * num_vertices);
		int f_input_node_label = open(input_node_label.c_str(), O_RDONLY);
		assert(f_input_node_label != -1);

		VertexId * node_labels = new VertexId [num_vertices];
		read_file(f_input_node_label, (uint8_t *) node_labels, num_vertices * sizeof(VertexId));
		write_file(f, (uint8_t *) node_labels, num_vertices * sizeof(VertexId));
		delete [] node_labels;
		
		assert(close(f_input_node_label) == 0);
	}

	void * edge_list;
	std::string input_edge_list = input_path + ".biedgelist";
	int f_input_edge_list = open(input_edge_list.c_str(), O_RDONLY);
	assert(f_input_edge_list != -1);
	if (has_edge_label) {
		edge_list = (void *) new EdgeStruct<VertexId> [num_edges];
		read_file(f_input_edge_list, (uint8_t*) edge_list, num_edges * sizeof(EdgeStruct<VertexId>));
	} else {
		edge_list = (void *) new EdgeStruct<Empty> [num_edges];
		read_file(f_input_edge_list, (uint8_t *) edge_list, num_edges * sizeof(EdgeStruct<Empty>));
	}
	assert(close(f_input_edge_list) == 0);

	//if (has_edge_label) {
	//	print_edge_list((EdgeStruct<VertexId>*) edge_list, num_edges);
	//} else {
	//	print_edge_list((EdgeStruct<Empty>*) edge_list, num_edges);
	//}

	if (has_edge_label) {
		simple_check_edge_list<VertexId>((EdgeStruct<VertexId>*) edge_list, num_edges);
	} else {
		simple_check_edge_list<Empty>((EdgeStruct<Empty>*) edge_list, num_edges);
	}

	if (is_directed == 0) { // adding reverse edges
		if (has_edge_label) {
			edge_list = (EdgeStruct<VertexId>*) duplicate_edge_list<VertexId>((EdgeStruct<VertexId>*) edge_list, num_edges);
		} else {
			edge_list = (EdgeStruct<Empty>*) duplicate_edge_list<Empty>((EdgeStruct<Empty>*) edge_list, num_edges);
		}
		num_edges *= 2;
	}

	// sorting edge list
	if (has_edge_label) {
		edge_list = (EdgeStruct<VertexId>*) sort_dst<VertexId>((EdgeStruct<VertexId>*) edge_list, num_edges);
		edge_list = (EdgeStruct<VertexId>*) sort_src<VertexId>((EdgeStruct<VertexId>*) edge_list, num_edges);
		//std::sort((EdgeStruct<VertexId>*) edge_list, ((EdgeStruct<VertexId>*)edge_list) + num_edges, cmp<VertexId>);
		check_if_sorted_correctly<VertexId>((EdgeStruct<VertexId>*) edge_list, num_edges);
		//print_edge_list((EdgeStruct<VertexId>*) edge_list, num_edges);
		write_file(f, (uint8_t*) edge_list, sizeof(EdgeStruct<VertexId>) * num_edges);
		delete [] ((EdgeStruct<VertexId>*) edge_list);
	} else {
		edge_list = (EdgeStruct<Empty>*) sort_dst<Empty>((EdgeStruct<Empty>*) edge_list, num_edges);
		edge_list = (EdgeStruct<Empty>*) sort_src<Empty>((EdgeStruct<Empty>*) edge_list, num_edges);
		//std::sort((EdgeStruct<Empty>*) edge_list, ((EdgeStruct<Empty>*)edge_list) + num_edges, cmp<Empty>);
		check_if_sorted_correctly<Empty>((EdgeStruct<Empty>*) edge_list, num_edges);
		//print_edge_list((EdgeStruct<Empty>*) edge_list, num_edges);
		write_file(f, (uint8_t*) edge_list, sizeof(EdgeStruct<Empty>) * num_edges);
		delete [] ((EdgeStruct<Empty>*) edge_list);
	}

	assert(close(f) == 0);

	return 0;
}
