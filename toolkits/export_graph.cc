#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#include <utility>
#include <functional>
#include <algorithm>

#include "graph.h"
#include "graph_loader.h"
#include "dwarves_debug.h"
#include "shared_memory_sys.h"
#include "timer.h"
#include "memory_pool.h"
#include "vertex_set.h"

bool cmp(EdgeId i, EdgeId j) {
	return i > j;
}

int main(int argc, char ** argv) {
	//Debug::get_instance()->enter_function("main");

	if (argc != 2) {
		Debug::get_instance()->print("usage: export graph [graph dataset]");
		exit(-1);
	}

	SharedMemorySys::init_shared_memory_sys();
	CSRGraph<Empty, Empty> graph;
	std::string graph_file_name = argv[1];
	CSRGraphLoader<Empty, Empty> graph_loader;
	graph_loader.load_graph(graph_file_name, graph);

	VertexId num_vertices = graph.get_num_vertices();
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		VertexSet neighbours = graph.get_neighbour_vertices_set(v_i);
		for (VertexId j = 0; j < neighbours.get_num_vertices(); ++ j) {
			VertexId v_j = neighbours.get_vertex(j);
			printf("%u %u\n", v_i, v_j);
		}
	}

	graph_loader.destroy_graph(graph);
	//Debug::get_instance()->leave_function("main");
	return 0;
}
