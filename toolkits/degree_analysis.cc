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
	Debug::get_instance()->enter_function("main");

	if (argc != 2) {
		Debug::get_instance()->print("usage: degree_analysis [graph dataset]");
		exit(-1);
	}

	SharedMemorySys::init_shared_memory_sys();
	CSRGraph<Empty, Empty> graph;
	std::string graph_file_name = argv[1];
	CSRGraphLoader<Empty, Empty> graph_loader;
	graph_loader.load_graph(graph_file_name, graph);

	VertexId num_vertices = graph.get_num_vertices();
	EdgeId * degree = new EdgeId[num_vertices];
	EdgeId three_star_count = 0;

	EdgeId degree_sum = 0;
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		degree[v_i] = graph.get_degree(v_i);
		degree_sum += (EdgeId) degree[v_i];
		three_star_count += (EdgeId) degree[v_i] * (degree[v_i] - 1) * (degree[v_i] - 2);
	}
	std::sort(degree, degree + num_vertices, cmp);
	Debug::get_instance()->print("max_degree: ", degree[0]);
	assert(three_star_count % 6 == 0);
	Debug::get_instance()->print("three star count (non-induced): ", three_star_count / 6);

	EdgeId partial_sum = 0;
	double bar = 0.1;
	double delta = 0.1;
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		partial_sum += (EdgeId) degree[v_i];
		if (double(v_i + 1) / double(num_vertices) >= bar) {
			Debug::get_instance()->print(bar * 100., "%", " vertices contribute to ", double(partial_sum) / double(degree_sum) * 100., "%", " degree", " degree theshold = ", degree[v_i]);
			bar += delta;
		}
	}

	delete [] degree;
	graph_loader.destroy_graph(graph);
	Debug::get_instance()->leave_function("main");
	return 0;
}
