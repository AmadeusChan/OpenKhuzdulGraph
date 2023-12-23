#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#include <utility>
#include <functional>
#include <algorithm>
#include <set>
#include <vector>

#include "graph.h"
#include "graph_loader.h"
#include "dwarves_debug.h"
#include "shared_memory_sys.h"
#include "timer.h"
#include "memory_pool.h"
#include "vertex_set.h"

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
	EdgeId num_edges = graph.get_num_edges();
	EdgeId * degree = new EdgeId[num_vertices];

	EdgeId degree_sum = 0;
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		degree[v_i] = graph.get_degree(v_i);
		degree_sum += (EdgeId) degree[v_i];
	}
	Debug::get_instance()->print("num_vertices = ", num_vertices);
	Debug::get_instance()->print("num_edges = ", num_edges);
	Debug::get_instance()->print("degree sum = ", degree_sum);
	assert(num_edges == degree_sum);;

	std::vector<std::pair<VertexId, VertexId>> all_edges;
	std::set<std::pair<VertexId, VertexId>> all_edges_set;
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		VertexSet N_v_i = graph.get_neighbour_vertices_set(v_i);
		VertexId size_N_v_i = N_v_i.get_num_vertices();
		assert(size_N_v_i == degree[v_i]);
		for (VertexId i = 1; i < size_N_v_i; ++ i) {
			assert(N_v_i.get_vertex(i) > N_v_i.get_vertex(i - 1));
		}
		for (VertexId i = 0; i < size_N_v_i; ++ i) {
			assert(N_v_i.get_vertex(i) != v_i); // no self loops
			all_edges.push_back(std::make_pair(v_i, N_v_i.get_vertex(i)));
			all_edges_set.insert(std::make_pair(v_i, N_v_i.get_vertex(i)));
		}
	}
	for (std::pair<VertexId, VertexId> edge: all_edges) {
		std::pair<VertexId, VertexId> inverse_edge = std::make_pair(edge.second, edge.first);
		assert(all_edges_set.find(inverse_edge) != all_edges_set.end());
	}

	delete [] degree;
	graph_loader.destroy_graph(graph);
	Debug::get_instance()->leave_function("main");
	return 0;
}
