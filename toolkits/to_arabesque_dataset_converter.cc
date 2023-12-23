#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#include <utility>
#include <functional>
#include <algorithm>
#include <string>
#include <fstream>

#include "graph.h"
#include "graph_loader.h"
#include "dwarves_debug.h"
#include "shared_memory_sys.h"
#include "timer.h"
#include "memory_pool.h"
#include "vertex_set.h"

template<typename NodeLabel>
void convert_graph(std::string input_graph_path, std::string output_graph_path) {
	bool is_labeled_graph = std::is_same<NodeLabel, Empty>::value != true;
	CSRGraph<NodeLabel, Empty> graph;
	CSRGraphLoader<NodeLabel, Empty> graph_loader;
	graph_loader.load_graph(input_graph_path, graph);

	VertexId num_vertices = graph.get_num_vertices();
	EdgeId num_edges = graph.get_num_edges() / 2;
	Debug::get_instance()->print("Number of vertices: ", num_vertices);
	Debug::get_instance()->print("Number of edges: ", num_edges);

	std::ofstream fout(output_graph_path.c_str());
	//fout << "# " << num_vertices << " " << num_edges << "\n";
	for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
		fout << v_i << " ";
		if (! is_labeled_graph) {
			fout << "0";
		} else {
			VertexId label = (VertexId) graph.get_node_label(v_i);
			fout << label;
		}
		VertexSet neighbours = graph.get_neighbour_vertices_set(v_i);
		for (VertexId v_j_idx = 0; v_j_idx < neighbours.get_num_vertices(); ++ v_j_idx) {
			VertexId v_j = neighbours.get_vertex(v_j_idx);
			fout << " " << v_j;
		}
		fout << "\n";
	}
	fout.close();
}

int main(int argc, char ** argv) {
	if (argc != 4) {
		Debug::get_instance()->print("usage: to_arabesque_dataset_converter [Dwarves format dataset (*.dgraph)] [Arabesque format dataset] [is labeled graph ? 1: 0]");
		exit(-1);
	}

	SharedMemorySys::init_shared_memory_sys();

	std::string input_graph_path = argv[1];
	std::string output_graph_path = argv[2];
	bool is_labeled_graph = (argv[3][0] == '1');

	Debug::get_instance()->print("Converting graph format... please patiently wait.");
	Debug::get_instance()->print("Input graph path: ", input_graph_path);
	Debug::get_instance()->print("Output graph path: ", output_graph_path);
	Debug::get_instance()->print(is_labeled_graph ? "Graph type: labeled": "Graph type: unlabeled");

	if (is_labeled_graph) {
		convert_graph<VertexId>(input_graph_path, output_graph_path);
	} else {
		convert_graph<Empty>(input_graph_path, output_graph_path);
	}

	return 0;
}





