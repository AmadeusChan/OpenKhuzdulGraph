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

void flush_buffered_edges(EdgeStruct<Empty> * edge_buffer, EdgeId num_buffered_edges, int fout) {
    write_file(fout, (uint8_t *) edge_buffer, sizeof(EdgeStruct<Empty>) * num_buffered_edges);
}

void convert_graph(std::string input_graph_path, std::string output_graph_path) {
    CSRGraph<Empty, Empty> graph;
    CSRGraphLoader<Empty, Empty> graph_loader;
    graph_loader.load_graph(input_graph_path, graph);

    VertexId num_vertices = graph.get_num_vertices();
    const EdgeId max_num_buffered_edges = 1 << 20;
    EdgeStruct<Empty> * edge_buffer = new EdgeStruct<Empty>[max_num_buffered_edges];
    EdgeId num_buffered_edges = 0;

    int fout = open(output_graph_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    assert(fout != -1);

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        VertexSet neighbours = graph.get_neighbour_vertices_set(v_i);
        VertexId num_neighbours = neighbours.get_num_vertices();
        for (VertexId v_j_idx = 0; v_j_idx < num_neighbours; ++ v_j_idx) {
            VertexId v_j = neighbours.get_vertex(v_j_idx);

            edge_buffer[num_buffered_edges].src = v_i;
            edge_buffer[num_buffered_edges].dst = v_j;
            ++ num_buffered_edges;
            if (num_buffered_edges == max_num_buffered_edges) {
                flush_buffered_edges(
                        edge_buffer, num_buffered_edges, fout
                        );
                num_buffered_edges = 0;
            }
        }
    }
    if (num_buffered_edges > 0) {
        flush_buffered_edges(edge_buffer, num_buffered_edges, fout);
        num_buffered_edges = 0;
    }

    assert(close(fout) == 0);

    delete [] edge_buffer;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        Debug::get_instance()->print("usage: to_biedgelist_converter [Dwarves format dataset (*.dgraph)] [binary edge list format dataset]");
        exit(-1);
    }

    SharedMemorySys::init_shared_memory_sys();

    std::string input_graph_path = argv[1];
    std::string output_graph_path = argv[2];
    //bool is_labeled_graph = (argv[3][0] == '1');

    Debug::get_instance()->print("Converting graph format... please patiently wait.");
    Debug::get_instance()->print("Input graph path: ", input_graph_path);
    Debug::get_instance()->print("Output graph path: ", output_graph_path);
    //Debug::get_instance()->print(is_labeled_graph ? "Graph type: labeled": "Graph type: unlabeled");

    convert_graph(input_graph_path, output_graph_path);

    return 0;
}



