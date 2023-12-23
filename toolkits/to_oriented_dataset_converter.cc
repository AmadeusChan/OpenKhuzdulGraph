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

struct DegreeStruct {
    VertexId degree;
    VertexId vtx;
};

void convert_graph(
        const std::string input_graph_path,
        const std::string output_graph_path
        ) {
    CSRGraph<Empty, Empty> graph;
    CSRGraphLoader<Empty, Empty> graph_loader;
    graph_loader.load_graph(input_graph_path, graph);

    VertexId num_vertices = graph.get_num_vertices();
    EdgeId num_undirected_edges = graph.get_num_edges();
    assert(num_undirected_edges % 2 == 0);
    EdgeId num_directed_edges = num_undirected_edges / 2;
    
    DegreeStruct * degree_array = new DegreeStruct [num_vertices];
    assert(degree_array != NULL);
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        degree_array[v_i].degree = graph.get_degree(v_i);
        degree_array[v_i].vtx = v_i;
    }
    printf("Sorting the degree array...\n");
    std::sort(degree_array, degree_array + num_vertices, 
            [](const DegreeStruct &left, const DegreeStruct &right) {
                if (left.degree != right.degree) {
                    return left.degree < right.degree;
                }
                return left.vtx < right.vtx;
            });

    printf("Verifying that the degree array is sorted...\n");
    for (VertexId v_i = 1; v_i < num_vertices; ++ v_i) {
        assert(degree_array[v_i - 1].degree < degree_array[v_i].degree ||
                (degree_array[v_i - 1].degree == degree_array[v_i].degree && 
                 degree_array[v_i - 1].vtx < degree_array[v_i].vtx));
    }

    printf("Performing vertex mapping...\n");
    VertexId * vertex_id_mapping = new VertexId [num_vertices];
    assert(vertex_id_mapping != NULL);
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        vertex_id_mapping[degree_array[v_i].vtx] = v_i;
    }

    // dumping the dataset
    int f = open(output_graph_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    assert(f != -1);
    GraphMetaDataStructOnDisk meta_data;

    meta_data.num_vertices = num_vertices;
    meta_data.num_edges = num_directed_edges;
    meta_data.has_node_label = meta_data.has_edge_label = 0;
    write_file(f, (uint8_t*) &meta_data, sizeof(GraphMetaDataStructOnDisk));

    VertexId max_degree = graph.get_max_degree();
    EdgeStruct<Empty> * out_edges = new EdgeStruct<Empty> [max_degree];
    assert(out_edges != NULL);
    EdgeId total_num_out_edges = 0;

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        VertexSet neighbours = graph.get_neighbour_vertices_set(degree_array[v_i].vtx);
        VertexId num_neighbours = neighbours.get_num_vertices();
        VertexId num_out_edges = 0;
        for (VertexId i = 0; i < num_neighbours; ++ i) {
            VertexId dst = vertex_id_mapping[neighbours.get_vertex(i)];
            if (dst > v_i) {
                out_edges[num_out_edges].src = v_i;
                out_edges[num_out_edges].dst = dst;
                num_out_edges ++;
            }
        }
        // sorting the out-edges
        std::sort(
                out_edges, out_edges + num_out_edges,
                [](const EdgeStruct<Empty> &left, const EdgeStruct<Empty> &right) {
                    return left.dst < right.dst;
                }
                );
        // dumping the edges
        write_file(f, (uint8_t*) out_edges,
                sizeof(EdgeStruct<Empty>) * num_out_edges);
        total_num_out_edges += num_out_edges;

        if ((v_i + 1) % 10000 == 0 || v_i == num_vertices - 1) {
            printf("Dumping the out-edges %.4f...\n", 1. * (v_i + 1) / num_vertices);
            printf("\033[F");
        }
    }
    printf("\n");

    assert(close(f) == 0);

    assert(total_num_out_edges == num_directed_edges);

    delete [] degree_array;
    delete [] vertex_id_mapping;
    delete [] out_edges;

    graph_loader.destroy_graph(graph);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <*.dgraph> <*.dgraph.oriented>\n",
                argv[0]);
        exit(-1);
    }

    SharedMemorySys::init_shared_memory_sys();

    std::string input_graph_path = argv[1];
    std::string output_graph_path = argv[2];

    printf("converting the graph... please patiently wait...\n");
    printf("input graph: %s\n", input_graph_path.c_str());
    printf("output graph: %s\n", output_graph_path.c_str());

    convert_graph(input_graph_path, output_graph_path);

    return 0;
}
