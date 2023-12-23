#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "graph.h"
#include "graph_loader.h"
#include "dwarves_debug.h"
#include "shared_memory_sys.h"
#include "timer.h"
#include "memory_pool.h"
#include "vertex_set.h"
#include "main_template.h"

const VertexId degree_threshold = 512;

void dfs(
	VertexId vtx, VertexId curr_depth, VertexId max_depth,
	CSRGraph<Empty, Empty> &graph, VertexId * colors, VertexId *bfs_nums, 
	VertexId * curr_bfs_num, VertexId * bfs_queue, VertexId &queue_tail
	) {
    if (curr_depth < max_depth) {
	VertexSet neighbours = graph.get_neighbour_vertices_set(vtx);
	VertexId num_neighbours = neighbours.get_num_vertices();
	for (VertexId i = 0; i < num_neighbours; ++ i) {
	    VertexId u = neighbours.get_vertex(i);
	    if (colors[u] == -1) { // not visited yet
	        colors[u] = colors[vtx]; // color propagation
	        bfs_nums[u] = curr_bfs_num[colors[u]] ++;
	        bfs_queue[queue_tail ++] = u;
	        dfs(
			u, curr_depth + 1, max_depth,
			graph, colors, bfs_nums, 
			curr_bfs_num, bfs_queue, queue_tail
			);
	    }
	}
    }
}

void bfs_dfs(CSRGraph<Empty, Empty> &graph, std::string output_file_name) {
    VertexId num_vertices = graph.get_num_vertices();
    VertexId * bfs_queue = new VertexId[num_vertices];
    VertexId * colors = new VertexId[num_vertices];
    VertexId * bfs_nums = new VertexId[num_vertices];
    EdgeId num_edges = graph.get_num_edges();

    VertexId queue_head = 0;
    VertexId queue_tail = 0;

    VertexId * high_degree_vertices = new VertexId[num_vertices];
    VertexId num_high_degree_vertices = 0;
    EdgeId degree_sum = 0;

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	if (graph.get_degree(v_i) >= degree_threshold) {
	    high_degree_vertices[num_high_degree_vertices ++] = v_i;
	    degree_sum += graph.get_degree(v_i);
	}
    }
    printf("Number of vertices: %u\n", num_vertices);
    printf("Number of high-degree vertices: %u\n", num_high_degree_vertices);
    printf("Volume of graph data introduced by high-degree vertices: %lu / %lu = %.5f\n", degree_sum, num_edges, 1. * degree_sum / num_edges);

    VertexId * curr_bfs_num = new VertexId [num_high_degree_vertices];
    for (VertexId i = 0; i < num_high_degree_vertices; ++ i) {
	curr_bfs_num[i] = 0;
    }

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	colors[v_i] = -1;
	bfs_nums[v_i] = -1;
    }
    for (VertexId i = 0; i < num_high_degree_vertices; ++ i) {
	VertexId vtx = high_degree_vertices[i];
	colors[vtx] = i;
	bfs_nums[vtx] = curr_bfs_num[i] ++;
	bfs_queue[queue_tail ++] = vtx;
    }

    while (queue_head < queue_tail) { // while the queue is not empty
	VertexId vtx = bfs_queue[queue_head ++];
	dfs(
		vtx, 0, 4, 
		graph, colors, bfs_nums, 
		curr_bfs_num, bfs_queue, queue_tail
	   );
    }
    printf("Number of vertices explored by BFS: %u / %u\n", queue_tail, num_vertices);
    //assert(queue_tail == num_vertices);

    VertexId sum = 0;
    for (VertexId i = 0; i < num_high_degree_vertices; ++ i) {
	sum += curr_bfs_num[i];
    }
    assert(sum == queue_tail);

    VertexId * new_vertex_ids = new VertexId[num_vertices];
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	new_vertex_ids[v_i] = -1;
    }
    VertexId * offset = new VertexId [num_vertices];
    VertexId curr_offset = 0;
    for (VertexId color = 0; color < num_high_degree_vertices; ++ color) {
	offset[color] = curr_offset;
	curr_offset += curr_bfs_num[color];
    }

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	VertexId c = colors[v_i];
	if (c != -1) {
	    new_vertex_ids[v_i] = offset[c] + bfs_nums[v_i];
	}
    }
    VertexId curr_used_vertex_id = queue_tail;
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	if (new_vertex_ids[v_i] == -1) {
	    new_vertex_ids[v_i] = curr_used_vertex_id ++;
	}
    }
    assert(curr_used_vertex_id == num_vertices);

    bool * mapped_vertex_ids = new bool[num_vertices];
    memset(mapped_vertex_ids, 0, sizeof(bool) * num_vertices);

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	mapped_vertex_ids[new_vertex_ids[v_i]] = true;
    }
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	assert(mapped_vertex_ids[v_i] == true);
    }

    EdgeStruct<Empty> * edges = new EdgeStruct<Empty> [num_edges];
    num_edges = 0;

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
	VertexSet neighbours = graph.get_neighbour_vertices_set(v_i);
	VertexId num_neighbours = neighbours.get_num_vertices();
	for (VertexId i = 0; i < num_neighbours; ++ i) {
	    VertexId u = neighbours.get_vertex(i);
	    edges[num_edges].src = new_vertex_ids[v_i];
	    edges[num_edges].dst = new_vertex_ids[u];
	    ++ num_edges;
	}
    }
    assert(num_edges == graph.get_num_edges());

    int f = open(output_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    assert(f != -1);

    write_file(f, (uint8_t*) edges, sizeof(EdgeStruct<Empty>) * num_edges);

    assert(close(f) == 0);

    delete [] bfs_queue;
    delete [] colors;
    delete [] bfs_nums;
    delete [] curr_bfs_num;
    delete [] new_vertex_ids;
    delete [] offset;
    delete [] mapped_vertex_ids;
    delete [] edges;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
	Debug::get_instance()->print("usage: ./locality_aware_reordering [graph (*.dgraph)] [output edge list]");
	exit(-1);
    }

    SharedMemorySys::init_shared_memory_sys();
    CSRGraph<Empty, Empty> graph;
    CSRGraphLoader<Empty, Empty> graph_loader;
    std::string graph_file_path = argv[1];
    graph_loader.load_graph(graph_file_path, graph);
    std::string output_file_name = argv[2];

    bfs_dfs(graph, output_file_name);

    graph_loader.destroy_graph(graph);
    return 0;
}
