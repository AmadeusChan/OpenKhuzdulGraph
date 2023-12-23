#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include <utility>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>

#include "graph.h"
#include "graph_loader.h"
#include "dwarves_debug.h"
#include "shared_memory_sys.h"
#include "timer.h"
#include "memory_pool.h"
#include "vertex_set.h"

typedef CSRGraph<Empty, Empty> Graph;
typedef CSRGraphLoader<Empty, Empty> GraphLoader;

const std::string trace_dir = "/ssd/amadeuschan/dist_comm_traces/uk";
const std::string graph_path = "/ssd/amadeuschan/shuffled_datasets/uk_2005.dgraph";
const int num_sockets = 2;
const int num_nodes = 4;
const int num_threads = 16;

struct Request {
    VertexId vid;
    VertexId degree;

    Request(VertexId vid_, VertexId degree_):
        vid(vid_), degree(degree_) {
        }

} __attribute__((packed));

std::vector<Request> requests;
uint64_t * num_accesses; // uint64_t [num_vertices]

struct CommPerVertex {
    VertexId vid;
    size_t comm_size;

    bool operator<(const CommPerVertex &other) const {
        return comm_size > other.comm_size;
    }
};
CommPerVertex * comm_volume_per_vid; // CommPerVertex [num_vertices]

int main(int argc, char ** argv) {

    SharedMemorySys::init_shared_memory_sys(num_threads);

    printf("Loading graph %s\n", graph_path.c_str());
    Graph graph;
    GraphLoader graph_loader;
    graph_loader.load_graph(graph_path, graph);
    VertexId num_vertices = graph.get_num_vertices();

    printf("Loading communication traces...\n");
    for (int n_i = 0; n_i < num_nodes; ++ n_i) {
        for (int s_i = 0; s_i < num_sockets; ++ s_i) {
            printf("    Loading trace: node %d, socket %d\n", n_i, s_i);

            std::string trace_path = trace_dir + "/node_" + std::to_string(n_i) 
                + "_socket_" + std::to_string(s_i) + ".txt";
            FILE * f = fopen(trace_path.c_str(), "r");
            assert(f != NULL);

            const int MAX_HEADER_LEN = 64;
            char headers[MAX_HEADER_LEN];
            assert(fscanf(f, "%s", headers) == 1);
            assert(fscanf(f, "%s", headers) == 1);

            VertexId vid, degree;
            while (fscanf(f, "%u,%u", &vid, &degree) != EOF) {
                if (graph.get_degree(vid) == degree) {
                    requests.push_back(Request(vid, degree));
                } else {
                    printf("Degree not matched: vid: %u, degree: %u / %u\n",
                            vid, graph.get_degree(vid), degree);
                }
            }

            assert(fclose(f) == 0);
        }
    }
    printf("Loaded communication traces.\n");
    printf("Number of requests: %lld\n", (long long int) requests.size());

    num_accesses = new uint64_t [num_vertices];
    comm_volume_per_vid = new CommPerVertex [num_vertices];
    memset(num_accesses, 0, sizeof(uint64_t) * num_vertices);

    size_t total_comm_volume = 0;
    size_t graph_data_size = 0;

    for (Request r: requests) {
        num_accesses[r.vid] ++;
        total_comm_volume += sizeof(VertexId) * r.degree;
    }
    printf("Total communication volume: %.3f GB\n",
            total_comm_volume / 1024. / 1024. / 1024.);

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        graph_data_size += graph.get_degree(v_i) * sizeof(VertexId);
    }
    printf("Grpah data size (edge-data): %.3f GB\n",
            graph_data_size / 1024. / 1024. / 1024.);

#pragma omp parallel for 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        VertexId d = graph.get_degree(v_i);
        comm_volume_per_vid[v_i].vid = v_i;
        comm_volume_per_vid[v_i].comm_size = (size_t) num_accesses[v_i] * sizeof(VertexId) * d;
    }
    std::sort(comm_volume_per_vid, comm_volume_per_vid + num_vertices);

    //for (int i = 0; i < 10; ++ i) {
    //    printf("%d %lu %lu\n", comm_volume_per_vid[i].vid, comm_volume_per_vid[i].comm_size, 
    //            num_accesses[comm_volume_per_vid[i].vid]);
    //}

    //double percentage_threshold = 0.05;
    size_t next_graph_data_threshold = graph_data_size / 40;
    size_t accum_graph_data = 0;
    size_t accum_comm_size = 0;

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        VertexId vid = comm_volume_per_vid[v_i].vid;
        VertexId d = graph.get_degree(vid);
        accum_graph_data += sizeof(VertexId) * d;
        accum_comm_size += comm_volume_per_vid[v_i].comm_size;

        if (v_i < 10) {
            printf("vid = %u, comm_size = %lu, degree = %u, graph_data_size = %.3f (MB)\n",
                    vid, comm_volume_per_vid[v_i].comm_size, d, d * sizeof(VertexId) / 1024. / 1024.);
        }

        if (accum_graph_data >= next_graph_data_threshold) {
            double graph_data_percentage = (double) accum_graph_data / graph_data_size;
            double comm_size_percentage = (double) accum_comm_size / total_comm_volume;
            printf("Most frequently accessed 0(percent) to %.3f (percent) graph data contributes to %.3f(percent) communication volume.\n",
                    100. * graph_data_percentage,
                    100. * comm_size_percentage);
            next_graph_data_threshold += graph_data_size / 40;
        }
    }

    graph_loader.destroy_graph(graph);
    delete [] num_accesses;
    delete [] comm_volume_per_vid;

    return 0;
}

