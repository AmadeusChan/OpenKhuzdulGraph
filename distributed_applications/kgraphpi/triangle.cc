#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <numa.h>
#include <math.h>

#include <thread>
#include <algorithm>
#include <string>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <vector>
#include <queue>
#include <utility>

#include "distributed_sys.h"
#include "graph.h"
#include "graph_loader.h"
#include "shared_memory_sys.h"
#include "types.h"
#include "utilities.h"
#include "vertex_set.h"
#include "timer.h"
#include "engine.h"
#include "distributed_application.h"

// This is a code template used to generate distributed pattern counting applications
// running on top of the Khuzdul infrastructure, currently it is used by:
// - k-GraphPi: a scalable version of the distributed GraphPi system
// Sepecifically, these modified systems only need to generrate the DistributedApplication::apply()
// function.

// the code generator will repalce this annotation with generated macros that
// are used by the generated programs:
// - MAX_COUNTING_PATTERN_SIZE: the max size of the pattern;
// - NUM_COUNTING_PATTERNS: the number of patterns being counted; 
#define MAX_COUNTING_PATTERN_SIZE 3
#define NUM_COUNTING_PATTERNS 1

namespace Khuzdul{

    Aggregator<EdgeId> * count;  
    // this indicates the pattern ID that is being matched 
    int curr_pattern_id; 
    
    // the code generator will recognize this annotation and generate the 
    // apply() function here
    // This apply() function is generated according to the algorithm synthesized by the GraphPi system.
    // This apply() function is generated according to the algorithm synthesized by the GraphPi system.
    void apply(ExtendableEmbedding &e, Context context, AbstractEmbeddingExplorationEngine * engine) { 
        switch (curr_pattern_id) {
            case 0:
                {
                    // Generating the pattern matching process of pattern 0
                    switch(e.get_size()) {
                        // Locating where each prefix resides
                        case 1:
                            {
                                // Extending a size-1 extendable embedding to a size-2 one
                                // Loading matched vertices...
                                VertexId v_0 = e.get_matched_vertex(0);
                                // Loading necessary prefixes
                                // Generating new prefixes
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexSet prefix_0 = nbrs;
                                // Scattering new extendable embedings or processing them (in the case that they are already complete)
                                VertexId num_v = prefix_0.get_num_vertices();
                                for (VertexId v_idx = 0; v_idx < num_v; ++ v_idx) {
                                    VertexId v = prefix_0.get_vertex(v_idx);
                                    bool is_symmetry_broken = v <= v_0;
                                    if (is_symmetry_broken) continue;
                                    bool valid = v != v_0;
                                    if (! valid) continue;
                                    // Scattering new extendable embeddings
                                    VertexId cached_obj_size = 0;
                                    engine->scatter(v, true, cached_obj_size, e, context);
                                }
                            }
                            break;
                        case 2:
                            {
                                // Extending a size-2 extendable embedding to a size-3 one
                                // Loading matched vertices...
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                // Loading necessary prefixes
                                VertexSet prefix_0 = e.get_matched_vertex_nbrs(0);
                                // Generating new prefixes
                                VertexSet nbrs = e.get_matched_vertex_nbrs(1);

                                VertexId buff_size = std::min(nbrs.get_num_vertices(), prefix_0.get_num_vertices()) * sizeof(VertexId);
                                VertexId * cached_ptx_1 = (VertexId*) engine->alloc_thread_local_scratchpad_buffer(context, buff_size);

                                VertexSet prefix_1(cached_ptx_1, 0);
                                prefix_0.intersect_with(&nbrs, &prefix_1);
                                // Scattering new extendable embedings or processing them (in the case that they are already complete)
                                VertexId num_last_v = prefix_1.get_num_vertices();
                                VertexId * last_v_list = prefix_1.get_vertices_list();
                                VertexId lower_bound = v_1;
                                VertexId trimed_num_last_v = trim_v_list_lower_bound(last_v_list, num_last_v, lower_bound);
                                EdgeId local_counter = num_last_v - trimed_num_last_v;
                                local_counter -= std::binary_search(last_v_list + trimed_num_last_v, last_v_list + num_last_v, v_0);
                                local_counter -= std::binary_search(last_v_list + trimed_num_last_v, last_v_list + num_last_v, v_1);
                                count->add(context, local_counter);

                                engine->clear_thread_local_scratchpad_buffer(context);
                            }
                            break;
                        default:
                            fprintf(stderr, "Invalid embedding size: %d\n", e.get_size());
                            exit(-1);
                    }
                }
                break;
            default:
                fprintf(stderr, "The pattern ID is not supported.\n");
                exit(-1);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3 && argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <graph_path (*.dgraph)> [num_threads (-1: all threads), default: all] [num_runs, default: 3] [need_warmup: false]\n",
                argv[0]);
        exit(-1);
    }

    DistributedSys::init_distributed_sys();

    // num_threads = -1 indicates that all CPU cores are utilized 
    int num_threads = argc >= 3 ? std::atoi(argv[2]): -1;
    std::string graph_path = argv[1]; 

    Khuzdul::DistributedApplication * app = new Khuzdul::DistributedApplication(
            num_threads, graph_path, MAX_COUNTING_PATTERN_SIZE, Khuzdul::apply
            );
    Khuzdul::count = app->alloc_aggregator<EdgeId>();

    int node_id = DistributedSys::get_instance()->get_node_id();

    Khuzdul::LocalPerformanceMetric::init_local_metrics();
    double average_runtime = 0;
    int runs = 1;
    if (argc >= 4) {
        runs = std::atoi(argv[3]);
    }
    bool need_warmup = false;
    if (argc >= 5) {
        if (strcmp(argv[4], "true") == 0) {
            need_warmup = true;
        } else if (strcmp(argv[4], "false") == 0) {
            need_warmup = false;
        } else {
            fprintf(stderr, "Bad need_warmup argument: %s\n", argv[4]);
            exit(-1);
        }
    }
    if (need_warmup) {
        if (! node_id) {
            printf("****** Warming up... ******\n");
        }
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            if (! node_id) {
                printf("****** Mining pattern %d ******\n", pattern_id);
            }
            Khuzdul::curr_pattern_id = pattern_id;
            Khuzdul::count->clear();
            app->run();
            EdgeId cnt = Khuzdul::count->evaluate();
            if (! node_id) {
                printf("    Pattern count %d: %lu\n", pattern_id, cnt);
            }
        }
    }
    for (int run = 0; run < runs; ++ run) {
        if (! node_id) {
            printf("\nrun = %d\n", run);
        }
        EdgeId cnt[NUM_COUNTING_PATTERNS];
        MPI_Barrier(MPI_COMM_WORLD);
        average_runtime -= get_time();
        // counting multiple patterns sequentially
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            if (! node_id) {
                printf("****** Mining pattern %d ******\n", pattern_id);
            }
            Khuzdul::curr_pattern_id = pattern_id;
            Khuzdul::count->clear();
            app->run();
            cnt[pattern_id] = Khuzdul::count->evaluate();
        }
        average_runtime += get_time();
        if (! node_id) {
            for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
                printf("    Pattern count %d: %lu\n", 
                        pattern_id, cnt[pattern_id]);
            }
        }
    }

    if (! node_id) {
        average_runtime /= double(runs);
        printf("\n************************************************\n");
        printf("Average runtime: %.3f (ms)\n", average_runtime * 1000);
        printf("************************************************\n\n");
    }

    app->dealloc_aggregator(Khuzdul::count);
    delete app;
    MPI_Barrier(MPI_COMM_WORLD);
    Khuzdul::LocalPerformanceMetric::print_metrics(need_warmup ? runs + 1: runs);

    DistributedSys::finalize_distributed_sys();
    return 0;
}



