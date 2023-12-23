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
#define MAX_COUNTING_PATTERN_SIZE 5
#define NUM_COUNTING_PATTERNS 1

int multiplicities[] = {1};

namespace Khuzdul{

#ifndef ENABLE_COMPUTE_FUSION
    Aggregator<EdgeId> * count;  
#else
    Aggregator<EdgeId> * counts[NUM_COUNTING_PATTERNS];  
#endif
    // this indicates the pattern ID that is being matched 
    int curr_pattern_id; 
    
    // the code generator will recognize this annotation and generate the 
    // apply() function here
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
                                    bool valid = true && v > v_0;
                                    if (! valid) continue;
                                    // Scattering new extendable embeddings
                                    VertexId cached_obj_size = 0;
                                    cached_obj_size += ((prefix_0.get_num_vertices() + 1) * sizeof(VertexId));
                                    cached_obj_size += (MAX_NUM_INTERMEDIATE_OBJS + 2) * sizeof(VertexId);
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
                                VertexId * cached_obj_start_ptx = (VertexId*) e.get_cached_obj(0);
                                init_cached_objs(cached_obj_start_ptx);
                                VertexId * cached_ptx_1 = create_cached_obj(cached_obj_start_ptx, prefix_0.get_num_vertices()) + 1;
                                VertexSet prefix_1(cached_ptx_1, 0);
                                prefix_0.intersect_with(&nbrs, &prefix_1);
                                *(cached_ptx_1 - 1) = prefix_1.get_num_vertices();
                                // Scattering new extendable embedings or processing them (in the case that they are already complete)
                                VertexId num_v = prefix_1.get_num_vertices();
                                for (VertexId v_idx = 0; v_idx < num_v; ++ v_idx) {
                                    VertexId v = prefix_1.get_vertex(v_idx);
                                    bool valid = true && v > v_1;
                                    if (! valid) continue;
                                    // Scattering new extendable embeddings
                                    VertexId cached_obj_size = 0;
                                    cached_obj_size += ((prefix_1.get_num_vertices() + 1) * sizeof(VertexId));
                                    cached_obj_size += (MAX_NUM_INTERMEDIATE_OBJS + 2) * sizeof(VertexId);
                                    engine->scatter(v, true, cached_obj_size, e, context);
                                }
                            }
                            break;
                        case 3:
                            {
                                // Extending a size-3 extendable embedding to a size-4 one
                                // Loading matched vertices...
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexId v_2 = e.get_matched_vertex(2);
                                // Loading necessary prefixes
                                VertexId * cached_ptx_prefix_1 = get_cached_obj((VertexId*) e.get_cached_obj(1), 0);
                                VertexSet prefix_1(cached_ptx_prefix_1 + 1, *cached_ptx_prefix_1);
                                // Generating new prefixes
                                VertexSet nbrs = e.get_matched_vertex_nbrs(2);
                                VertexId * cached_obj_start_ptx = (VertexId*) e.get_cached_obj(0);
                                init_cached_objs(cached_obj_start_ptx);
                                VertexId * cached_ptx_2 = create_cached_obj(cached_obj_start_ptx, prefix_1.get_num_vertices()) + 1;
                                VertexSet prefix_2(cached_ptx_2, 0);
                                prefix_1.intersect_with(&nbrs, &prefix_2);
                                *(cached_ptx_2 - 1) = prefix_2.get_num_vertices();
                                // Scattering new extendable embedings or processing them (in the case that they are already complete)
                                VertexId num_v = prefix_2.get_num_vertices();
                                for (VertexId v_idx = 0; v_idx < num_v; ++ v_idx) {
                                    VertexId v = prefix_2.get_vertex(v_idx);
                                    bool valid = true && v > v_2;
                                    if (! valid) continue;
                                    // Scattering new extendable embeddings
                                    VertexId cached_obj_size = 0;
                                    engine->scatter(v, true, cached_obj_size, e, context);
                                }
                            }
                            break;
                        case 4:
                            {
                                // Extending a size-4 extendable embedding to a size-5 one
                                // Loading matched vertices...
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexId v_2 = e.get_matched_vertex(2);
                                VertexId v_3 = e.get_matched_vertex(3);
                                // Loading necessary prefixes
                                VertexId * cached_ptx_prefix_2 = get_cached_obj((VertexId*) e.get_cached_obj(1), 0);
                                VertexSet prefix_2(cached_ptx_prefix_2 + 1, *cached_ptx_prefix_2);
                                // Generating new prefixes
                                VertexSet nbrs = e.get_matched_vertex_nbrs(3);
                                VertexId * cached_ptx_3 = (VertexId*) engine->alloc_thread_local_scratchpad_buffer(context, prefix_2.get_num_vertices() * sizeof(VertexId));
                                VertexSet prefix_3(cached_ptx_3, 0);
                                prefix_2.intersect_with(&nbrs, &prefix_3);
                                // Scattering new extendable embedings or processing them (in the case that they are already complete)
                                EdgeId local_counter = 0;
                                VertexId num_v = prefix_3.get_num_vertices();
                                for (VertexId v_idx = 0; v_idx < num_v; ++ v_idx) {
                                    VertexId v = prefix_3.get_vertex(v_idx);
                                    bool valid = true && v > v_3;
                                    local_counter += valid;
                                }
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
#ifndef ENABLE_COMPUTE_FUSION
    Khuzdul::count = app->alloc_aggregator<EdgeId>();
#else
    for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
        Khuzdul::counts[pattern_id] = app->alloc_aggregator<EdgeId>();
    }
#endif

    int node_id = DistributedSys::get_instance()->get_node_id();

    Khuzdul::LocalPerformanceMetric::init_local_metrics();
    double average_runtime = 0;
    int runs = 3;
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
#ifndef ENABLE_COMPUTE_FUSION
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
#else
        Khuzdul::curr_pattern_id = 0;
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            Khuzdul::counts[pattern_id]->clear();
        }
        app->run();
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            EdgeId cnt = Khuzdul::counts[pattern_id]->evaluate() / multiplicities[pattern_id]; ;
            if (! node_id) {
                printf("    Pattern count %d: %lu\n", pattern_id, cnt);
            }
        }
#endif
    }
    for (int run = 0; run < runs; ++ run) {
        if (! node_id) {
            printf("\nrun = %d\n", run);
        }
        EdgeId cnt[NUM_COUNTING_PATTERNS];
        MPI_Barrier(MPI_COMM_WORLD);
        average_runtime -= get_time();
#ifndef ENABLE_COMPUTE_FUSION
        // counting multiple patterns sequentially
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            if (! node_id) {
                printf("****** Mining pattern %d ******\n", pattern_id);
            }
            Khuzdul::curr_pattern_id = pattern_id;
            Khuzdul::count->clear();
            app->run();
            cnt[pattern_id] = Khuzdul::count->evaluate();
            assert(cnt[pattern_id] % multiplicities[pattern_id] == 0);
            cnt[pattern_id] /= multiplicities[pattern_id];
        }
#else
        Khuzdul::curr_pattern_id = 0;
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            Khuzdul::counts[pattern_id]->clear();
        }
        app->run();
        for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
            EdgeId unadjusted_cnt = Khuzdul::counts[pattern_id]->evaluate();
            assert(unadjusted_cnt % multiplicities[pattern_id] == 0);
            cnt[pattern_id] = unadjusted_cnt / multiplicities[pattern_id];
        }
#endif
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

#ifndef ENABLE_COMPUTE_FUSION
    app->dealloc_aggregator(Khuzdul::count);
#else
    for (int pattern_id = 0; pattern_id < NUM_COUNTING_PATTERNS; ++ pattern_id) {
        app->dealloc_aggregator(Khuzdul::counts[pattern_id]);
    }
#endif
    delete app;
    MPI_Barrier(MPI_COMM_WORLD);
    Khuzdul::LocalPerformanceMetric::print_metrics(need_warmup ? runs + 1: runs);

    DistributedSys::finalize_distributed_sys();
    return 0;
}

