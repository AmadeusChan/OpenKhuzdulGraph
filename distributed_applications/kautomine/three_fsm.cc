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

namespace Khuzdul {

    bool frequent_vertex_patterns[MAX_NUM_LABELS];
    
    LabeledPatternHashTable * frequent_edge_patterns;
    LabeledPatternHashTable * frequent_wedge_patterns;
    LabeledPatternHashTable * frequent_triangle_patterns;
    LabeledPatternHashTable * frequent_four_chain_patterns;
    LabeledPatternHashTable * frequent_three_star_patterns;
    
    DistributedPatternDomain * pattern_domain;

    enum UnlabeledPattern { // the unlabeled patterns are enssentially templates for their labeled versions
        Edge,
        Wedge,
        Triangle,
        FourChain,
        ThreeStar
    };

    UnlabeledPattern pattern_template;
    LabeledPattern pattern;

    void apply(ExtendableEmbedding &e, Context context, AbstractEmbeddingExplorationEngine * engine) { 
        switch(pattern_template) {
            case Edge:
                {
                    assert(e.get_size() == 1);
                    VertexId v_0 = e.get_matched_vertex(0);
                    if (engine->get_vertex_label(v_0) != pattern.labels[0]) return ;
                    VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                    VertexId num_nbrs = nbrs.get_num_vertices();
                    for (VertexId v_1_idx = 0; v_1_idx < num_nbrs; ++ v_1_idx) {
                        VertexId v_1 = nbrs.get_vertex(v_1_idx);
                        bool enable = engine->get_vertex_label(v_1) == pattern.labels[1];
                        pattern_domain->insert(context, 0, v_0, enable);
                        pattern_domain->insert(context, 1, v_1, enable);
                    }
                }
                break;
            case Wedge:
                {
                    switch (e.get_size()) {
                        case 1:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                if (engine->get_vertex_label(v_0) != pattern.labels[0]) return ;
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_1_idx = 0; v_1_idx < num_nbrs; ++ v_1_idx) {
                                    VertexId v_1 = nbrs.get_vertex(v_1_idx);
                                    if (engine->get_vertex_label(v_1) != pattern.labels[1]) continue;
                                    engine->scatter(v_1, false, 0, e, context, true);
                                }
                            }
                            break;
                        case 2:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_2_idx = 0; v_2_idx < num_nbrs; ++ v_2_idx) {
                                    VertexId v_2 = nbrs.get_vertex(v_2_idx);
                                    bool enable = engine->get_vertex_label(v_2) == pattern.labels[2] && 
                                        v_2 != v_1;
                                    pattern_domain->insert(context, 0, v_0, enable);
                                    pattern_domain->insert(context, 1, v_1, enable);
                                    pattern_domain->insert(context, 2, v_2, enable);
                                }
                            }
                            break;
                        default:
                            fprintf(stderr, "Invalid embedding size: %d\n", e.get_size());  
                            exit(-1);
                    }
                }
                break;
            case Triangle:
                {
                    switch (e.get_size()) {
                        case 1:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                if (engine->get_vertex_label(v_0) != pattern.labels[0]) return ;
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_1_idx = 0; v_1_idx < num_nbrs; ++ v_1_idx) {
                                    VertexId v_1 = nbrs.get_vertex(v_1_idx);
                                    if (engine->get_vertex_label(v_1) != pattern.labels[1]) continue;
                                    engine->scatter(v_1, true, 0, e, context);
                                }
                            }
                            break;
                        case 2:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexSet v_0_nbrs = e.get_matched_vertex_nbrs(0);
                                VertexSet v_1_nbrs = e.get_matched_vertex_nbrs(1);
                                VertexId buff_size = std::min(
                                        v_0_nbrs.get_num_vertices(), v_1_nbrs.get_num_vertices()
                                        ) * sizeof(VertexId);
                                VertexId * buff = (VertexId*) engine->alloc_thread_local_scratchpad_buffer(context, buff_size);
                                VertexSet intersect_result(buff, 0);
                                v_0_nbrs.intersect_with(&v_1_nbrs, &intersect_result);
                                VertexId num_v_2 = intersect_result.get_num_vertices();
                                for (VertexId v_2_idx = 0; v_2_idx < num_v_2; ++ v_2_idx) {
                                    VertexId v_2 = intersect_result.get_vertex(v_2_idx);
                                    bool enable = engine->get_vertex_label(v_2) == pattern.labels[2];
                                    pattern_domain->insert(context, 0, v_0, enable);
                                    pattern_domain->insert(context, 1, v_1, enable);
                                    pattern_domain->insert(context, 2, v_2, enable);
                                }
                                engine->clear_thread_local_scratchpad_buffer(context);
                            }
                            break;
                        default:
                            fprintf(stderr, "Invalid embedding size: %d\n", e.get_size());  
                            exit(-1);
                    }
                }
                break;
            case FourChain:
                {
                    switch (e.get_size()) {
                        case 1:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                if (engine->get_vertex_label(v_0) != pattern.labels[0]) return ;
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_1_idx = 0; v_1_idx < num_nbrs; ++ v_1_idx) {
                                    VertexId v_1 = nbrs.get_vertex(v_1_idx);
                                    if (engine->get_vertex_label(v_1) != pattern.labels[1]) continue;
                                    engine->scatter(v_1, true, 0, e, context);
                                }
                            }
                            break;
                        case 2:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_2_idx = 0; v_2_idx < num_nbrs; ++ v_2_idx) {
                                    VertexId v_2 = nbrs.get_vertex(v_2_idx);
                                    if (engine->get_vertex_label(v_2) != pattern.labels[2] || v_1 == v_2) continue;
                                    engine->scatter(v_2, false, 0, e, context, true);
                                }
                            } 
                            break;
                        case 3:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexId v_2 = e.get_matched_vertex(2);
                                VertexSet nbrs = e.get_matched_vertex_nbrs(1);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_3_idx = 0; v_3_idx < num_nbrs; ++ v_3_idx) {
                                    VertexId v_3 = nbrs.get_vertex(v_3_idx);
                                    bool enable = engine->get_vertex_label(v_3) == pattern.labels[3] &&
                                        v_3 != v_0 && v_3 != v_2;
                                    pattern_domain->insert(context, 0, v_0, enable);
                                    pattern_domain->insert(context, 1, v_1, enable);
                                    pattern_domain->insert(context, 2, v_2, enable);
                                    pattern_domain->insert(context, 3, v_3, enable);
                                }
                            }
                            break;
                        default:
                            fprintf(stderr, "Invalid embedding size: %d\n", e.get_size());  
                            exit(-1);
                    }
                }
                break;
            case ThreeStar:
                {
                    switch (e.get_size()) {
                        case 1:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                if (engine->get_vertex_label(v_0) != pattern.labels[0]) return ;
                                VertexSet nbrs = e.get_matched_vertex_nbrs(0);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_1_idx = 0; v_1_idx < num_nbrs; ++ v_1_idx) {
                                    VertexId v_1 = nbrs.get_vertex(v_1_idx);
                                    if (engine->get_vertex_label(v_1) != pattern.labels[1]) continue;
                                    engine->scatter(v_1, true, 0, e, context);
                                }
                            }
                            break;
                        case 2:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexSet nbrs = e.get_matched_vertex_nbrs(1);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_2_idx = 0; v_2_idx < num_nbrs; ++ v_2_idx) {
                                    VertexId v_2 = nbrs.get_vertex(v_2_idx);
                                    if (engine->get_vertex_label(v_2) != pattern.labels[2] || v_0 == v_2) continue;
                                    engine->scatter(v_2, false, 0, e, context, true);
                                }
                            }
                            break;
                        case 3:
                            {
                                VertexId v_0 = e.get_matched_vertex(0);
                                VertexId v_1 = e.get_matched_vertex(1);
                                VertexId v_2 = e.get_matched_vertex(2);
                                VertexSet nbrs = e.get_matched_vertex_nbrs(1);
                                VertexId num_nbrs = nbrs.get_num_vertices();
                                for (VertexId v_3_idx = 0; v_3_idx < num_nbrs; ++ v_3_idx) {
                                    VertexId v_3 = nbrs.get_vertex(v_3_idx);
                                    bool enable = engine->get_vertex_label(v_3) == pattern.labels[3] &&
                                        v_3 != v_0 && v_3 != v_2;
                                    pattern_domain->insert(context, 0, v_0, enable);
                                    pattern_domain->insert(context, 1, v_1, enable);
                                    pattern_domain->insert(context, 2, v_2, enable);
                                    pattern_domain->insert(context, 3, v_3, enable);
                                }
                            }
                            break;
                        default:
                            fprintf(stderr, "Invalid embedding size: %d\n", e.get_size());  
                            exit(-1);
                    }
                }
                break;
            default:
                fprintf(stderr, "Not supported pattern template!\n");
                exit(-1);
        }
    }

    void init(DistributedApplication * app) {
        VertexId num_vertices = app->get_graph_obj()->get_num_vertices();
        int num_threads = app->get_num_threads();
        int num_sockets = app->get_num_sockets();

        frequent_edge_patterns = new LabeledPatternHashTable();
        frequent_wedge_patterns = new LabeledPatternHashTable();
        frequent_triangle_patterns = new LabeledPatternHashTable();
        frequent_four_chain_patterns = new LabeledPatternHashTable();
        frequent_three_star_patterns = new LabeledPatternHashTable();
        
        pattern_domain = new DistributedPatternDomain(4, num_threads, num_sockets, num_vertices);
    }

    void finalize(DistributedApplication * app) {
        delete frequent_edge_patterns;
        delete frequent_wedge_patterns;
        delete frequent_triangle_patterns;
        delete frequent_four_chain_patterns;
        delete frequent_three_star_patterns;

        delete pattern_domain;
    }

    void setup(DistributedApplication * app, VertexId threshold) {
        // finding the frequent single-vertex patterns
        DistGraph * graph = app->get_graph_obj();
        LabelId num_labels = graph->get_num_labels();
        LabelId num_frequent_vertex_patterns = 0;
        EdgeId support_sum = 0;

        memset(frequent_vertex_patterns, 0, sizeof(frequent_vertex_patterns));
        for (LabelId l = 0; l < num_labels; ++ l) {
            VertexId support = graph->get_labeled_vertices_set(l).get_num_vertices();
            frequent_vertex_patterns[l] = (support >= threshold);
            num_frequent_vertex_patterns += (support >= threshold);
            support_sum += support;
        }

        if (DistributedSys::get_instance()->is_master_node()) {
            printf("Number of frequent single-vertex patterns: %u, support sum: %lu\n",
                    num_frequent_vertex_patterns, support_sum);
        }
    }

    struct CandidateLabeledPattern {
        LabeledPattern canonical_pattern;
        std::vector<LabeledPattern> non_canonical_automorphisms;
    };
    
    // this class should be generated by the compiler
    class CandidateLabeledPatternGenerator {
        public:
            static void get_candidate_edge_patterns(
                    DistGraph * graph,
                    std::vector<CandidateLabeledPattern> &candidates
                    ) {
                LabelId num_labels = graph->get_num_labels();
                candidates.clear();
    
                CandidateLabeledPattern candidate;
                LabeledPattern automorhism;
                candidate.canonical_pattern.size = 2;
                automorhism.size = 2;
    
                for (LabelId l_0 = 0; l_0 < num_labels; ++ l_0) {
                    if (! frequent_vertex_patterns[l_0]) continue;
                    for (LabelId l_1 = 0; l_1 < num_labels; ++ l_1) {
                        if (! frequent_vertex_patterns[l_1]) continue;
                        if (l_0 <= l_1) {
                            candidate.canonical_pattern.labels[0] = l_0;
                            candidate.canonical_pattern.labels[1] = l_1;
                            candidate.non_canonical_automorphisms.clear();
                            automorhism.labels[0] = l_1;
                            automorhism.labels[1] = l_0;
                            candidate.non_canonical_automorphisms.push_back(automorhism);
                            candidates.push_back(candidate);
                        }
                    }
                }
            }
    
            static void get_candidate_wedge_patterns(
                    DistGraph * graph,
                    std::vector<CandidateLabeledPattern> &candidates
                    ) {
                LabelId num_labels = graph->get_num_labels();
                candidates.clear();
    
                CandidateLabeledPattern candidate;
                LabeledPattern automorhism;
                candidate.canonical_pattern.size = 3;
                automorhism.size = 3;
    
                LabeledPattern subpattern;
    
                for (LabelId l_0 = 0; l_0 < num_labels; ++ l_0) {
                    if (! frequent_vertex_patterns[l_0]) continue;
                    for (LabelId l_1 = 0; l_1 < num_labels; ++ l_1) {
                        subpattern.size = 2;
                        subpattern.labels[0] = l_0;
                        subpattern.labels[1] = l_1;
                        if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                        for (LabelId l_2 = 0; l_2 < num_labels; ++ l_2) {
                            subpattern.size = 2;
                            subpattern.labels[0] = l_0;
                            subpattern.labels[1] = l_2;
                            if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                            if (l_1 <= l_2) {
                                // discovered a canonical candidate pattern
                                candidate.canonical_pattern.labels[0] = l_0;
                                candidate.canonical_pattern.labels[1] = l_1;
                                candidate.canonical_pattern.labels[2] = l_2;
                                candidate.non_canonical_automorphisms.clear();
                                automorhism.labels[0] = l_0;
                                automorhism.labels[1] = l_2;
                                automorhism.labels[2] = l_1;
                                candidate.non_canonical_automorphisms.push_back(automorhism);
                                candidates.push_back(candidate);
                            }
                        }
                    }
                }
            }
    
            static void get_candidate_triangle_patterns(
                    DistGraph * graph,
                    std::vector<CandidateLabeledPattern> &candidates
                    ) {
                LabelId num_labels = graph->get_num_labels();
                candidates.clear();
    
                CandidateLabeledPattern candidate;
                LabeledPattern automorhism;
                candidate.canonical_pattern.size = 3;
                automorhism.size = 3;
    
                LabeledPattern subpattern;
    
                for (LabelId l_0 = 0; l_0 < num_labels; ++ l_0) {
                    if (! frequent_vertex_patterns[l_0]) continue;
                    for (LabelId l_1 = 0; l_1 < num_labels; ++ l_1) {
                        subpattern.size = 2;
                        subpattern.labels[0] = l_0;
                        subpattern.labels[1] = l_1;
                        if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                        for (LabelId l_2 = 0; l_2 < num_labels; ++ l_2) {
                            subpattern.size = 3;
                            subpattern.labels[0] = l_0;
                            subpattern.labels[1] = l_1;
                            subpattern.labels[2] = l_2;
                            if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                            subpattern.labels[0] = l_1;
                            subpattern.labels[1] = l_0;
                            subpattern.labels[2] = l_2;
                            if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                            subpattern.labels[0] = l_2;
                            subpattern.labels[1] = l_0;
                            subpattern.labels[2] = l_1;
                            if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                            if (l_0 <= l_1 && l_1 <= l_2) {
                                candidate.canonical_pattern.labels[0] = l_0;
                                candidate.canonical_pattern.labels[1] = l_1;
                                candidate.canonical_pattern.labels[2] = l_2;
                                candidate.non_canonical_automorphisms.clear();
                                automorhism.labels[0] = l_0; 
                                automorhism.labels[1] = l_2;
                                automorhism.labels[2] = l_1;
                                candidate.non_canonical_automorphisms.push_back(automorhism);
                                automorhism.labels[0] = l_1;
                                automorhism.labels[1] = l_0;
                                automorhism.labels[2] = l_2;
                                candidate.non_canonical_automorphisms.push_back(automorhism);
                                automorhism.labels[0] = l_1;
                                automorhism.labels[1] = l_2;
                                automorhism.labels[2] = l_0;
                                candidate.non_canonical_automorphisms.push_back(automorhism);
                                automorhism.labels[0] = l_2;
                                automorhism.labels[1] = l_0;
                                automorhism.labels[2] = l_1;
                                candidate.non_canonical_automorphisms.push_back(automorhism);
                                automorhism.labels[0] = l_2;
                                automorhism.labels[1] = l_1;
                                automorhism.labels[2] = l_0;
                                candidates.push_back(candidate);
                            }
                        }
                    }
                }
            }
    
            static void get_candidate_four_chain_patterns(
                    DistGraph * graph,
                    std::vector<CandidateLabeledPattern> &candidates
                    ) {
                LabelId num_labels = graph->get_num_labels();
                candidates.clear();
    
                CandidateLabeledPattern candidate;
                LabeledPattern automorhism;
                candidate.canonical_pattern.size = 4;
                automorhism.size = 4;
    
                LabeledPattern subpattern;
    
                for (LabelId l_0 = 0; l_0 < num_labels; ++ l_0) {
                    if (! frequent_vertex_patterns[l_0]) continue;
                    for (LabelId l_1 = 0; l_1 < num_labels; ++ l_1) {
                        subpattern.size = 2;
                        subpattern.labels[0] = l_0;
                        subpattern.labels[1] = l_1;
                        if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                        for (LabelId l_2 = 0; l_2 < num_labels; ++ l_2) {
                            subpattern.size = 3;
                            subpattern.labels[0] = l_0;
                            subpattern.labels[1] = l_1;
                            subpattern.labels[2] = l_2;
                            if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                            for (LabelId l_3 = 0; l_3 < num_labels; ++ l_3) {
                                subpattern.size = 3;
                                subpattern.labels[0] = l_1;
                                subpattern.labels[1] = l_0;
                                subpattern.labels[2] = l_3;
                                if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                                subpattern.size = 2;
                                subpattern.labels[0] = l_0;
                                subpattern.labels[1] = l_2;
                                if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                                subpattern.size = 2;
                                subpattern.labels[0] = l_1;
                                subpattern.labels[1] = l_3;
                                if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                                if (l_0 <= l_1) {
                                    candidate.canonical_pattern.labels[0] = l_0;
                                    candidate.canonical_pattern.labels[1] = l_1;
                                    candidate.canonical_pattern.labels[2] = l_2;
                                    candidate.canonical_pattern.labels[3] = l_3;
                                    candidate.non_canonical_automorphisms.clear();
                                    automorhism.labels[0] = l_1;
                                    automorhism.labels[1] = l_0;
                                    automorhism.labels[2] = l_3;
                                    automorhism.labels[3] = l_2;
                                    candidate.non_canonical_automorphisms.push_back(automorhism);
                                    candidates.push_back(candidate);
                                }
                            }
                        }
                    }
                }
            }
    
            static void get_candidate_three_star_patterns(
                    DistGraph * graph,
                    std::vector<CandidateLabeledPattern> &candidates
                    ) {
                LabelId num_labels = graph->get_num_labels();
                candidates.clear();
    
                CandidateLabeledPattern candidate;
                LabeledPattern automorhism;
                candidate.canonical_pattern.size = 4;
                automorhism.size = 4;
    
                LabeledPattern subpattern;
    
                for (LabelId l_0 = 0; l_0 < num_labels; ++ l_0) {
                    if (! frequent_vertex_patterns[l_0]) continue;
                    for (LabelId l_1 = 0; l_1 < num_labels; ++ l_1) {
                        subpattern.size = 2;
                        subpattern.labels[0] = l_0;
                        subpattern.labels[1] = l_1;
                        if (! frequent_edge_patterns->is_exist(subpattern)) continue;
                        for (LabelId l_2 = 0; l_2 < num_labels; ++ l_2) {
                            subpattern.size = 3;
                            subpattern.labels[0] = l_0;
                            subpattern.labels[1] = l_1;
                            subpattern.labels[2] = l_2;
                            if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                            for (LabelId l_3 = 0; l_3 < num_labels; ++ l_3) {
                                subpattern.size = 3;
                                subpattern.labels[0] = l_0;
                                subpattern.labels[1] = l_2;
                                subpattern.labels[2] = l_3;
                                if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                                subpattern.labels[0] = l_0;
                                subpattern.labels[1] = l_1;
                                subpattern.labels[2] = l_3;
                                if (! frequent_wedge_patterns->is_exist(subpattern)) continue;
                                if (l_1 <= l_2 && l_2 <= l_3) {
                                    candidate.canonical_pattern.labels[0] = l_1;
                                    candidate.canonical_pattern.labels[1] = l_0;
                                    candidate.canonical_pattern.labels[2] = l_2;
                                    candidate.canonical_pattern.labels[3] = l_3;
                                    candidate.non_canonical_automorphisms.clear();
                                    automorhism.labels[0] = l_1;
                                    automorhism.labels[1] = l_0;
                                    automorhism.labels[2] = l_3;
                                    automorhism.labels[3] = l_2;
                                    candidate.non_canonical_automorphisms.push_back(automorhism);
                                    automorhism.labels[0] = l_2;
                                    automorhism.labels[1] = l_0;
                                    automorhism.labels[2] = l_1;
                                    automorhism.labels[3] = l_3;
                                    candidate.non_canonical_automorphisms.push_back(automorhism);
                                    automorhism.labels[0] = l_2;
                                    automorhism.labels[1] = l_0;
                                    automorhism.labels[2] = l_3;
                                    automorhism.labels[3] = l_1;
                                    candidate.non_canonical_automorphisms.push_back(automorhism);
                                    automorhism.labels[0] = l_3;
                                    automorhism.labels[1] = l_0;
                                    automorhism.labels[2] = l_1;
                                    automorhism.labels[3] = l_2;
                                    candidate.non_canonical_automorphisms.push_back(automorhism);
                                    automorhism.labels[0] = l_3;
                                    automorhism.labels[1] = l_0;
                                    automorhism.labels[2] = l_2;
                                    automorhism.labels[3] = l_1;
                                    candidate.non_canonical_automorphisms.push_back(automorhism);
                                    candidates.push_back(candidate);
                                }
                            }
                        }
                    }
                }
            }
    };

    std::vector<CandidateLabeledPattern> candidates;

    void mining_frequent_edge_patterns(DistributedApplication * app, VertexId threshold) {
        CandidateLabeledPatternGenerator::get_candidate_edge_patterns(app->get_graph_obj(), candidates);
        frequent_edge_patterns->clear();
        int num_candidates = candidates.size();
        int num_frequent_edge_patterns = 0;
        EdgeId support_sum = 0;
        pattern_template = Edge;
        for (int i = 0; i < num_candidates; ++ i) {
            LabeledPattern p = candidates[i].canonical_pattern;
            pattern_domain->clear();
            pattern = p;
            app->run(true, true, pattern.labels[0]);
            VertexId support = pattern_domain->get_support(p.size);
            if (support >= threshold) {
                frequent_edge_patterns->insert(p);
                for (LabeledPattern j: candidates[i].non_canonical_automorphisms) {
                    frequent_edge_patterns->insert(j);
                }
                num_frequent_edge_patterns ++;
                support_sum += (EdgeId) support;
            }
        }
        if (DistributedSys::get_instance()->is_master_node()) {
            printf("Number of frequent single-edge patterns: %u, support sum of them: %lu\n",
                    num_frequent_edge_patterns, support_sum);
        }
    }

    void mining_frequent_wedge_patterns(DistributedApplication * app, VertexId threshold) {
        CandidateLabeledPatternGenerator::get_candidate_wedge_patterns(app->get_graph_obj(), candidates);
        frequent_wedge_patterns->clear();
        int num_candidates = candidates.size();
        int num_frequent_wedge_patterns = 0;
        EdgeId support_sum = 0;
        pattern_template = Wedge;
        for (int i = 0; i < num_candidates; ++ i) {
            LabeledPattern p = candidates[i].canonical_pattern;
            pattern_domain->clear();
            pattern = p;
            app->run(true, true, pattern.labels[0]);
            VertexId support = pattern_domain->get_support(p.size);
            if (support >= threshold) {
                frequent_wedge_patterns->insert(p);
                for (LabeledPattern j: candidates[i].non_canonical_automorphisms) {
                    frequent_wedge_patterns->insert(j);
                }
                num_frequent_wedge_patterns ++;
                support_sum += (EdgeId) support;
            }
            //printf("    ******** the support of wedge(%u, %u, %u) is %u\n",
            //        p.labels[0], p.labels[1], p.labels[2], support);
        }
        if (DistributedSys::get_instance()->is_master_node()) {
            //printf("Number of candidates: %d\n", num_candidates);
            printf("Number of frequent wedge patterns: %u, support sum of them: %lu\n",
                    num_frequent_wedge_patterns, support_sum);
        }
    }

    void mining_frequent_triangle_patterns(DistributedApplication * app, VertexId threshold) {
        CandidateLabeledPatternGenerator::get_candidate_triangle_patterns(app->get_graph_obj(), candidates);
        frequent_triangle_patterns->clear();
        int num_candidates = candidates.size();
        int num_frequent_triangle_patterns = 0;
        EdgeId support_sum = 0;
        pattern_template = Triangle;
        for (int i = 0; i < num_candidates; ++ i) {
            LabeledPattern p = candidates[i].canonical_pattern;
            pattern_domain->clear();
            pattern = p;
            app->run(true, true, pattern.labels[0]);
            VertexId support = pattern_domain->get_support(p.size);
            if (support >= threshold) {
                frequent_triangle_patterns->insert(p);
                for (LabeledPattern j: candidates[i].non_canonical_automorphisms) {
                    frequent_triangle_patterns->insert(j);
                }
                num_frequent_triangle_patterns ++;
                support_sum += (EdgeId) support;
            }
        }
        if (DistributedSys::get_instance()->is_master_node()) {
            printf("Number of frequent triangle patterns: %u, support sum of them: %lu\n",
                    num_frequent_triangle_patterns, support_sum);
        }
    }
    
    void mining_frequent_four_chain_patterns(DistributedApplication * app, VertexId threshold) {
        CandidateLabeledPatternGenerator::get_candidate_four_chain_patterns(app->get_graph_obj(), candidates);
        frequent_four_chain_patterns->clear();
        int num_candidates = candidates.size();
        int num_frequent_four_chain_patterns = 0;
        EdgeId support_sum = 0;
        pattern_template = FourChain;
        for (int i = 0; i < num_candidates; ++ i) {
            if (DistributedSys::get_instance()->is_master_node()) {
                printf("    Processing candidates: %d / %d\n", i + 1, num_candidates);
            }
            LabeledPattern p = candidates[i].canonical_pattern;
            pattern_domain->clear();
            pattern = p;
            app->run(true, true, pattern.labels[0]);
            VertexId support = pattern_domain->get_support(p.size);
            if (support >= threshold) {
                frequent_four_chain_patterns->insert(p);
                for (LabeledPattern j: candidates[i].non_canonical_automorphisms) {
                    frequent_four_chain_patterns->insert(j);
                }
                num_frequent_four_chain_patterns ++;
                support_sum += (EdgeId) support;
            }
        }
        if (DistributedSys::get_instance()->is_master_node()) {
            printf("Number of frequent four-chain patterns: %u, support sum of them: %lu\n",
                    num_frequent_four_chain_patterns, support_sum);
        }
    }

    void mining_frequent_three_star_patterns(DistributedApplication * app, VertexId threshold) {
        CandidateLabeledPatternGenerator::get_candidate_three_star_patterns(app->get_graph_obj(), candidates);
        frequent_three_star_patterns->clear();
        int num_candidates = candidates.size();
        int num_frequent_three_star_patterns = 0;
        EdgeId support_sum = 0;
        pattern_template = ThreeStar;
        for (int i = 0; i < num_candidates; ++ i) {
            if (DistributedSys::get_instance()->is_master_node()) {
                printf("    Processing candidates: %d / %d\n", i + 1, num_candidates);
            }
            LabeledPattern p = candidates[i].canonical_pattern;
            pattern_domain->clear();
            pattern = p;
            app->run(true, true, pattern.labels[0]);
            VertexId support = pattern_domain->get_support(p.size);
            if (support >= threshold) {
                frequent_three_star_patterns->insert(p);
                for (LabeledPattern j: candidates[i].non_canonical_automorphisms) {
                    frequent_three_star_patterns->insert(j);
                }
                num_frequent_three_star_patterns ++;
                support_sum += (EdgeId) support;
            }
        }
        if (DistributedSys::get_instance()->is_master_node()) {
            printf("Number of frequent three-star patterns: %u, support sum of them: %lu\n",
                    num_frequent_three_star_patterns, support_sum);
        }
    }

    void mining_frequent_patterns(DistributedApplication * app, VertexId threshold) {
        setup(app, threshold);
        mining_frequent_edge_patterns(app, threshold);
        mining_frequent_wedge_patterns(app, threshold);
        mining_frequent_triangle_patterns(app, threshold);
        mining_frequent_four_chain_patterns(app, threshold); 
        mining_frequent_three_star_patterns(app, threshold);
    }
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s <graph_path (*.dgraph)> <support threshold> [num_threads, default: all]\n",
                argv[0]);
        exit(-1);
    }
    DistributedSys::init_distributed_sys();

    // num_threads = -1 indicates that all CPU cores are utilized 
    int num_threads = argc == 4 ? std::atoi(argv[3]): -1;
    std::string graph_path = argv[1]; 
    VertexId threshold = std::atoi(argv[2]);

    Khuzdul::DistributedApplication * app = new Khuzdul::DistributedApplication(
            num_threads, graph_path, 4, Khuzdul::apply, true, true
            );
    int node_id = DistributedSys::get_instance()->get_node_id();

    Khuzdul::init(app);

    Khuzdul::LocalPerformanceMetric::init_local_metrics();
    double average_runtime = 0;
    int runs = 1;  
    for (int run = 0; run < runs; ++ run) {
        if (! node_id) {
            printf("\nrun = %d\n", run);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        average_runtime -= get_time();
        Khuzdul::mining_frequent_patterns(app, threshold);
        average_runtime += get_time();
    }

    if (! node_id) {
        average_runtime /= double(runs);
        printf("\n************************************************\n");
        printf("Average runtime: %.3f (ms)\n", average_runtime * 1000);
        printf("************************************************\n\n");
    }

    Khuzdul::finalize(app);

    MPI_Barrier(MPI_COMM_WORLD);
    delete app;
    Khuzdul::LocalPerformanceMetric::print_metrics(runs);

    DistributedSys::finalize_distributed_sys();
    return 0;
}
