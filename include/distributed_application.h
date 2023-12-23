#ifndef DISTRIBUTED_APPLICATION_H
#define DISTRIBUTED_APPLICATION_H

#include <functional>

#include "engine.h"
#include "graph.h"
#include "graph_loader.h"

namespace Khuzdul {

    class DistributedApplication {
        private:
            int num_threads_;
            int num_sockets_;

            int max_depth_;

            int node_id_;
            int num_nodes_;

            DistGraph graphs_[MAX_NUM_SOCKETS];
            DistGraphLoader graph_loader_;
            int num_partitions_;

            EmbeddingExplorationEngine * engines_[MAX_NUM_SOCKETS];
            WorkloadDistributer * workload_distributer_;

            ApplyFunction apply_fun_;

            long long memory_size_per_level_; // the amount of memory allocated for the chunk of a level, default: 2GB

            void init_execution_engines(bool finer_grained_multithreading, VertexId cache_degree_threshold, double relative_cache_size);
            void finalize_execution_engines();
            void process_hub_vertex(VertexId hub_vertex, int partition_id, int socket_id, VertexId * hub_vertex_nbrs);

        public:
            DistributedApplication(int num_threads, std::string graph_path, int max_pattern_size, ApplyFunction apply_fun, 
                    bool load_vertex_labels = false, bool finer_grained_multithreading = false,
                    long long memory_size_per_level = (long long) 4 * 1024 * 1024 * 1024,
                    VertexId cache_degree_threshold = 64,
                    double relative_cache_size = 0.15
                    );
            virtual ~DistributedApplication();
    
            template<typename T>
                Aggregator<T>* alloc_aggregator() {
                    return new Aggregator<T>(num_sockets_, num_threads_);
                }
            template<typename T>
                void dealloc_aggregator(Aggregator<T>* a) {
                    delete a;
                }
            void run(bool disable_output = false, bool restrict_starting_vertex_label = false, LabelId starting_vertex_label = 0);

            // helper functions
            int get_num_threads() {
                return num_threads_;
            }
            int get_num_sockets() {
                return num_sockets_;
            }
            DistGraph * get_graph_obj(int socket_id = 0) {
                return &graphs_[socket_id];
            }
    };

}


#endif
