#include <functional>

#include "distributed_application.h"
#include "utilities.h"
#include "engine.h"

namespace Khuzdul {

    // DistributedApplication
    
    void DistributedApplication::init_execution_engines(bool 
            finer_grained_multithreading, VertexId cache_degree_threshold,
            double relative_cache_size) {
        printf("Initializing execution engines...\n");

        //lli max_num_embeddings_global_queue = 256 * 1024 * 1024 / sizeof(CompactExtendableEmbedding) * QUEUE_SIZE_SCALE; 
        //lli global_intermediate_buffer_size = 512 * 1024 * 1024; // 512 MB
        //lli global_graph_data_buffer_size = GRAPH_DATA_BUFFER_PER_LEVEL;

        lli max_num_embeddings_global_queue = (memory_size_per_level_ / 8) / sizeof(CompactExtendableEmbedding) * QUEUE_SIZE_SCALE;
        lli global_intermediate_buffer_size = memory_size_per_level_ / 4;
        lli global_graph_data_buffer_size = memory_size_per_level_ / 2;

        printf("    Max Depth: %d\n", max_depth_);

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            printf("Allocating the execution engine (size: %u) on node %d...\n", 
                    sizeof(EmbeddingExplorationEngine), s_i);
            int partition_id = node_id_ * num_sockets_ + s_i;
            void * ptx = numa_alloc_onnode(sizeof(EmbeddingExplorationEngine), s_i);
            printf("Check point A.\n");
            engines_[s_i] = new(ptx) EmbeddingExplorationEngine(
                    &graphs_[s_i],
                    max_depth_,
                    max_num_embeddings_global_queue,
                    global_intermediate_buffer_size,
                    global_graph_data_buffer_size,
                    s_i, num_sockets_,
                    num_threads_ / num_sockets_,
                    apply_fun_,
                    finer_grained_multithreading ? SMALL_CHUNK_SIZE: CHUNK_SIZE,
                    cache_degree_threshold, 
                    relative_cache_size
                );
            printf("Check point B.\n");
        }
        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            engines_[s_i]->set_engines_on_the_same_node(engines_);
        }

        printf("Finished initliazing engines...\n");

        workload_distributer_ = new WorkloadDistributer(node_id_, num_nodes_, num_sockets_,
                max_depth_ + 1, graphs_); 
    }
    
    void DistributedApplication::finalize_execution_engines() {
        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            void * ptx = engines_[s_i];
            engines_[s_i]->~EmbeddingExplorationEngine();
            numa_free(ptx, sizeof(EmbeddingExplorationEngine));
        }
        delete workload_distributer_; 
    }

    DistributedApplication::DistributedApplication(
            int num_threads, std::string graph_path, int max_pattern_size, ApplyFunction apply_fun, 
            bool load_vertex_labels, bool finer_grained_multithreading,
            long long memory_size_per_level, VertexId cache_degree_threshold,
            double relative_cache_size
            ) {
        assert(max_pattern_size >= 3);
        max_depth_ = max_pattern_size - 2;
        apply_fun_ = apply_fun;

        node_id_ = DistributedSys::get_instance()->get_node_id();
        num_nodes_ = DistributedSys::get_instance()->get_num_nodes();

        // two cores per socket are reserved for communication purpose
        int num_threads_limit = numa_num_configured_cpus() - 2 * numa_num_configured_nodes();
        if (num_threads == -1) {
            num_threads_ = num_threads_limit;
        } else {
            if (num_threads <= 0 || num_threads > num_threads_limit) {
                fprintf(stderr, "The number of threads should be between 1 and %d\n",
                        num_threads_limit);
                exit(-1);
            }
            num_threads_ = num_threads;
        }

        int num_threads_limit_per_socket = numa_num_configured_cpus() / numa_num_configured_nodes() - 2; 
        if (num_threads_ <= num_threads_limit_per_socket) {
            num_sockets_ = 1;
        } else {
            num_sockets_ = numa_num_configured_nodes();
            assert(num_threads_ % num_sockets_ == 0);
        }

        memory_size_per_level_ = memory_size_per_level / num_sockets_;
        printf("Chunk size per level: %.3f MB\n", memory_size_per_level / 1024. / 1024.);

        // loading graph
        double load_graph_time = - get_time();
        printf("Number of used sockets: %d\n", num_sockets_);
        printf("Loading graph %s...\n", graph_path.c_str());
        graph_loader_.load_graphs(graph_path, graphs_, num_sockets_, SHARE_DEGREE_ARRAY, load_vertex_labels);
        load_graph_time += get_time();
        printf("It takes %.3f seconds to load the graph.\n", load_graph_time);
        LocalPerformanceMetric::load_graph_time = load_graph_time / 60.;

        init_execution_engines(finer_grained_multithreading, cache_degree_threshold, relative_cache_size);
    }

    DistributedApplication::~DistributedApplication() {
        LocalPerformanceMetric::memory_consumption = get_mem_usage();

        finalize_execution_engines();

        graph_loader_.destroy_graphs(graphs_, num_sockets_);
    }

    class ScatterFuncIntercepter: public AbstractEmbeddingExplorationEngine {
        private:
            ScatterFunction intercepter_;
            DistGraph * graph_;
        public:
            ScatterFuncIntercepter(
                    ScatterFunction intercepter,
                    DistGraph * graph
                    ): intercepter_(intercepter), graph_(graph) {
            }
            void scatter( 
                    const VertexId new_v, 
                    const bool is_new_v_active, 
                    const uint32_t cached_obj_size,
                    ExtendableEmbedding &parent,
                    const Context context,
                    const bool is_finalized = false // finalized embedding will no longer has new active vertices
                    ) {
                intercepter_(new_v, is_new_v_active, cached_obj_size, parent, context, is_finalized);
            }
            void* alloc_thread_local_scratchpad_buffer(const Context &context, const size_t &size) {
                assert(false);
            }
            void clear_thread_local_scratchpad_buffer(const Context &context) {
                assert(false);
            }
            LabelId get_vertex_label(const VertexId v) {
                return graph_->get_vertex_label(v);
            }
    };

    void DistributedApplication::process_hub_vertex(
            VertexId hub_vertex,
            int partition_id, 
            int socket_id,
            VertexId * hub_vertex_nbrs
            ) {
        VertexId degree = engines_[socket_id]->get_degree(hub_vertex);
        int target_partition_id = engines_[socket_id]->get_vertex_master_partition(hub_vertex);
        int target_node_id = target_partition_id / num_sockets_;
        int target_socket_id = target_partition_id % num_sockets_;

        // fetch the adjacent graph data of the hub vertex 
        if (target_partition_id == partition_id) {
            // the graph data resides on the local node same socket
            hub_vertex_nbrs = engines_[socket_id]->local_graph_->get_neighbours_ptx(hub_vertex);
        } else if (target_node_id == node_id_) {
            // the graph data resides on the local node but a remote socket
            VertexId * data_on_remote_socket = engines_[target_socket_id]->local_graph_->get_neighbours_ptx(hub_vertex);
            memcpy(hub_vertex_nbrs, data_on_remote_socket, sizeof(VertexId) * degree);
        } else {
            // the graph data resides on a remote node
            // construct a request (meta data + vertex)
            GraphDataRequestMetaData graph_data_request_meta_data;
            graph_data_request_meta_data.graph_data_disp = 0;
            graph_data_request_meta_data.num_requested_vertices = 1;
            graph_data_request_meta_data.request_sender_socket_id = socket_id;
            graph_data_request_meta_data.request_sender_node_id = node_id_;

            int request_meta_tag = get_request_meta_data_tag(target_socket_id);
            int request_vertices_tag = get_request_vertices_data_tag(socket_id, target_socket_id);
            int respond_tag = get_respond_data_tag(target_socket_id, socket_id);

            // send out the meta data
            // this is valid since at the same time, the request sending thread is actually busy waiting
            // --> no race condition
            MPI_Send(
                    &graph_data_request_meta_data, sizeof(GraphDataRequestMetaData), MPI_CHAR,
                    target_node_id, request_meta_tag, MPI_COMM_WORLD
                    );
            // sending the requested vertex (the hub vertex)
            MPI_Send(
                    &hub_vertex, 1, DistributedSys::get_mpi_data_type<VertexId>(),
                    target_node_id, request_vertices_tag, MPI_COMM_WORLD
                    );
            lli received_graph_data_size;
            MPI_Status status;
            MPI_Recv(
                    &received_graph_data_size, 1,
                    MPI_LONG_LONG_INT, 
                    target_node_id, respond_tag, MPI_COMM_WORLD, &status
                    );
            assert(received_graph_data_size == sizeof(VertexId) * degree);

            // copy the recieved graph data to our local buffer
            memcpy(hub_vertex_nbrs, engines_[socket_id]->global_graph_data_buffer_[0], 
                    sizeof(VertexId) * degree);
        }
        
        // construct the corresponding single-vertex embedding
        CompactExtendableEmbedding compact_e;
        compact_e.parent = nullptr;
        compact_e.new_vertex_graph_data = hub_vertex_nbrs;
        compact_e.new_vertex = hub_vertex;
        compact_e.new_vertex_degree = degree;
        compact_e.new_vertex_partition_id = target_partition_id;
        compact_e.cached_object_offset = 0;
        compact_e.cached_object_size = 0;

        ExtendableEmbedding e;
        e.matched_vertices_nbrs_ptx[0] = hub_vertex_nbrs;
        e.matched_vertices_num_nbrs[0] = degree;
        e.matched_vertices[0] = hub_vertex;
        e.size = 1;
        e.cached_objects[0] = nullptr;
        e.compact_version = &compact_e;

        // process the constructed extendable embeddings
        ScatterFunction scatter_fun = [&](
                const VertexId new_v, 
                const bool is_new_v_active, 
                const uint32_t cached_obj_size,
                ExtendableEmbedding &parent,
                const Context context,
                const bool is_finalized = false // finalized embedding will no longer has new active vertices
                ) {
            int belonged_partition_id = engines_[socket_id]->get_vertex_master_partition(new_v);
            if (belonged_partition_id == partition_id) {
                engines_[socket_id]->scatter_entry_level_extendable_embedding(
                        new_v, is_new_v_active, cached_obj_size, parent, context, is_finalized
                        );
            }
        };
        ScatterFuncIntercepter intercepter(scatter_fun, &graphs_[socket_id]);

        // invoke the apply function to intercept the size-2 extendable embeddings
        Context context;
        context.thread_id = 0;
        context.socket_id = socket_id;
        context.depth = -1;
        apply_fun_(e, context, &intercepter);

        // flush the engine
        engines_[socket_id]->flush_all_extendable_embeddings();
    }

    void DistributedApplication::run(bool disable_output, bool restrict_starting_vertex_label, LabelId starting_vertex_label) {
        uint64_t start_cycles = get_cycle();
        workload_distributer_->reset_workload(restrict_starting_vertex_label, starting_vertex_label);

        std::thread * main_threads[MAX_NUM_SOCKETS];

        CPUUtilizationMeasurer cpu_utilization_measurer;
        cpu_utilization_measurer.start_measurement();

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            engines_[s_i]->clear_all_buffers();
            engines_[s_i]->clear_graph_data_cache();
            main_threads[s_i] = new std::thread([&](int socket_id) {
                    assert(numa_run_on_node(socket_id) == 0);

                    int partition_id = node_id_ * num_sockets_ + socket_id;

                    double runtime = -get_time();

                    // processing hub vertices
                    VertexId num_hub_vertices = engines_[socket_id]->get_num_hub_vertices();     
                    if (num_hub_vertices > 0) {
                        fprintf(stderr, "do not support hub-vertices yet since there is a unfixed bug.\n");
                        exit(-1);
                    }
                    VertexId * hub_vertices = engines_[socket_id]->get_hub_vertices();
                    if (! disable_output) {
                        printf("node: %d, socket: %d, processing hub vertices (%u vertices)...\n",
                                node_id_, socket_id, num_hub_vertices);
                    }
                    VertexId max_degree = engines_[socket_id]->get_max_degree();
                    VertexId * hub_vertex_nbrs = (VertexId*) numa_alloc_onnode(
                            max_degree * sizeof(VertexId), socket_id
                            );
                    engines_[socket_id]->set_level_0_embedding_size(1);
                    for (VertexId v_0_idx = 0; v_0_idx < num_hub_vertices; ++ v_0_idx) {
                        if (! disable_output) {
                            printf("    node: %d, socket: %d, processing the (%u / %u)-th hub vertex %u (degree: %u)\n",
                                    node_id_, socket_id, v_0_idx + 1, num_hub_vertices, hub_vertices[v_0_idx], 
                                    engines_[socket_id]->get_degree(hub_vertices[v_0_idx]));
                        }
                        if (restrict_starting_vertex_label && 
                                engines_[socket_id]->get_vertex_label(hub_vertices[v_0_idx]) != starting_vertex_label) {
                            continue;
                        }
                        process_hub_vertex(
                                hub_vertices[v_0_idx], partition_id, socket_id, hub_vertex_nbrs
                                );
                    }
                    numa_free(hub_vertex_nbrs, max_degree * sizeof(VertexId));

                    // processing the non-hub vertices
                    if (! disable_output) {
                        printf("node: %d, socket: %d, processing non-hub vertices...\n",
                                node_id_, socket_id);
                    }
                    engines_[socket_id]->set_level_0_embedding_size(0);
                    Workload workload;
                    while (true) {
                        workload_distributer_->fetch_next_batch(socket_id, workload, disable_output);
                        if (workload.num_v_0 == 0) { // no more unprocessed single-vertex embeddings
                            break;
                        }
                        for (VertexId v_0_idx = 0; v_0_idx < workload.num_v_0; ++ v_0_idx) {
                            VertexId v_0 = workload.v_0_list[v_0_idx];
                            // skip hub vertices (differentiated processing)
                            if (engines_[socket_id]->get_degree(v_0) >= HUB_VERTEX_DEGREE_TH) continue;  
                            engines_[socket_id]->scatter_vertex_extendable_embedding(v_0); 
                        }
                        engines_[socket_id]->flush_all_extendable_embeddings(); 
                    }

                    runtime += get_time();
                    if (! disable_output) {
                        printf("node: %d, socket %d, runtime: %.3f (ms)\n", 
                                node_id_, socket_id, runtime * 1000.);
                    }

                }, s_i);
        }
        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            main_threads[s_i]->join();
            delete main_threads[s_i];
        }

        uint64_t end_cycles = get_cycle();
        uint64_t num_cycles = end_cycles - start_cycles;
        if (! disable_output) {
            printf("#cycles: %lu\n", num_cycles);
        }

        cpu_utilization_measurer.end_measurement();
        double cpu_utilization = cpu_utilization_measurer.get_utilization();
        LocalPerformanceMetric::mutex.lock();
        LocalPerformanceMetric::cpu_utilization += cpu_utilization;
        LocalPerformanceMetric::mutex.unlock();
        //printf("****** CPU utilization: %.3f ******\n", cpu_utilization);
    }

}





