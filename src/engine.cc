#include <numa.h>

#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "engine.h"
#include "distributed_application.h"
#include "graph.h"

// NumaAwareBufferManager
    
NumaAwareBufferManager::NumaAwareBufferManager(int num_sockets): num_sockets_(num_sockets) {
    total_registered_buffer_size = new lli [num_sockets];
    registered_buffers = new std::vector<NumaAwareBuffer>* [num_sockets];
    allocated_ptx_ = new void* [num_sockets];
    for (int s_i = 0; s_i < num_sockets; ++ s_i) {
        total_registered_buffer_size[s_i] = 0;
        registered_buffers[s_i] = new std::vector<NumaAwareBuffer>();
        registered_buffers[s_i]->clear();
    }
}

NumaAwareBufferManager::~NumaAwareBufferManager() {
    delete [] total_registered_buffer_size;
    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
        delete registered_buffers[s_i];
    }
    delete [] registered_buffers;
    delete [] allocated_ptx_;
}

void NumaAwareBufferManager::register_numa_aware_buffer(
        lli buffer_size, uint8_t ** buffer_pointer, int socket_id) {
    assert(socket_id >= 0 && socket_id < num_sockets_);
    total_registered_buffer_size[socket_id] += buffer_size;
    NumaAwareBuffer buffer;
    buffer.buffer_size = buffer_size;
    buffer.buffer_pointer = buffer_pointer;
    registered_buffers[socket_id]->push_back(buffer);
}

void NumaAwareBufferManager::allocate_numa_aware_buffers() {
    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
        if (total_registered_buffer_size[s_i] == 0) {
            continue;
        }
        uint8_t * ptx = (uint8_t*) numa_alloc_onnode(total_registered_buffer_size[s_i], s_i);
        allocated_ptx_[s_i] = ptx;
        if (ptx == NULL) {
            fprintf(stderr, "Failed to allocate %lld bytes memory on socket %d.\n", 
                    total_registered_buffer_size[s_i], s_i);
            exit(-1);
        }
        memset(ptx, 0, total_registered_buffer_size[s_i]);
        int num_buffers = registered_buffers[s_i]->size();
        lli size_sum = 0;
        for (int i = 0; i < num_buffers; ++ i) {
            NumaAwareBuffer buffer = registered_buffers[s_i]->at(i);
            *buffer.buffer_pointer = ptx;
            ptx += buffer.buffer_size;
            size_sum += buffer.buffer_size;
        }
        assert(size_sum == total_registered_buffer_size[s_i]);
    }
}

// must be called after allocate_numa_aware_buffers()
void NumaAwareBufferManager::deallocate_numa_aware_buffers() {
    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
        if (total_registered_buffer_size[s_i] > 0) {
            numa_free(allocated_ptx_[s_i], total_registered_buffer_size[s_i]);
        }
    }
}

lli NumaAwareBufferManager::get_total_allocated_size() {
    lli total_buffer_size = 0;
    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
        total_buffer_size += total_registered_buffer_size[s_i];
    }
    return total_buffer_size;
}

namespace Khuzdul {

    // GraphDataRequestSender
    
    // return value: 
    // there is some pending request: the depth
    // there is no pending request: -1
    int GraphDataRequestSender::get_next_request_batch(
            lli & num_requested_vertices, 
            lli & requested_graph_data_size,
            int & target_partition_id,
            int & previous_target_partition_id,
            lli & graph_data_disp,
            lli graph_data_start_addr,
            lli & curr_embedding_idx,
            VertexId * requested_vertices
            ) {
        int max_depth = engine_->max_depth_;
        GlobalEmbeddingQueueState * global_queue_states = engine_->global_queue_states_;
        CompactExtendableEmbedding ** global_shuffled_embedding_queues = engine_->global_shuffled_embedding_queues_;
        lli * global_embedding_queue_size = engine_->global_embedding_queue_size_;
        volatile lli * global_num_ready_embeddings = engine_->global_num_ready_embeddings_;

        int depth = -1;
        for (int d_i = 0; d_i <= max_depth; ++ d_i) {
            if (global_queue_states[d_i] == PartialReady && 
                    global_num_ready_embeddings[d_i] < global_embedding_queue_size[d_i]) {
                depth = d_i;
            }
        }

        // there is no pending extendable embeddings to perform the gather operation
        // launch another attempt later 
        if (depth == -1) {
            return -1;
        }

        engine_->global_embedding_queue_mutex_.lock();
        if (! (global_queue_states[depth] == PartialReady && 
                    global_num_ready_embeddings[depth] < global_embedding_queue_size[depth])) {
            engine_->global_embedding_queue_mutex_.unlock();
            return -1;
        }
        engine_->global_embedding_queue_mutex_.unlock();

        // so far, we've verified that there are some pending requests
        num_requested_vertices = 0;
        requested_graph_data_size = 0;

        lli num_ready_embeddings = global_num_ready_embeddings[depth];
        lli num_embeddings = global_embedding_queue_size[depth];
        curr_embedding_idx = num_ready_embeddings;
        CompactExtendableEmbedding * embedding = global_shuffled_embedding_queues[depth] + num_ready_embeddings;
        CompactExtendableEmbedding * first_embedding = embedding;
        target_partition_id = embedding->new_vertex_partition_id;
        assert(target_partition_id != engine_->partition_id_);

        // maintain the generation no correspondingly
        previous_target_partition_id = target_partition_id;
        uint32_t curr_generation = curr_generation_[depth];
        uint32_t * previous_request_generation = previous_request_generation_[depth];
        VertexId * previous_request_vertex_id = previous_request_vertex_id_[depth];
        VertexId ** previous_request_graph_buffer = previous_request_graph_buffer_[depth];

        lli graph_data_recv_buffer_addr = (lli) embedding->new_vertex_graph_data;
        graph_data_disp = graph_data_recv_buffer_addr - graph_data_start_addr;

        lli num_empty_requests = 0; 

        uint8_t * curr_graph_data_ptx = (uint8_t*) embedding->new_vertex_graph_data;
        while (curr_embedding_idx < num_embeddings && 
                num_requested_vertices + 1 < num_vertices_per_request && 
                embedding->new_vertex_partition_id == target_partition_id) {
            VertexId v = embedding->new_vertex;
            VertexId hash = v & vertex_id_mask_;
            // whether the request is duplicated in the outgoing request queue
            bool is_duplicated = previous_request_generation[hash] == curr_generation && 
                previous_request_vertex_id[hash] == v;
            // whether the requested data exists in the graph data cache
            bool is_cached = is_cache_line_valid_[hash] && 
                cached_vertices_[hash] == v;
            num_cache_gets_ ++;
            num_cacge_hits_ += is_cached;
            // whether the request should be sent
            bool should_request_be_sent = ! (is_duplicated || is_cached);
            size_t required_graph_data_buffer_size = should_request_be_sent ? embedding->new_vertex_degree * sizeof(VertexId): 0;
            if (requested_graph_data_size + required_graph_data_buffer_size > graph_data_size_per_request) {
                break;
            }

            // do not need to send duplicated request
            requested_vertices[num_requested_vertices] = v;
            num_requested_vertices += should_request_be_sent ? 1: 0;
            requested_graph_data_size += required_graph_data_buffer_size;
            num_empty_requests += (should_request_be_sent && embedding->new_vertex_degree == 0) ? 1: 0;

            embedding->new_vertex_graph_data = is_cached ? cache_line_data_ptx_[hash] : (
                    is_duplicated ? previous_request_graph_buffer[hash] : ((VertexId*) curr_graph_data_ptx)
                    );
            curr_graph_data_ptx += required_graph_data_buffer_size;
            
            // determine whether replace the content in the hash slot 
            bool replace_the_hash_slot = previous_request_generation[hash] < curr_generation && ! is_cached;
            assert(! (is_duplicated && replace_the_hash_slot)); // if the request is duplicated, it will never fill in the hash slot
            previous_request_generation[hash] = replace_the_hash_slot ? curr_generation: previous_request_generation[hash];
            previous_request_vertex_id[hash] = replace_the_hash_slot ? v : previous_request_vertex_id[hash];
            previous_request_graph_buffer[hash] = replace_the_hash_slot ? embedding->new_vertex_graph_data : previous_request_graph_buffer[hash];

            ++ curr_embedding_idx;
            ++ embedding;
        }

        assert(curr_embedding_idx <= num_embeddings);
        curr_generation_[depth] += (curr_embedding_idx == num_embeddings || 
            embedding->new_vertex_partition_id != target_partition_id);
        //printf("generation[0]: %u\n", curr_generation_[0]);

        //printf("N%d.S%d: Sending a batch of requests: num_requests: %lld (empty requests: %lld), requested_data_size: %.3f (MB), avg_data_size_per_request: %.3f (Bytes)\n",
        //        DistributedSys::get_instance()->get_node_id(), socket_id_,
        //        num_requested_vertices, num_empty_requests, requested_graph_data_size * 1. / 1024. / 1024.,
        //        requested_graph_data_size * 1. / num_requested_vertices);

        return depth;
    }

    void GraphDataRequestSender::thread_main() {
        assert(numa_run_on_node(socket_id_) == 0);
        num_cacge_hits_ = 0;
        num_cache_gets_ = 0;

        printf("Node %d, socket %d: Starting the GraphDataRequestSender...\n", 
                DistributedSys::get_instance()->get_node_id(), socket_id_);
        printf("Node %d, socket %d: graph data cache size: %.3f (MB)\n",
                DistributedSys::get_instance()->get_node_id(), socket_id_,
                max_cached_graph_data_size_ * sizeof(VertexId) / 1024. / 1024.);

        int max_depth = engine_->max_depth_;
        GlobalEmbeddingQueueState * global_queue_states = engine_->global_queue_states_;
        CompactExtendableEmbedding ** global_shuffled_embedding_queues = engine_->global_shuffled_embedding_queues_;
        lli * global_embedding_queue_size = engine_->global_embedding_queue_size_;
        volatile lli * global_num_ready_embeddings = engine_->global_num_ready_embeddings_;
        int partition_id = engine_->partition_id_;

        lli graph_data_start_addr = (lli) engine_->global_graph_data_buffer_[0];
        GraphDataRequestMetaData request_meta_data;

        VertexId * requested_vertices = (VertexId*) numa_alloc_onnode(
                sizeof(VertexId) * (num_vertices_per_request + 1), socket_id_);
        assert(requested_vertices != NULL);

        double sender_comm_time = 0;
        double sender_comm_volume = 0;
        double sender_remote_socket_time = 0;
        double sender_remote_socket_volume = 0;
        double useful_work_time = 0;

        double sum_dup_rate = 0;
        double average_dup_rate = 0;
        lli num_requests = 0;

        int target_partition_id;
        int previous_target_partition_id = -1;

        int node_id = DistributedSys::get_instance()->get_node_id();
        int num_nodes = DistributedSys::get_instance()->get_num_nodes();

        while (is_terminated_ == false) {
            __asm volatile ("pause" ::: "memory");

            lli num_requested_vertices;
            lli requested_graph_data_size;
            lli graph_data_disp;
            lli curr_embedding_idx;

            double start_time = get_time();
            int depth = get_next_request_batch(
                    num_requested_vertices, 
                    requested_graph_data_size,
                    target_partition_id, 
                    previous_target_partition_id,
                    graph_data_disp,
                    graph_data_start_addr,
                    curr_embedding_idx,
                    requested_vertices
                    );

            if (depth != -1) {
                if (num_requested_vertices > 0) {
                    int target_node_id = target_partition_id / num_sockets_;
                    int target_socket_id = target_partition_id % num_sockets_;

                    //printf("Node %d socket %d sent %llu requests to node %d socket %d, checksum: %u\n",
                    //        node_id, socket_id_, num_requested_vertices, target_node_id, target_socket_id,
                    //        sent_vid_checksum);

                    if (target_node_id != node_id) {
                        // if the graph data resides on a remote node, fetch it using network communication (MPI)
                        sender_comm_time -= get_time();
    
                        assert(graph_data_disp % sizeof(VertexId) == 0);
                        request_meta_data.graph_data_disp = graph_data_disp / sizeof(VertexId);
                        request_meta_data.num_requested_vertices = num_requested_vertices;
                        request_meta_data.request_sender_socket_id = socket_id_;
                        request_meta_data.request_sender_node_id = node_id;
    
                        int request_meta_tag = get_request_meta_data_tag(target_socket_id);
                        int request_vertices_tag = get_request_vertices_data_tag(socket_id_, target_socket_id);
                        int respond_tag = get_respond_data_tag(target_socket_id, socket_id_);
    
                        // sending the meta data
                        MPI_Send(
                                &request_meta_data, sizeof(GraphDataRequestMetaData), MPI_CHAR,
                                target_node_id, request_meta_tag, MPI_COMM_WORLD
                                );
                        // sending the requested vertices
                        MPI_Send(
                                requested_vertices, num_requested_vertices,
                                DistributedSys::get_mpi_data_type<VertexId>(),
                                target_node_id, request_vertices_tag, MPI_COMM_WORLD
                                );
                        sender_comm_volume += num_requested_vertices * sizeof(VertexId);
                        // receiving the token (indicating that the data transfer via MPI_Put is complete)
                        lli received_graph_data_size;
                        MPI_Status status;
                        MPI_Recv(
                                &received_graph_data_size, 1,
                                MPI_LONG_LONG_INT,
                                target_node_id, respond_tag, MPI_COMM_WORLD, &status
                                );
                        assert(received_graph_data_size == requested_graph_data_size);
                        
                        sender_comm_time += get_time();
                        sender_comm_volume += requested_graph_data_size;
                    } else {
                        // otherwise, directly access it (remote-numa accesses)
                        sender_remote_socket_time -= get_time();
                        assert(socket_id_ != target_socket_id);

                        DistGraph * local_graph = engine_->local_graph_;
                        DistGraph * remote_socket_graph = engine_->engines_same_node_[target_socket_id]->local_graph_;
                        assert(graph_data_disp % sizeof(VertexId) == 0);
                        VertexId * graph_data_base_ptx = (VertexId*) engine_->global_graph_data_buffer_[0];
                        VertexId * graph_data_ptx = graph_data_base_ptx + (graph_data_disp / sizeof(VertexId));

                        for (int v_i = 0; v_i < num_requested_vertices; ++ v_i) {
                            VertexId requested_vertex = requested_vertices[v_i];
                            VertexId * requested_vertex_data = remote_socket_graph->get_neighbours_ptx(requested_vertex);
                            VertexId degree = local_graph->get_degree(requested_vertex);
                            memcpy(graph_data_ptx, requested_vertex_data, sizeof(VertexId) * degree);
                            graph_data_ptx += degree;
                        }
                        sender_remote_socket_time += get_time();
                        sender_remote_socket_volume += requested_graph_data_size;
                    }
                }

                // avoid cache update cost after it is full
                if (1. * cached_graph_data_size_ / max_cached_graph_data_size_ < 0.99) {   
                    lli embedding_idx_begin = global_num_ready_embeddings[depth];
                    lli embedding_idx_end = curr_embedding_idx;
                    CompactExtendableEmbedding * embedding = global_shuffled_embedding_queues[depth] + embedding_idx_begin;
                    for (lli embedding_idx = embedding_idx_begin; embedding_idx < embedding_idx_end; ++ embedding_idx, ++ embedding) {
                        VertexId v = embedding->new_vertex;
                        VertexId degree = embedding->new_vertex_degree;
                        VertexId hash = v & vertex_id_mask_;
                        bool could_be_cached = ((uint64_t) cached_graph_data_size_ + degree) <= max_cached_graph_data_size_ && 
                            ! is_cache_line_valid_[hash] && degree >= cache_degree_threshold;
                        cached_vertices_[hash] = could_be_cached ? v: cached_vertices_[hash];
                        cache_line_data_ptx_[hash] = could_be_cached ? &cached_graph_data_[cached_graph_data_size_] : cache_line_data_ptx_[hash];
                        size_t memcpy_size = could_be_cached ? sizeof(VertexId) * degree : 0;
                        memcpy(&cached_graph_data_[cached_graph_data_size_], embedding->new_vertex_graph_data, memcpy_size);
                        cached_graph_data_size_ += could_be_cached ? degree: 0;
                        is_cache_line_valid_[hash] = could_be_cached ? true: is_cache_line_valid_[hash];
                    }
                }

                engine_->global_embedding_queue_mutex_.lock();
                global_num_ready_embeddings[depth] = curr_embedding_idx;
                engine_->global_embedding_queue_mutex_.unlock();

                double end_time = get_time();
                useful_work_time += end_time - start_time;
            }
        }

        numa_free((void*) requested_vertices, sizeof(VertexId) * (num_vertices_per_request + 1));

        double bandwidth = sender_comm_volume / 1024. / 1024. / sender_comm_time;
        double remote_socket_bandwidth = sender_remote_socket_volume / 1024. / 1024. / sender_remote_socket_time;
        //printf("Sender bandwidth: %.3f (Bytes/sec)\n", bandwidth);

        LocalPerformanceMetric::mutex.lock();
        LocalPerformanceMetric::aggregated_request_sender_bandwidth += bandwidth;
        LocalPerformanceMetric::aggregated_request_sender_volume += sender_comm_volume / 1024. / 1024.;
        LocalPerformanceMetric::aggregated_request_sender_remote_socket_volume += sender_remote_socket_volume / 1024. / 1024.;
        LocalPerformanceMetric::aggregated_request_sender_remote_socket_bandwidth += remote_socket_bandwidth;
        LocalPerformanceMetric::sender_thread_useful_work_time += useful_work_time / double(num_sockets_);
        LocalPerformanceMetric::sender_thread_communication_time += sender_comm_time / double(num_sockets_);
        LocalPerformanceMetric::cache_hit_rate += 100. * num_cacge_hits_ / num_cache_gets_ / double(num_sockets_);
        LocalPerformanceMetric::mutex.unlock();

    }

    GraphDataRequestSender::GraphDataRequestSender(EmbeddingExplorationEngine * engine, VertexId cache_degree_threshold, double relative_cache_size) {
        this->cache_degree_threshold = cache_degree_threshold;
        relative_cache_size_ = relative_cache_size;
        engine_ = engine;
        socket_id_ = engine->socket_id_;
        num_sockets_ = engine->num_sockets_;

        // initialize the data structures used to filter away duplicated requests
        VertexId num_vertices = engine_->local_graph_->get_num_vertices();
        for (hash_set_size_ = (1 << num_confused_bits << num_confused_bits); hash_set_size_ < num_vertices; hash_set_size_ <<= 1) {
        }
        hash_set_size_ >>= num_confused_bits;
        vertex_id_mask_ = hash_set_size_ - 1; // used to perform fast hash value calculation

        buff_manager_ = new NumaAwareBufferManager(numa_num_configured_nodes());

        int max_depth = engine_->max_depth_;
        for (int d_i = 0; d_i <= max_depth; ++ d_i) {
            curr_generation_[d_i] = 1;

            buff_manager_->register_numa_aware_buffer(
                    sizeof(uint32_t) * hash_set_size_, 
                    (uint8_t**) &previous_request_generation_[d_i],
                    socket_id_
                    );
            buff_manager_->register_numa_aware_buffer(
                    sizeof(VertexId) * hash_set_size_,
                    (uint8_t**) &previous_request_vertex_id_[d_i],
                    socket_id_
                    );
            buff_manager_->register_numa_aware_buffer(
                    sizeof(VertexId*) * hash_set_size_,
                    (uint8_t**) &previous_request_graph_buffer_[d_i],
                    socket_id_
                    );
        }

        // allocating data for the graph data cache
        // the total graph size is (edge data) sizeof(VertexId) * num_edges
        buff_manager_->register_numa_aware_buffer(
                sizeof(bool) * hash_set_size_,
                (uint8_t**) &is_cache_line_valid_,
                socket_id_
                );
        buff_manager_->register_numa_aware_buffer(
                sizeof(VertexId) * hash_set_size_,
                (uint8_t**) &cached_vertices_, 
                socket_id_
                );
        buff_manager_->register_numa_aware_buffer(
                sizeof(VertexId*) * hash_set_size_,
                (uint8_t**) &cache_line_data_ptx_,
                socket_id_
                );

        EdgeId num_edges = engine_->local_graph_->get_num_edges();
        uint32_t d = (uint32_t) (1. / (relative_cache_size_ / num_sockets_));
        max_cached_graph_data_size_ = num_edges / d;

        // calculate the memory that is available for graph data cache
        size_t remained_mem_size = MAX_MEMORY_SIZE;

        assert(remained_mem_size > engine_->local_graph_->get_graph_data_size() * num_sockets_);
        remained_mem_size -= engine_->local_graph_->get_graph_data_size() * num_sockets_;

        assert(remained_mem_size > engine_->engine_data_size_ * num_sockets_);
        remained_mem_size -= engine_->engine_data_size_ * num_sockets_;

        assert(remained_mem_size > buff_manager_->get_total_allocated_size() * num_sockets_);
        remained_mem_size -= buff_manager_->get_total_allocated_size() * num_sockets_;

        size_t max_mem_size_for_cache = remained_mem_size / num_sockets_;

        int node_id = DistributedSys::get_instance()->get_node_id();
        max_cached_graph_data_size_ = std::min(max_cached_graph_data_size_, max_mem_size_for_cache / sizeof(VertexId));
        double ratio = max_cached_graph_data_size_ * 1. / (engine_->local_graph_->get_num_edges());
        printf("    Node %d, socket %d, graph data cache size: (%.3f / %.3f)  (GB)\n",
                node_id, socket_id_, max_cached_graph_data_size_ * sizeof(VertexId) / 1024. / 1024. / 1024.,
                max_mem_size_for_cache / 1024. / 1024. / 1024.);
        printf("    The cache is %.3f of the graph size.\n", ratio);

        buff_manager_->register_numa_aware_buffer(
                sizeof(VertexId) * max_cached_graph_data_size_,
                (uint8_t**) &cached_graph_data_,
                socket_id_
                );
        buff_manager_->allocate_numa_aware_buffers();

        lli sender_thread_buff_size = buff_manager_->get_total_allocated_size();
        printf("*** Each sender thread takes %.3f (GB) memory space.\n",
                sender_thread_buff_size / 1024. / 1024. / 1024.);

        clear_cache();

        thread_entry_ = [&]() {
            thread_main();
        };
        is_terminated_ = false;
        thread_ = new std::thread(thread_entry_);
    }

    GraphDataRequestSender::~GraphDataRequestSender() {
        is_terminated_ = true;
        //printf("Waiting for the Sender thread to join...\n");
        thread_->join();
        //printf("Sender thread joint.\n");
        delete thread_;

        buff_manager_->deallocate_numa_aware_buffers(); 
        delete buff_manager_;
        //printf("The request sender thread is terminated.\n");
    }

    void GraphDataRequestSender::clear_cache() {
        // must invoke this function before each run, otherwise the performance measured will not be accurate
        memset((void*) cached_graph_data_, 0, sizeof(VertexId) * max_cached_graph_data_size_);
        memset((void*) is_cache_line_valid_, 0, sizeof(bool) * hash_set_size_);
        memset((void*) cached_vertices_, 0, sizeof(VertexId) * hash_set_size_);
        memset((void*) cache_line_data_ptx_, 0, sizeof(VertexId*) * hash_set_size_);
        cached_graph_data_size_ = 0;
    }

    // GraphDataRequestHandler
    void GraphDataRequestHandler::thread_main() {
        assert(numa_run_on_node(socket_id_) == 0);

        int node_id = DistributedSys::get_instance()->get_node_id();
        printf("Node %d, socket %d: Starting GraphDataRequestHandler...\n", 
                node_id, socket_id_);

        GraphDataRequestMetaData request_meta_data;
        VertexId * requested_vertices = (VertexId*) numa_alloc_onnode(
                sizeof(VertexId) * num_vertices_per_request, socket_id_);
        VertexId * requested_graph_data = (VertexId*) numa_alloc_onnode(
                graph_data_size_per_request, socket_id_);
        assert(requested_vertices != NULL);
        assert(requested_graph_data != NULL);

        int request_meta_tag = get_request_meta_data_tag(socket_id_);

        double handler_comm_time = 0;
        double handler_comm_volume = 0;

        VertexId num_vertices = graph_->get_num_vertices();

        while (true) {
            // probing incoming message
            MPI_Status status;
            MPI_Request mpi_request;
            MPI_Irecv(
                    &request_meta_data, sizeof(GraphDataRequestMetaData), MPI_CHAR,
                    MPI_ANY_SOURCE, request_meta_tag, MPI_COMM_WORLD, &mpi_request
                    );
            int irecv_flag = 0;
            while (! is_terminated_ && ! irecv_flag) {
                MPI_Test(&mpi_request, &irecv_flag, &status);
            }
            if (is_terminated_) {
                break;
            }
            assert(irecv_flag);


            int request_sender_node_id = status.MPI_SOURCE;
            int request_sender_socket_id = request_meta_data.request_sender_socket_id;
            assert(request_meta_data.request_sender_node_id == request_sender_node_id);
            // receiving the requested vertices
            int request_vertices_tag = get_request_vertices_data_tag(request_sender_socket_id, socket_id_);
            MPI_Recv(
                    requested_vertices, request_meta_data.num_requested_vertices, 
                    DistributedSys::get_mpi_data_type<VertexId>(),
                    request_sender_node_id, request_vertices_tag, MPI_COMM_WORLD, &status
                    );
            //printf("Node %d, socket %d, received a graph data request with %u vertices from node %d socket %d\n",
            //        node_id, socket_id_, request_meta_data.num_requested_vertices, 
            //        request_sender_node_id, request_sender_socket_id);
            //printf("    Requested vertex: %u\n", requested_vertices[0]);

            // gathering the requested graph data
            handler_comm_time -= get_time();
            lli num_neighbours_sum = 0;
            lli num_vertices_per_batch = comm_batch_size / sizeof(VertexId);
            lli remote_disp = request_meta_data.graph_data_disp;
            lli local_disp = 0;
            lli unsent_ready_num_vertices = 0;

            for (lli i = 0; i < request_meta_data.num_requested_vertices; ++ i) {
                VertexId vtx = requested_vertices[i];
                assert(vtx >= 0 && vtx < num_vertices);
                assert(graph_->is_local_vertex(vtx));
                VertexId degree = graph_->get_degree(vtx);
                VertexId * nbrs = graph_->get_neighbours_ptx(vtx);
                memcpy(&requested_graph_data[num_neighbours_sum], 
                        nbrs, degree * sizeof(VertexId));
                num_neighbours_sum += degree;

                unsent_ready_num_vertices += degree;
                if ((unsent_ready_num_vertices >= num_vertices_per_batch) || 
                        i == request_meta_data.num_requested_vertices - 1) {
                    MPI_Put(
                            requested_graph_data + local_disp, unsent_ready_num_vertices,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            request_sender_node_id,
                            remote_disp, unsent_ready_num_vertices,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            engine_->engines_same_node_[request_sender_socket_id]->win_
                           );
                    remote_disp += unsent_ready_num_vertices;
                    local_disp += unsent_ready_num_vertices;
                    unsent_ready_num_vertices = 0;
                }
            }

            lli requested_graph_data_size = num_neighbours_sum * sizeof(VertexId);
            handler_comm_volume += requested_graph_data_size;

            // puting the graph data to the memory of the remote machine
            MPI_Win_flush(request_sender_node_id, 
                    engine_->engines_same_node_[request_sender_socket_id]->win_);

            // sending the acknowledgement 
            int respond_tag = get_respond_data_tag(socket_id_, request_sender_socket_id);
            MPI_Send(
                    &requested_graph_data_size, 1,
                    MPI_LONG_LONG_INT, 
                    request_sender_node_id, respond_tag, MPI_COMM_WORLD
                    );
            handler_comm_time += get_time();
        }

        numa_free((void*) requested_vertices, sizeof(VertexId) * num_vertices_per_request);
        numa_free((void*) requested_graph_data, graph_data_size_per_request);

        LocalPerformanceMetric::mutex.lock();
        LocalPerformanceMetric::aggregated_request_handler_bandwidth += handler_comm_volume / 1024. / 1024. / handler_comm_time;
        LocalPerformanceMetric::aggregated_request_handler_volume += handler_comm_volume / 1024. / 1024.;
        LocalPerformanceMetric::mutex.unlock();
    }
    
    GraphDataRequestHandler::GraphDataRequestHandler(EmbeddingExplorationEngine * engine) {
        engine_ = engine;
        graph_ = engine_->local_graph_;
        socket_id_ = engine_->socket_id_;
        num_sockets_ = engine_->num_sockets_;
        thread_entry_ = [&]() {
            thread_main();
        };
        is_terminated_ = false;
        thread_ = new std::thread(thread_entry_);
    }

    GraphDataRequestHandler::~GraphDataRequestHandler() {
        //printf("Waiting for the Reciever thread to joint.\n");
        is_terminated_ = true;
        thread_->join();
        delete thread_;
        //printf("The request handler thread is terminated.\n");
    }

    // EmbeddingExplorationEngine

    void EmbeddingExplorationEngine::init_buffers() {
        buff_manager_ = new NumaAwareBufferManager(numa_num_configured_nodes());
        int s_i = socket_id_;

        // setting up the global queue && its managed buffers
        for (int d_i = 0; d_i <= max_depth_; ++ d_i) {
            if (num_partitions_ > 1) { 
                buff_manager_->register_numa_aware_buffer(
                        sizeof(CompactExtendableEmbedding) * max_num_embeddings_global_queue_, 
                        (uint8_t**) &global_shuffled_embedding_queues_[d_i], s_i);
            }
            buff_manager_->register_numa_aware_buffer(
                    sizeof(CompactExtendableEmbedding) * max_num_embeddings_global_queue_,  
                    (uint8_t**) &global_embedding_queues_[d_i], s_i);
            buff_manager_->register_numa_aware_buffer(global_intermediate_buffer_size_,
                    &global_intermediate_buffer_[d_i], s_i);
        }

        // make sure that the graph data buffer is a continous space
        for (int d_i = 0; d_i <= max_depth_; ++ d_i) {
            buff_manager_->register_numa_aware_buffer(global_graph_data_buffer_size_, 
                    &global_graph_data_buffer_[d_i], s_i);
        }

        for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
            for (int d_i = 0; d_i <= max_depth_; ++ d_i) {
                buff_manager_->register_numa_aware_buffer(
                        sizeof(CompactExtendableEmbedding) * max_num_embeddings_local_queue_, 
                        (uint8_t**) &local_embedding_queues_[t_i][d_i], s_i);
            }
        }

        // register for the thread-local scratchpad buffer
        for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
            for (int d_i = 0; d_i <= max_depth_; ++ d_i) {
                buff_manager_->register_numa_aware_buffer(
                        local_scratchpad_size_, (uint8_t**) &thread_local_scratchpad_[t_i][d_i], s_i
                        );
            }
        }

        buff_manager_->allocate_numa_aware_buffers();

        lli engine_buff_size = buff_manager_->get_total_allocated_size();
        engine_data_size_ = engine_buff_size;
        printf("*** Each engine takes %.3f (GB) buffer size\n", 
                engine_buff_size / 1024. / 1024. / 1024.);

        if (num_partitions_ == 1) { 
            for (int d_i = 0; d_i <= max_depth_; ++ d_i) {
                global_shuffled_embedding_queues_[d_i] = global_embedding_queues_[d_i];
            }
        } 
    }

    void EmbeddingExplorationEngine::release_buffer() {
        buff_manager_->deallocate_numa_aware_buffers();
        delete buff_manager_;
    }

    void EmbeddingExplorationEngine::shuffle_global_embedding_queue(int depth) {
        double shuffle_time = - get_time();

        //printf("*** Going to shuffle the embedding queue at level %d\n", depth);
        
        assert(num_partitions_ > 0);
        lli num_embeddings = global_embedding_queue_size_[depth];

        lli num_embeddings_per_partition[MAX_NUM_THREADS][MAX_NUM_PARTITIONS];
        lli graph_data_size_per_partition[MAX_NUM_THREADS][MAX_NUM_PARTITIONS];
        lli intermediate_data_size_per_partition[MAX_NUM_THREADS][MAX_NUM_PARTITIONS];

        memset(num_embeddings_per_partition, 0, sizeof(num_embeddings_per_partition));
        memset(graph_data_size_per_partition, 0, sizeof(graph_data_size_per_partition));
        memset(intermediate_data_size_per_partition, 0, sizeof(intermediate_data_size_per_partition));

        // dividing the embeddings, and assign each part to a computation thread && perform local computation
        {
            std::thread * threads[num_computation_threads_];
            auto thread_main = [&](int thread_id) {
                assert(numa_run_on_node(socket_id_) == 0);

                lli begin_idx = (num_embeddings / num_computation_threads_) * thread_id;
                lli end_idx = begin_idx + (num_embeddings / num_computation_threads_);
                end_idx = thread_id == num_computation_threads_ - 1 ? num_embeddings: end_idx;
                CompactExtendableEmbedding * embedding = global_embedding_queues_[depth] + begin_idx;
                for (lli idx = begin_idx; idx < end_idx; ++ idx, ++ embedding) {
                    int relative_partition_id = (num_partitions_ - partition_id_ + embedding->new_vertex_partition_id) % num_partitions_;
                    uint64_t required_graph_data_buffer_size = relative_partition_id == 0 ? 0: ((uint64_t) embedding->new_vertex_graph_data);
                    num_embeddings_per_partition[thread_id][relative_partition_id] += 1;
                    graph_data_size_per_partition[thread_id][relative_partition_id] += required_graph_data_buffer_size;
                    intermediate_data_size_per_partition[thread_id][relative_partition_id] += embedding->cached_object_size;
                }
            };
            for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
                threads[t_i] = new std::thread(thread_main, t_i);
            }
            for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
                threads[t_i]->join();
                delete threads[t_i];
            }
        }

        lli embedding_index[MAX_NUM_THREADS][MAX_NUM_PARTITIONS];
        lli graph_data_index[MAX_NUM_THREADS][MAX_NUM_PARTITIONS];
        lli intermediate_data_index[MAX_NUM_THREADS][MAX_NUM_PARTITIONS];

        lli previous_num_embeddings = 0;
        lli previous_graph_data_size = 0;
        lli previous_intermediate_data_size = 0;

        for (int n_i = 0; n_i < num_partitions_; ++ n_i) {
            for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
                embedding_index[t_i][n_i] = previous_num_embeddings;
                graph_data_index[t_i][n_i] = previous_graph_data_size;
                intermediate_data_index[t_i][n_i] = previous_intermediate_data_size;

                previous_num_embeddings += num_embeddings_per_partition[t_i][n_i];
                previous_graph_data_size += graph_data_size_per_partition[t_i][n_i];
                previous_intermediate_data_size += intermediate_data_size_per_partition[t_i][n_i];
            }
        }

        assert(previous_num_embeddings == num_embeddings);
        assert(previous_graph_data_size <= global_used_graph_data_buffer_size_[depth]);
        assert(previous_intermediate_data_size == global_used_intermediate_buffer_size_[depth]);

        // shuffling the embeddings
        {
            auto thread_main = [&](int thread_id) {
                assert(numa_run_on_node(socket_id_) == 0);

                lli begin_idx = (num_embeddings / num_computation_threads_) * thread_id;
                lli end_idx = begin_idx + (num_embeddings / num_computation_threads_);
                end_idx = thread_id == num_computation_threads_ - 1 ? num_embeddings: end_idx;
                CompactExtendableEmbedding * embedding = global_embedding_queues_[depth] + begin_idx;
                CompactExtendableEmbedding * shuffled_embedding_queue = global_shuffled_embedding_queues_[depth];
                uint8_t * graph_data_base = global_graph_data_buffer_[depth];

                for (lli idx = begin_idx; idx < end_idx; ++ idx, ++ embedding) {
                    int relative_partition_id = (num_partitions_ - partition_id_ + embedding->new_vertex_partition_id) % num_partitions_;
                    uint64_t required_graph_data_buffer_size = relative_partition_id == 0 ? 0: ((uint64_t) embedding->new_vertex_graph_data);

                    lli embedding_idx = embedding_index[thread_id][relative_partition_id];
                    lli graph_idx = graph_data_index[thread_id][relative_partition_id];
                    lli intermeidate_idx = intermediate_data_index[thread_id][relative_partition_id];

                    embedding_index[thread_id][relative_partition_id] ++;
                    graph_data_index[thread_id][relative_partition_id] += required_graph_data_buffer_size;
                    intermediate_data_index[thread_id][relative_partition_id] += embedding->cached_object_size;

                    CompactExtendableEmbedding * target_embedding = shuffled_embedding_queue + embedding_idx;
                    *target_embedding = *embedding;
                    target_embedding->new_vertex_graph_data
                        = (embedding->new_vertex_partition_id != partition_id_) ? (VertexId *)(graph_data_base + graph_idx): embedding->new_vertex_graph_data;
                    embedding->new_vertex_graph_data = target_embedding->new_vertex_graph_data;
                    target_embedding->cached_object_offset = intermeidate_idx;
                }
            };

            std::thread * threads[num_computation_threads_];
            for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
                threads[t_i] = new std::thread(thread_main, t_i);
            }
            for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
                threads[t_i]->join();
                delete threads[t_i];
            }
        }

        lli num_local_embeddings = 0;
        for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
            num_local_embeddings += num_embeddings_per_partition[t_i][0];
        }
        //printf("    *** Shuffling, depth: %d, #local_embeddings: %lld / %lld / %lld, intermediate_buffer_size: %.3f (MB), graph_buffer_size: %.3f (MB)\n",
        //        depth, num_local_embeddings, num_embeddings, max_num_embeddings_global_queue_,
        //        global_used_intermediate_buffer_size_[depth] / 1024. / 1024.,
        //        global_used_graph_data_buffer_size_[depth] / 1024. / 1024.);

        global_num_ready_embeddings_[depth] = num_local_embeddings;
        //}

        shuffle_time += get_time();
        shuffle_time_ += shuffle_time * 1000.;
        //printf("    Shuffling time: %.3f (ms)\n", shuffle_time * 1000.);
    }

    void EmbeddingExplorationEngine::clear_global_queue(int depth) {
        global_embedding_queue_size_[depth] = 0;
        global_num_ready_embeddings_[depth] = 0;

        global_used_graph_data_buffer_size_[depth] = 0;
        global_used_intermediate_buffer_size_[depth] = 0;
    }

    int EmbeddingExplorationEngine::flush_local_embeddings(int thread_id, int depth) {
        if (depth <= max_depth_ && 
                local_embedding_queue_size_[thread_id][depth] > 0) { // there are flushable local embeddings

            lli local_queue_size = local_embedding_queue_size_[thread_id][depth];
            lli local_used_graph_data_size = local_needed_graph_data_buffer_size_[thread_id][depth];
            lli local_used_intermediate_data_size = local_needed_intermediate_buffer_size_[thread_id][depth];

            lli new_embedding_queue_size;
            lli new_used_graph_data_size;
            lli new_used_intermediate_data_size;

            global_embedding_queue_mutex_.lock();

            new_embedding_queue_size = global_embedding_queue_size_[depth] + local_queue_size;
            new_used_graph_data_size = global_used_graph_data_buffer_size_[depth] + local_used_graph_data_size;
            new_used_intermediate_data_size = global_used_intermediate_buffer_size_[depth] + local_used_intermediate_data_size;

            if (new_embedding_queue_size <= max_num_embeddings_global_queue_ &&
                    new_used_graph_data_size <= global_graph_data_buffer_size_ &&
                    new_used_intermediate_data_size <= global_intermediate_buffer_size_) {
                global_embedding_queue_size_[depth] = new_embedding_queue_size;
                global_used_graph_data_buffer_size_[depth] = new_used_graph_data_size;
                global_used_intermediate_buffer_size_[depth] = new_used_intermediate_data_size;
                global_embedding_queue_mutex_.unlock();
            } else {
                global_embedding_queue_mutex_.unlock();
                bool exceed_queue_size = new_embedding_queue_size > max_num_embeddings_global_queue_;
                bool exceed_graph_buffer_size = new_used_graph_data_size > global_graph_data_buffer_size_;
                bool exceed_intermediate_buffer_size = new_used_intermediate_data_size > global_intermediate_buffer_size_;
                //printf("failed to flush local embeddings to the global embedding queue due to the lack of:"
                //        "queue size(%d), graph buffer size(%d), intermediate buffer size(%d)\n",
                //        (int) exceed_queue_size, (int) exceed_graph_buffer_size, (int) exceed_intermediate_buffer_size);
                return -1;
            }

            // reserve buffer of cached objects
            CompactExtendableEmbedding * embedding = local_embedding_queues_[thread_id][depth];
            uint32_t intermediate_data_offset = new_used_intermediate_data_size - local_used_intermediate_data_size;
            for (lli i = 0; i < local_queue_size; ++ i, ++ embedding) {
                embedding->cached_object_offset = intermediate_data_offset;
                intermediate_data_offset += embedding->cached_object_size;
            }
            //assert(intermediate_data_offset == new_used_intermediate_data_size);

            // copying local data to the global queue
            memcpy(
                    &global_embedding_queues_[depth][new_embedding_queue_size - local_queue_size], 
                    &local_embedding_queues_[thread_id][depth][0],
                    sizeof(CompactExtendableEmbedding) * local_queue_size
                  );

            local_embedding_queue_size_[thread_id][depth] = 0;
            local_needed_graph_data_buffer_size_[thread_id][depth] = 0;
            local_needed_intermediate_buffer_size_[thread_id][depth] = 0;
        }
        return 0;
    }

    void EmbeddingExplorationEngine::change_global_queue_state_to_partial_ready(int depth) {
        assert(global_queue_states_[depth] == Filling);

        shuffle_global_embedding_queue(depth);
        global_embedding_queue_mutex_.lock();
        global_queue_states_[depth] = PartialReady;
        global_embedding_queue_mutex_.unlock();
    }

    void EmbeddingExplorationEngine::extend_embeddings(int depth) {
        //printf("Starting to extending the embeddings at level %d\n", depth);
        assert(depth >= 0 && depth <= max_depth_);
        assert(global_queue_states_[depth] == PartialReady);

        next_depth_ = depth + 1;

        volatile lli num_distributed_embeddings = 0; // the number of embeddings that have been assigned to a thread
        volatile lli num_extended_embeddings = 0; // the number of extended (processed) embeddings
        lli num_embeddings_to_extend = global_embedding_queue_size_[depth];
        num_extensions_ += num_embeddings_to_extend;

        volatile bool is_terminated = false;

        volatile double comm_wait_time_sum = 0.;
        volatile double computation_time_sum = 0.;
        std::mutex comm_wait_time_sum_mutex;

        num_suspended_threads_[depth] = 0;
        global_phases_[depth] = 0;

        // starting the computation threads
        auto computation_thread_main = [&](int thread_id) {
            assert(numa_run_on_node(socket_id_) == 0);
            double scatter_time = 0.;

            Context context;
            context.thread_id = thread_id;
            context.socket_id = socket_id_;
            context.depth = depth;
            context.scatter_time = &scatter_time;

            lli thread_begin = 0;
            lli thread_curr = 0;
            lli thread_end = 0;

            local_phases_[depth][thread_id][0] = 0;

            ExtendableEmbedding e;
            e.size = next_depth_ + level_0_emebeding_size_; 
            ApplyFunction apply_fun = apply_fun_;

            double comm_wait_time = 0.;
            double computation_time = 0.;

            while (true) {
                // suspend the computation thread
                {
                    std::lock_guard<std::mutex> lk(num_suspended_comp_threads_mutex_[depth]);
                    if (global_phases_[depth] != local_phases_[depth][thread_id][0]) {
                        global_phases_[depth] = local_phases_[depth][thread_id][0];
                        num_suspended_threads_[depth] = 1;
                    } else {
                        num_suspended_threads_[depth] += 1;
                    }
                    if (num_suspended_threads_[depth] == num_computation_threads_) {
                        // this indicates that all comp threads will go to sleep
                        // so that main thread should be woken up
                        num_suspended_comp_threads_cv_[depth].notify_one();
                    }
                }
                pthread_barrier_wait(&comp_thread_barrier_[depth]);
                local_phases_[depth][thread_id][0] ^= 1;
                // the computation threads are terminated
                if (is_terminated) {
                    break;
                }
                while (true) {
                    // if the thread doesn't own any workload at this point
                    // fetch them
                    // lock-free workload distribution
                    if (thread_curr >= thread_end) {
                        thread_begin = thread_curr = 
                            __sync_fetch_and_add(&num_distributed_embeddings, chunk_size_);
                        thread_end = thread_begin + chunk_size_ < num_embeddings_to_extend ? thread_begin + chunk_size_: num_embeddings_to_extend;
                    }
                    if (thread_curr >= thread_end) {
                        // this indicates that there is no more pending workload 
                        // the computation thread should go to sleep and wait for a new workload batch
                        break;
                    }
                    // busy waiting until the necessary graph data is ready
                    comm_wait_time -= get_time();
                    while (num_partitions_ > 1) {
                        __asm volatile ("pause" ::: "memory");
                        bool cond = global_num_ready_embeddings_[depth] >= thread_end;
                        if (cond) break;
                    }
                    comm_wait_time += get_time();

#ifdef ENABLE_BREAKDOWN_PROFILING
                    computation_time -= get_time();
#endif
                    // extend the extendable embeddings
                    CompactExtendableEmbedding * compact_e = global_shuffled_embedding_queues_[depth] + thread_curr;
                    while (thread_curr < thread_end) {
                        // construct the extendable embedding from its compact version
                        CompactExtendableEmbedding * e_i = compact_e;
                        //uint8_t * cached_object_ptx = global_intermediate_buffer_[depth];
                        for (int j = e.size - 1; j >= 0; -- j) {
                            assert(e_i != nullptr);
                            e.matched_vertices_nbrs_ptx[j] = e_i->new_vertex_graph_data;
                            e.matched_vertices_num_nbrs[j] = e_i->new_vertex_degree;
                            e.matched_vertices[j] = e_i->new_vertex;
                            e.cached_objects[j] = (void*) (global_intermediate_buffer_[j] + e_i->cached_object_offset);
                            e_i = e_i->parent;
                        }
                        e.compact_version = compact_e;

                        apply_fun(e, context, this);
                        ++ thread_curr;
                        ++ compact_e;
                    }
#ifdef ENABLE_BREAKDOWN_PROFILING
                    computation_time += get_time();
#endif
                    // committed the workload
                    lli num_locally_extended_embeddings = thread_end - thread_begin;
                    __sync_fetch_and_add(&num_extended_embeddings, num_locally_extended_embeddings);
                }
            }
            comm_wait_time_sum_mutex.lock();
            comm_wait_time_sum += comm_wait_time;
            computation_time_sum += computation_time - scatter_time;
            comm_wait_time_sum_mutex.unlock();
        };
        std::thread * computation_threads[num_computation_threads_];
        for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
            computation_threads[t_i] = new std::thread(computation_thread_main, t_i);
        }

        int main_thread_local_phase = 0;

        while (num_extended_embeddings < num_embeddings_to_extend) {
            // wake up all computation threads
            pthread_barrier_wait(&comp_thread_barrier_[depth]);
            main_thread_local_phase ^= 1;
            // wait until all computation threads are suspended
            std::unique_lock<std::mutex> lk(num_suspended_comp_threads_mutex_[depth]); 
            num_suspended_comp_threads_cv_[depth].wait(
                    lk, [&] {
                            return global_phases_[depth] == main_thread_local_phase && 
                                    num_suspended_threads_[depth] == num_computation_threads_;
                        }
                    );
            lk.unlock();
            //printf("All computation threads have been suspended...\n");
            //num_synchronization_points ++;
            // two possible reasons for the suspension:
            // 1) the next-level queue is full;
            // 2) all embeddings of the current batch have been extended
            if (depth < max_depth_) {
                // since some embeddings are not completely extended
                // it falls within the first case
                if (num_extended_embeddings < num_embeddings_to_extend) {
                    // the next-level should not be empty
                    assert(global_embedding_queue_size_[next_depth_] > 0);
                    change_global_queue_state_to_partial_ready(next_depth_);
                    extend_embeddings(depth + 1);
                    next_depth_ = depth + 1;
                    // at this point, the next-level queue should be empty
                    assert(global_embedding_queue_size_[next_depth_] == 0);
                }
            } else {
                // if the current level is the last level
                // the first case should not occur
                assert(num_extended_embeddings == num_embeddings_to_extend);
            }
        }

        // so far, all embeddings at this level have been processed
        // wake up the computation threads and terminate them
        is_terminated = true;
        pthread_barrier_wait(&comp_thread_barrier_[depth]);
        for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
            computation_threads[t_i]->join();
            delete computation_threads[t_i];
        }
        
        comm_time_on_critical_path_ += comm_wait_time_sum / double(num_computation_threads_) * 1000.;
        actual_computation_time_ += computation_time_sum / double(num_computation_threads_) * 1000.;

        // at this point, the next-level queue may not be empty
        // we should flush it
        if (depth < max_depth_) {
            while (true) {
                // flush the local buffer first
                for (int t_i = 0; t_i < num_computation_threads_; ++ t_i) {
                    flush_local_embeddings(t_i, next_depth_);
                }
                if (global_embedding_queue_size_[next_depth_] > 0) {
                    change_global_queue_state_to_partial_ready(next_depth_);
                    extend_embeddings(next_depth_);
                    next_depth_ = depth + 1;
                } else {
                    break;
                }
            }
        }

        // there should be no more embeddings in the next-level queue
        // clear to current-level embedding queue
        global_queue_states_[depth] = Filling;
        clear_global_queue(depth);
    }

    void EmbeddingExplorationEngine::clear_all_buffers() {
        //throw_entry_executable_counter_ = 0;
        for (int i = 0; i <= max_depth_; ++ i) {
            global_queue_states_[i] = Filling;
            clear_global_queue(i);
        }
        for (int i = 0; i < num_computation_threads_; ++ i) {
            for (int j = 0; j <= max_depth_; ++ j) {
                local_embedding_queue_size_[i][j] = 0;
                local_needed_graph_data_buffer_size_[i][j] = 0;
                local_needed_intermediate_buffer_size_[i][j] = 0;
                allocated_scratchpad_size_[i][j] = 0;
            }
        }
    }

    void* EmbeddingExplorationEngine::alloc_thread_local_scratchpad_buffer(const Context &context, const size_t &size) {
        size_t allocated_size = allocated_scratchpad_size_[context.thread_id][context.depth];
        assert(allocated_size + size <= local_scratchpad_size_);
        allocated_scratchpad_size_[context.thread_id][context.depth] = allocated_size + size;
        void * buff = (void*)(thread_local_scratchpad_[context.thread_id][context.depth] + allocated_size);
        return buff;
    }

    void EmbeddingExplorationEngine::clear_thread_local_scratchpad_buffer(const Context &context) {
        allocated_scratchpad_size_[context.thread_id][context.depth] = 0;
    }

    void EmbeddingExplorationEngine::clear_graph_data_cache() {
        graph_data_request_sender_->clear_cache(); 
    }

    EmbeddingExplorationEngine::EmbeddingExplorationEngine(
            DistGraph * local_graph,
            lli max_depth, // depth = 0...max_depth
            lli max_num_embeddings_global_queue,
            lli global_intermediate_buffer_size,
            lli global_graph_data_buffer_size,
            int socket_id, 
            int num_sockets,
            int num_computation_threads,
            ApplyFunction apply_fun,
            int chunk_size,
            VertexId cache_degree_threshold,
            double relative_cache_size
            ) {
        apply_fun_ = apply_fun;
        chunk_size_ = chunk_size;

        local_graph_ = local_graph;
        num_sockets_ = num_sockets;
        socket_id_ = socket_id;
        num_partitions_ = local_graph_->get_num_partitions();
        partition_id_ = local_graph_->get_partition_id();

        printf("    Engine socket id: %d, partition_id: %d / %d, chunk_size: %d\n", 
                socket_id_, partition_id_, num_partitions_, chunk_size);

        num_computation_threads_ = num_computation_threads;

        max_depth_ = max_depth;

        max_num_embeddings_global_queue_ = max_num_embeddings_global_queue; 
        global_intermediate_buffer_size_ = global_intermediate_buffer_size;
        global_graph_data_buffer_size_ = global_graph_data_buffer_size;

        const int scale = num_computation_threads_ * 8;
        VertexId max_degree = local_graph_->get_max_degree();
        // scatterred by one apply function
        // ensure that almost all local queued extendable embedding resides on the L1-D cache
        max_num_embeddings_local_queue_ = L1_DCACHE_SIZE / 2 / sizeof(CompactExtendableEmbedding);

        local_graph_data_buffer_size_ = std::min(std::max(global_graph_data_buffer_size / scale, (lli) max_degree * 8), global_graph_data_buffer_size);
        local_intermediate_data_buffer_size_ = std::min(std::max(global_intermediate_buffer_size / scale, (lli) max_degree * 8), global_intermediate_buffer_size_);

        assert(local_graph_data_buffer_size_ > (lli) max_degree * sizeof(VertexId));
        assert(local_intermediate_data_buffer_size_ > (lli) max_degree * sizeof(VertexId));

        local_scratchpad_size_ = std::max(
                (size_t) LOCAL_SCRATCHPAD_SIZE, sizeof(VertexId) * local_graph->get_max_degree() + 1024 * 1024
                );
        printf("Local scratchpad size: %.3f MB\n", (double) local_scratchpad_size_ / 1024. / 1024.);

        init_buffers();
        clear_all_buffers();

        // exposing the graph data buffer to enable one-side data transferring
        int num_nodes = DistributedSys::get_instance()->get_num_nodes();
        int node_id = DistributedSys::get_instance()->get_node_id();

        void * graph_data_base_ptx = global_graph_data_buffer_[0];
        lli graph_data_buffer_size = (lli) global_graph_data_buffer_size_ * (max_depth_ + 1);
        if (num_nodes > 1) {
            MPI_Win_create(
                    graph_data_base_ptx, graph_data_buffer_size,
                    sizeof(VertexId), MPI_INFO_NULL, MPI_COMM_WORLD, &win_
                    );
            for (int n_i = 0; n_i < num_nodes; ++ n_i) {
                // passive synchronization
                if (n_i != node_id) {
                    MPI_Win_lock(MPI_LOCK_SHARED, n_i, 0, win_);
                }
            }
        }

        // starting the communication threads 
        graph_data_request_sender_ = new GraphDataRequestSender(this, cache_degree_threshold, relative_cache_size);
        graph_data_request_handler_ = new GraphDataRequestHandler(this);

        cache_vertex_id_mask_ = graph_data_request_sender_->get_cache_vertex_id_mask(); 
        is_cache_line_valid_ = graph_data_request_sender_->get_is_cache_line_valid();
        cached_vertices_ = graph_data_request_sender_->get_cached_vertices();

        // init the performance metrics
        shuffle_time_ = 0;
        comm_time_on_critical_path_ = 0;
        actual_computation_time_ = 0;

        for (int depth = 0; depth <= max_depth_; ++ depth) {
            pthread_barrier_init(&comp_thread_barrier_[depth], NULL, num_computation_threads_ + 1);
        }

        level_0_emebeding_size_ = 0;
        num_extensions_ = 0;
    }

    EmbeddingExplorationEngine::~EmbeddingExplorationEngine() {
        //printf("Cleaning up the EmbeddingExplorationEngine object...\n");
        int num_nodes = DistributedSys::get_instance()->get_num_nodes();
        int node_id = DistributedSys::get_instance()->get_node_id();
        // unlocking the remote windows
        if (num_nodes > 1) {
            for (int n_i = 0; n_i < num_nodes; ++ n_i) {
                if (n_i != node_id) {
                    MPI_Win_unlock(n_i, win_);
                }
            }
            MPI_Win_free(&win_);
        }

        //printf("Waiting for the communication threads to terminate...\n"); 
        delete graph_data_request_sender_; 
        delete graph_data_request_handler_;

        //printf("Releasing buffer...\n");
        release_buffer(); 

        //printf("Updating global performance metrics...\n");
        LocalPerformanceMetric::mutex.lock();
        LocalPerformanceMetric::shuffle_time += shuffle_time_ / double(num_sockets_); 
        LocalPerformanceMetric::comm_time_on_critical_path += comm_time_on_critical_path_ / double(num_sockets_);
        LocalPerformanceMetric::actual_computation_time += actual_computation_time_ / double(num_sockets_);
        LocalPerformanceMetric::num_extensions += num_extensions_;
        LocalPerformanceMetric::mutex.unlock();
        //printf("Done finalizing the embedding exploration engine.\n");
    }

    void EmbeddingExplorationEngine::flush_all_extendable_embeddings() {
        //printf("Flushing single-vertex extendable embeddings: %lld\n", 
        //        global_embedding_queue_size_[0]);
        change_global_queue_state_to_partial_ready(0);
        extend_embeddings(0);
    }

    void EmbeddingExplorationEngine::scatter_vertex_extendable_embedding(const VertexId v) {
        // pushing the single-vertex extendable embedding to the processing engine
        VertexId degree = local_graph_->get_degree(v);
        int partition_id = local_graph_->get_vertex_master_partition(v);
        bool is_local = partition_id == partition_id_;
        lli required_graph_data_buffer_size = is_local ? 0: sizeof(VertexId) * degree;

        lli queue_size = global_embedding_queue_size_[0];
        lli used_graph_data_size = global_used_graph_data_buffer_size_[0];

        if (
                queue_size + 1 > max_num_embeddings_global_queue_ || 
                used_graph_data_size + required_graph_data_buffer_size > global_graph_data_buffer_size_
           ) {
            flush_all_extendable_embeddings();

            queue_size = global_embedding_queue_size_[0];
            used_graph_data_size = global_used_graph_data_buffer_size_[0];

            assert(queue_size + 1 <= max_num_embeddings_local_queue_);
            assert(used_graph_data_size + required_graph_data_buffer_size <= global_graph_data_buffer_size_);
        }

        CompactExtendableEmbedding * e = global_embedding_queues_[0] + queue_size;
        e->parent = nullptr;
        e->new_vertex_graph_data = is_local ? local_graph_->get_neighbours_ptx(v): ((VertexId*) required_graph_data_buffer_size);
        e->new_vertex = v;
        e->new_vertex_degree = degree;
        e->new_vertex_partition_id = partition_id;
        e->cached_object_offset = 0;
        e->cached_object_size = 0;

        global_embedding_queue_size_[0] = queue_size + 1;
        global_used_graph_data_buffer_size_[0] += required_graph_data_buffer_size;
    }

    void EmbeddingExplorationEngine::scatter_entry_level_extendable_embedding(
                    const VertexId new_v, 
                    const bool is_new_v_active, 
                    const uint32_t cached_obj_size,
                    ExtendableEmbedding &parent,
                    const Context context,
                    const bool is_finalized 
                    ) {
        VertexId degree = local_graph_->get_degree(new_v);
        int partition_id = local_graph_->get_vertex_master_partition(new_v);
        bool is_local = partition_id == partition_id_;
        lli required_graph_data_buffer_size = is_local ? 0: sizeof(VertexId) * degree;

        lli queue_size = global_embedding_queue_size_[0];
        lli used_graph_data_size = global_used_graph_data_buffer_size_[0];
        lli used_intermediate_data_size = global_used_intermediate_buffer_size_[0];

        if (queue_size + 1 > max_num_embeddings_global_queue_ ||
                used_graph_data_size + required_graph_data_buffer_size > global_graph_data_buffer_size_ ||
                used_intermediate_data_size + cached_obj_size > global_intermediate_buffer_size_) {
            flush_all_extendable_embeddings();

            queue_size = global_embedding_queue_size_[0];
            used_graph_data_size = global_used_graph_data_buffer_size_[0];
            used_intermediate_data_size = global_used_intermediate_buffer_size_[0];

            assert(queue_size + 1 <= max_num_embeddings_global_queue_);
            assert(used_graph_data_size + required_graph_data_buffer_size <= global_graph_data_buffer_size_);
            assert(used_intermediate_data_size + cached_obj_size <= global_intermediate_buffer_size_);
        }

        CompactExtendableEmbedding * e = global_embedding_queues_[0] + queue_size;
        e->parent = parent.compact_version;
        e->new_vertex_graph_data = is_local ? local_graph_->get_neighbours_ptx(new_v): ((VertexId*) required_graph_data_buffer_size);
        e->new_vertex = new_v;
        e->new_vertex_degree = degree;
        e->new_vertex_partition_id = partition_id;
        e->cached_object_offset = global_used_intermediate_buffer_size_[0];
        e->cached_object_size = cached_obj_size;

        global_embedding_queue_size_[0] = queue_size + 1;
        global_used_graph_data_buffer_size_[0] += required_graph_data_buffer_size;
        global_used_intermediate_buffer_size_[0] += cached_obj_size;
    }

    void EmbeddingExplorationEngine::suspend_thread(const int thread_id) {
        {
            std::lock_guard<std::mutex> lk(num_suspended_comp_threads_mutex_[next_depth_ - 1]);
            if (global_phases_[next_depth_ - 1] != local_phases_[next_depth_ - 1][thread_id][0]) {
                global_phases_[next_depth_ - 1] = local_phases_[next_depth_ - 1][thread_id][0];
                num_suspended_threads_[next_depth_ - 1] = 1;
            } else {
                num_suspended_threads_[next_depth_ - 1] += 1;
            }
            if (num_suspended_threads_[next_depth_ - 1] == num_computation_threads_) {
                // wake up the main thread
                num_suspended_comp_threads_cv_[next_depth_ - 1].notify_one();
            }
        }
        pthread_barrier_wait(&comp_thread_barrier_[next_depth_ - 1]);
        local_phases_[next_depth_ - 1][thread_id][0] ^= 1;
    }

    //uint64_t last_cycle = 0;

    void EmbeddingExplorationEngine::scatter( 
            const VertexId new_v, 
            const bool is_new_v_active, 
            const uint32_t cached_obj_size,
            ExtendableEmbedding &parent,
            const Context context,
            const bool is_finalized 
            ) {
        if (! is_finalized) {
#ifdef ENABLE_BREAKDOWN_PROFILING
            *(context.scatter_time) -= get_time();
#endif
            int thread_id = context.thread_id;
            int socket_id = context.socket_id;

            VertexId degree = local_graph_->get_degree(new_v);
            int partition_id = is_new_v_active ? local_graph_->get_vertex_master_partition(new_v): partition_id_;
            bool is_local = partition_id == partition_id_;
            VertexId hash = new_v & cache_vertex_id_mask_;  
            bool is_cached = is_cache_line_valid_[hash] && cached_vertices_[hash] == new_v;
            bool need_graph_data_buffer = ! is_local && ! is_cached;
            uint64_t required_graph_data_buffer_size = need_graph_data_buffer ? degree * sizeof(VertexId): 0;

            lli local_queue_size = local_embedding_queue_size_[thread_id][next_depth_];
            lli local_graph_size = local_needed_graph_data_buffer_size_[thread_id][next_depth_];
            lli local_intermediate_size = local_needed_intermediate_buffer_size_[thread_id][next_depth_];

            if ( 
                    local_queue_size + 1 > max_num_embeddings_local_queue_ ||
                    local_graph_size + required_graph_data_buffer_size > local_graph_data_buffer_size_ ||
                    local_intermediate_size + cached_obj_size > local_intermediate_data_buffer_size_
               ) {
                int r = flush_local_embeddings(thread_id, next_depth_);
                while (r == -1) {
                    suspend_thread(thread_id);
                    r = flush_local_embeddings(thread_id, next_depth_);
                }
                local_queue_size = local_embedding_queue_size_[thread_id][next_depth_];
                local_graph_size = local_needed_graph_data_buffer_size_[thread_id][next_depth_];
                local_intermediate_size = local_needed_intermediate_buffer_size_[thread_id][next_depth_];
            }
            // push the compact extendable embedding to the local queue
            CompactExtendableEmbedding * e = local_embedding_queues_[thread_id][next_depth_] + local_queue_size;
            e->parent = parent.compact_version;
            e->new_vertex_graph_data = is_local? local_graph_->get_neighbours_ptx_relaxed(new_v): ((VertexId*) required_graph_data_buffer_size);
            e->new_vertex = new_v;
            e->new_vertex_degree = degree;
            e->new_vertex_partition_id = partition_id;
            e->cached_object_size = cached_obj_size;
            // update the queue meta information
            local_embedding_queue_size_[thread_id][next_depth_] = local_queue_size + 1;
            local_needed_graph_data_buffer_size_[thread_id][next_depth_] = local_graph_size + required_graph_data_buffer_size;
            local_needed_intermediate_buffer_size_[thread_id][next_depth_] = local_intermediate_size + cached_obj_size;
#ifdef ENABLE_BREAKDOWN_PROFILING
            *(context.scatter_time) += get_time();
#endif
        } else {
            //assert(is_new_v_active == false);
            // the cached object is placed on the stack
            uint8_t cached_obj[cached_obj_size]; // if the runtime stack overflow, please increase the stack capacity
            ExtendableEmbedding &e = parent;
            e.matched_vertices_nbrs_ptx[e.size] = nullptr;
            e.matched_vertices[e.size] = new_v;
            e.cached_objects[e.size] = (void*) cached_obj;
            e.size ++;
            this->apply_fun_(e, context, this);
            e.size --;
        }
    }

    // WorkloadDistributer
    
    void WorkloadDistributer::fetch_next_batch_local(int s_i, Workload &workload, bool disable_output) {
        VertexId boundary;
        if (! restrict_starting_vertex_label_) {
            boundary = num_local_vertices_[s_i];
        } else {
            boundary = local_labeled_vertices_idx_[s_i][starting_vertex_label_ + 1]
                - local_labeled_vertices_idx_[s_i][starting_vertex_label_];
            //printf("%u %u %u\n", starting_vertex_label_, local_labeled_vertices_idx_[s_i][starting_vertex_label_ + 1], local_labeled_vertices_idx_[s_i][starting_vertex_label_]);
        }
        //printf("Boundary: %u\n", boundary);
        VertexId pos;
        VertexId delta = batch_size;

        progress_win_mutex_.lock();
        int ret = MPI_Fetch_and_op(&delta, &pos, DistributedSys::get_mpi_data_type<VertexId>(),
                node_id_, s_i, MPI_SUM, progress_win_);
        assert(ret == MPI_SUCCESS);
        MPI_Win_flush(node_id_, progress_win_);
        progress_win_mutex_.unlock();
        //printf("pos: %u\n", pos);

        if (pos >= boundary) {
            workload.num_v_0 = 0;
            workload.v_0_list = nullptr;
        } else {
            VertexId v_0_idx_begin = pos;
            VertexId v_0_idx_end = std::min(pos + batch_size, boundary);
            workload.num_v_0 = v_0_idx_end - v_0_idx_begin;
            if (! restrict_starting_vertex_label_) {
                workload.v_0_list = local_vertices_[s_i] + v_0_idx_begin;
            } else {
                workload.v_0_list = local_labeled_vertices_[s_i] + local_labeled_vertices_idx_[s_i][starting_vertex_label_] + v_0_idx_begin;
            }
            
            if (! disable_output) {
                printf("Node %d, Socket %d: Scatterring vertex embeddings: %u / %u = %.3f, batch_size: %u\n",
                        node_id_, s_i, v_0_idx_end, boundary, (double) v_0_idx_end / boundary, batch_size);
            }
        }
    }

    void WorkloadDistributer::next_remote_partition(int s_i) {
        remote_partition_id_[s_i] += 1;
        remote_partition_id_[s_i] %= num_partitions_;
    }
    
    void WorkloadDistributer::fetch_next_batch_remote(int s_i, Workload &workload, bool disable_output) {
        workload.num_v_0 = 0;
        workload.v_0_list = nullptr;

        int partition_id = node_id_ * num_sockets_ + s_i;

        while (remote_partition_id_[s_i] != partition_id) {
            int remote_p_i = remote_partition_id_[s_i];
            int remote_node_id = remote_p_i / num_sockets_;
            int remote_socket_id = remote_p_i % num_sockets_;

            VertexId boundary;
            if (! restrict_starting_vertex_label_) {
                boundary = num_local_vertices_per_partition_[remote_p_i];
            } else {
                boundary = global_labeled_vertices_idx_[remote_node_id][remote_socket_id][starting_vertex_label_ + 1] -
                    global_labeled_vertices_idx_[remote_node_id][remote_socket_id][starting_vertex_label_];
            }
            VertexId pos;
            VertexId delta = batch_size;


            // since multiple sockets on the same machine may use the MPI
            // windows concurrently
            progress_win_mutex_.lock();
            int ret = MPI_Fetch_and_op(
                    &delta, &pos, DistributedSys::get_mpi_data_type<VertexId>(),
                    remote_node_id, remote_socket_id, MPI_SUM, progress_win_
                    );
            assert(ret == MPI_SUCCESS);
            MPI_Win_flush(remote_node_id, progress_win_);
            progress_win_mutex_.unlock();

            if (pos < boundary) {
                VertexId v_0_idx_begin = pos;
                VertexId v_0_idx_end = std::min(pos + batch_size, boundary);
                workload.num_v_0 = v_0_idx_end - v_0_idx_begin;
                workload.v_0_list = stolen_vertices_buff_[s_i];

                if (! restrict_starting_vertex_label_) {
                    MPI_Get(
                            stolen_vertices_buff_[s_i], v_0_idx_end - v_0_idx_begin,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            remote_node_id, 
                            v_0_idx_begin, v_0_idx_end - v_0_idx_begin,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            local_vertices_win_[remote_socket_id]
                           );
                    MPI_Win_flush(remote_node_id, local_vertices_win_[remote_socket_id]);
                } else {
                    MPI_Get(
                            stolen_vertices_buff_[s_i], v_0_idx_end - v_0_idx_begin,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            remote_node_id, 
                            global_labeled_vertices_idx_[remote_node_id][remote_socket_id][starting_vertex_label_] + v_0_idx_begin,
                            v_0_idx_end - v_0_idx_begin,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            local_labeled_vertices_win_[remote_socket_id]
                           );
                    MPI_Win_flush(remote_node_id, local_labeled_vertices_win_[remote_socket_id]);
                }
                if (! disable_output) {
                    printf("Node %d socket %d stolen tasks %u v_0s from node %d socket %d\n", 
                            node_id_, s_i, v_0_idx_end - v_0_idx_begin, remote_node_id, remote_socket_id);
                }
                break;
            } else {
                next_remote_partition(s_i);
            }
        }
    }

    void WorkloadDistributer::reset_workload(
            bool restrict_starting_vertex_label,
            LabelId starting_vertex_label
            ) {
        // reset the progress
        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            curr_progress_[s_i] = 0;
            remote_partition_id_[s_i] = node_id_ * num_sockets_ + s_i;
            next_remote_partition(s_i);
        }
        // reset the batch size 
        batch_size = MAX_WORKLOAD_BATCH_SIZE;
        for (int i = 1; i <= max_pattern_size_; ++ i) {
            batch_size /= i;
        }
        if (batch_size == 0) {
            batch_size = 1;
        }
        restrict_starting_vertex_label_ = restrict_starting_vertex_label;
        starting_vertex_label_ = starting_vertex_label;
    }

    bool WorkloadDistributer::fetch_next_batch(int s_i, Workload &workload, bool disable_output) {
        fetch_next_batch_local(s_i, workload, disable_output);
        if (workload.num_v_0 != 0) {
            return workload.v_0_list + workload.num_v_0 != local_vertices_[s_i] + num_local_vertices_[s_i];
        }
        fetch_next_batch_remote(s_i, workload, disable_output);
        return false;
    }

    WorkloadDistributer::WorkloadDistributer(int node_id, int num_nodes, int num_sockets,
            int max_pattern_size, DistGraph * graphs) {
        node_id_ = node_id;
        num_nodes_ = num_nodes;
        num_sockets_ = num_sockets;
        num_partitions_ = num_nodes_ * num_sockets_;
        max_pattern_size_ = max_pattern_size;

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            stolen_vertices_buff_[s_i] = (VertexId*) numa_alloc_onnode(
                    sizeof(VertexId) * MAX_WORKLOAD_BATCH_SIZE, s_i
                    );
            memset(stolen_vertices_buff_[s_i], 0, sizeof(VertexId) * MAX_WORKLOAD_BATCH_SIZE);
        }

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            graphs_[s_i] = &graphs[s_i];
        }

        reset_workload();
        
        // create the windows here
        MPI_Win_create(curr_progress_, sizeof(VertexId) * num_sockets_,
                sizeof(VertexId), MPI_INFO_NULL, MPI_COMM_WORLD, &progress_win_);
        // passive synchronization
        for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
            MPI_Win_lock(MPI_LOCK_SHARED, n_i, 0, progress_win_);
        }

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            num_local_vertices_[s_i] = graphs_[s_i]->get_num_local_vertices();
        }
        MPI_Allgather(
                num_local_vertices_, num_sockets_,
                DistributedSys::get_mpi_data_type<VertexId>(),
                num_local_vertices_per_partition_, num_sockets_,
                DistributedSys::get_mpi_data_type<VertexId>(),
                MPI_COMM_WORLD
                );
        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            assert(num_local_vertices_[s_i] == num_local_vertices_per_partition_[node_id_ * num_sockets_ + s_i]);
        }

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            local_vertices_[s_i] = graphs_[s_i]->get_local_vertices();
            MPI_Win_create(
                    local_vertices_[s_i], sizeof(VertexId) * num_local_vertices_[s_i],
                    sizeof(VertexId), MPI_INFO_NULL, MPI_COMM_WORLD, &local_vertices_win_[s_i]
                    );
            for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
                MPI_Win_lock(MPI_LOCK_SHARED, n_i, 0, local_vertices_win_[s_i]);
            }
        }

        // supporting labeled graphs
        if (graphs_[0]->is_labeled_graph()) {
            LabelId num_labels = graphs_[0]->get_num_labels();
            for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                assert(graphs_[s_i]->is_labeled_graph());
                local_labeled_vertices_[s_i] = (VertexId*) numa_alloc_onnode(
                        sizeof(VertexId) * num_local_vertices_[s_i], s_i
                        );
                local_labeled_vertices_idx_[s_i] = (VertexId*) numa_alloc_onnode(
                        sizeof(VertexId) * (num_labels + 1), s_i
                        );
                VertexId num_vertices = num_local_vertices_[s_i];
                VertexId * vertices = local_labeled_vertices_[s_i];
                VertexId * vertices_idx = local_labeled_vertices_idx_[s_i];
                memset(vertices_idx, 0, sizeof(VertexId) * (num_labels + 1));
                for (VertexId v_idx = 0; v_idx < num_vertices; ++ v_idx) {
                    VertexId v = local_vertices_[s_i][v_idx];
                    LabelId l = graphs_[s_i]->get_vertex_label(v);
                    vertices_idx[l + 1] ++;
                }
                for (LabelId l = 1; l <= num_labels; ++ l) {
                    vertices_idx[l] += vertices_idx[l - 1];
                }
                //for (LabelId l = 0; l <= num_labels; ++ l) {
                //    printf("%u ", vertices_idx[l]);
                //}
                //printf("\n");
                VertexId * tmp_idx = new VertexId [num_labels + 1];
                memcpy(tmp_idx, vertices_idx, sizeof(VertexId) * (num_labels + 1));
                for (VertexId v_idx = 0; v_idx < num_vertices; ++ v_idx) {
                    VertexId v = local_vertices_[s_i][v_idx];
                    LabelId l = graphs_[s_i]->get_vertex_label(v);
                    vertices[tmp_idx[l] ++] = v;
                }
                delete [] tmp_idx;
                MPI_Win_create(
                        local_labeled_vertices_[s_i], sizeof(VertexId) * num_local_vertices_[s_i],
                        sizeof(VertexId), MPI_INFO_NULL, MPI_COMM_WORLD, &local_labeled_vertices_win_[s_i]
                        );
                for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
                    MPI_Win_lock(MPI_LOCK_SHARED, n_i, 0, local_labeled_vertices_win_[s_i]);
                }
                memcpy(global_labeled_vertices_idx_[node_id][s_i], vertices_idx, 
                        sizeof(VertexId) * (num_labels + 1));
            }
            for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
                for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                    MPI_Bcast(
                            global_labeled_vertices_idx_[n_i][s_i], num_labels + 1,
                            DistributedSys::get_mpi_data_type<VertexId>(),
                            n_i, MPI_COMM_WORLD
                            );
                }
            }
        }
    }

    WorkloadDistributer::~WorkloadDistributer() {
        for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
            MPI_Win_unlock(n_i, progress_win_);
        }
        MPI_Win_free(&progress_win_);

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
                MPI_Win_unlock(n_i, local_vertices_win_[s_i]);
            }
            MPI_Win_free(&local_vertices_win_[s_i]);
        }

        for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
            numa_free(stolen_vertices_buff_[s_i], sizeof(VertexId) * batch_size);
        }

        if (graphs_[0]->is_labeled_graph()) {
            for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                LabelId num_labels = graphs_[s_i]->get_num_labels();
                for (int n_i = 0; n_i < num_nodes_; ++ n_i) {
                    MPI_Win_unlock(n_i, local_labeled_vertices_win_[s_i]);
                }
                MPI_Win_free(&local_labeled_vertices_win_[s_i]);
                numa_free(local_labeled_vertices_[s_i], sizeof(VertexId) * num_local_vertices_[s_i]);
                numa_free(local_labeled_vertices_idx_[s_i], sizeof(VertexId) * (num_labels + 1));
            }
        }
    }

    // PerformanceMetricLogger
    
    std::vector<PerformanceMetric> PerformanceMetricLogger::collected_metrics;

    void PerformanceMetricLogger::clear_metrics() {
        collected_metrics.clear();
    }

    void PerformanceMetricLogger::report_metric(std::string metric_name,
            std::string metric_unit, double value) {
        PerformanceMetric metric;
        metric.metric_name = metric_name;
        metric.metric_unit = metric_unit;
        metric.value = value;
        collected_metrics.push_back(metric);
    }

    void PerformanceMetricLogger::print_metircs() {
        int num_collected_metrics = collected_metrics.size();
        int num_nodes = DistributedSys::get_instance()->get_num_nodes();
        int node_id = DistributedSys::get_instance()->get_node_id();

        // verification
        int global_max, global_min;
        MPI_Allreduce(&num_collected_metrics, &global_max, 1,
                MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&num_collected_metrics, &global_min, 1,
                MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        assert(num_collected_metrics == global_max);
        assert(num_collected_metrics == global_min);

        // printing the metrics
        if (node_id == 0) {
            printf("\n******************** Performance Metrics ********************\n");
            printf("MetricName,\tMetricAverage,\tMetricMax,\tMetricMin,\tMetricUnit\n");
        }
        std::sort(collected_metrics.begin(), collected_metrics.end());
        for (int i = 0; i < num_collected_metrics; ++ i) {
            PerformanceMetric metric = collected_metrics[i];
            double metric_average, metric_max, metric_min;
            MPI_Allreduce(&metric.value, &metric_average, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            metric_average /= double(num_nodes);
            MPI_Allreduce(&metric.value, &metric_max, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&metric.value, &metric_min, 1, 
                    MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

            if (node_id == 0) {
                printf("%s,\t%.3f,\t%.3f,\t%.3f,\t%s\n",
                        metric.metric_name.c_str(),
                        metric_average, metric_max,metric_min, 
                        metric.metric_unit.c_str());
            }
        }
        if (node_id == 0) {
            printf("*************************************************************\n");
        }
    }

    // LocalPerformanceMetric
    
    std::mutex LocalPerformanceMetric::mutex;
    double LocalPerformanceMetric::aggregated_request_sender_bandwidth;
    double LocalPerformanceMetric::aggregated_request_sender_volume;
    double LocalPerformanceMetric::aggregated_request_sender_remote_socket_volume;
    double LocalPerformanceMetric::aggregated_request_sender_remote_socket_bandwidth;
    double LocalPerformanceMetric::request_sender_time;
    double LocalPerformanceMetric::aggregated_request_handler_bandwidth;
    double LocalPerformanceMetric::aggregated_request_handler_volume;
    double LocalPerformanceMetric::comm_time_on_critical_path;
    double LocalPerformanceMetric::shuffle_time;
    double LocalPerformanceMetric::memory_consumption;
    double LocalPerformanceMetric::load_graph_time;
    double LocalPerformanceMetric::sender_thread_useful_work_time;
    double LocalPerformanceMetric::sender_thread_communication_time;
    double LocalPerformanceMetric::actual_computation_time;
    double LocalPerformanceMetric::cpu_utilization;
    double LocalPerformanceMetric::num_extensions;
    double LocalPerformanceMetric::cache_hit_rate;
}






