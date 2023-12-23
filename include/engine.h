#ifndef ENGINE_H
#define ENGINE_H

#include <unistd.h>
#include <numa.h>
#include <pthread.h>

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "graph.h"
#include "utilities.h"
#include "timer.h"
#include "distributed_sys.h"
#include "shared_memory_sys.h"
#include "graph_loader.h"

//#define HUGE_GRAPH // 

#define MAX_PATTERN_SIZE 5
#define MAX_DEPTH MAX_PATTERN_SIZE
#define MAX_NUM_INTERMEDIATE_OBJS 5
#define MAX_MEMORY_SIZE ((size_t) 58 * 1024 * 1024 * 1024) // at most 58GB memory consumption (our own cluster) TODO
//#define MAX_MEMORY_SIZE ((size_t) 116 * 1024 * 1024 * 1024) // at most 116GB memory consumption (USC-HPC)
//#define MAX_MEMORY_SIZE ((size_t) 230 * 1024 * 1024 * 1024) // at most 112GB memory consumption (USC-HPC)
//#define MAX_MEMORY_SIZE ((size_t) 240 * 1024 * 1024 * 1024) // at most 240GB memory consumption (USC-HPC epyc nodes)

#ifdef HUGE_GRAPH
    //#define DATA_SIZE_PER_REQUEST (256 * 1024 * 1024) 
    //#define CHUNK_SIZE 4
    #define SHARE_DEGREE_ARRAY (true) // TODO share the degree arrary across different sockets to reduce memory footprint
    #define MAX_WORKLOAD_BATCH_SIZE (32768 * 4 * 32) // huge graphs have more vertices
    #define QUEUE_SIZE_SCALE 2
#else
    //#define DATA_SIZE_PER_REQUEST (64 * 1024 * 1024) 
    //#define CHUNK_SIZE 64
    #define SHARE_DEGREE_ARRAY (false)
    #define MAX_WORKLOAD_BATCH_SIZE (32768 * 4)
    #define QUEUE_SIZE_SCALE 1
#endif

#define DATA_SIZE_PER_REQUEST (16 * 1024 * 1024) 
#define CHUNK_SIZE 64  
#define SMALL_CHUNK_SIZE 4

// system parameters
#define LOCAL_SCRATCHPAD_SIZE (64 * 1024 * 1024)  
#define GRAPH_DATA_BUFFER_PER_LEVEL ((long long) 1 * 1024 * 1024 * 1024)
#define MAX_NUM_PARTITIONS (MAX_NUM_SOCKETS * MAX_NUM_NODES)
#define MAX_NUM_VERTICES_PER_REQUEST (64 * 1024 * 1024)
#define L1_DCACHE_SIZE (32 * 1024)
#define L2_CACHE_SIZE (256 * 1024)
#define LLC_CACHE_SIZE (20 * 1024 * 1024)
//#define RELATIVE_CACHE_SIZE 0.15

#define HASH_A ((EdgeId) 10000723)
#define HASH_B ((EdgeId) 10001269)
#define HASH_MASK ((EdgeId) 0xFFFFFF) // 24-bit
#define MAX_VID_BITS ((EdgeId) 31) // the number of vertices should be less than 2 billion
//#define ENABLE_BREAKDOWN_PROFILING //TODO: disable profiling when necessary

//const int chunk_size = CHUNK_SIZE;

typedef HashBasedNumaAwareDistributedCSRGraph DistGraph;
typedef JointHashBasedNumaAwareDistGraphLoader DistGraphLoader;
typedef long long int lli;

struct Context {
    int socket_id;
    int thread_id;
    int depth;
    double * scatter_time;
} __attribute__((packed));

struct NumaAwareBuffer {
    lli buffer_size;
    uint8_t ** buffer_pointer;
};

class NumaAwareBufferManager {
    private:
        std::vector<NumaAwareBuffer> ** registered_buffers; // [num_sockets_]
        lli * total_registered_buffer_size; // [num_sockets_]
        int num_sockets_;
        void ** allocated_ptx_;

    public:
        NumaAwareBufferManager(int num_sockets);
        ~NumaAwareBufferManager();

        void register_numa_aware_buffer(lli buffer_size, uint8_t ** buffer_pointer, int socket_id);
        void allocate_numa_aware_buffers();
        void deallocate_numa_aware_buffers(); // must be called after allocate_numa_aware_buffers()
        lli get_total_allocated_size();

        template<typename T>
            T* alloc_numa_aware_array(int array_size, int socket_id) {
                void * ptx = numa_alloc_onnode(sizeof(T) * array_size, socket_id);
                assert(ptx != NULL);
                return (T*) ptx;
            }
};

struct Pattern {
    PVertexId pattern_size;
    PEdgeId num_edges;
    uint16_t adj_matrix[MAX_PATTERN_SIZE]; // 2 x 8 = 16 bytes
    PVertexId degree[MAX_PATTERN_SIZE];

    inline void init() {
        num_edges = 0;
        memset(adj_matrix, 0, sizeof(adj_matrix));
        memset(degree, 0, sizeof(degree));
    }
    Pattern() {
        pattern_size = 0;
        init();
    }
    Pattern(PVertexId p_size) {
        pattern_size = p_size;
        init();
    }
    inline bool operator == (const Pattern &other) const {
        if (pattern_size != other.pattern_size || num_edges != other.num_edges) {
            return false;
        }
        for (PVertexId v_i = 0; v_i < pattern_size; ++ v_i) {
            if (adj_matrix[v_i] != other.adj_matrix[v_i]) {
                return false;
            }
        }
        return true;
    }

    inline uint16_t has_edge(PVertexId src, PVertexId dst) const {
        assert(src < MAX_PATTERN_SIZE && dst < MAX_PATTERN_SIZE);
        return adj_matrix[src] & (1 << dst);
    }
    inline void add_edge(PVertexId src, PVertexId dst) {
        bool not_connected = has_edge(src, dst) == 0;
        num_edges += not_connected;
        degree[src] += not_connected;
        degree[dst] += not_connected;
        adj_matrix[src] |= (1 << dst);
        adj_matrix[dst] |= (1 << src);
    }
    inline void clear_edges() {
        num_edges = 0;
        memset(adj_matrix, 0, sizeof(adj_matrix));
        memset(degree, 0, sizeof(degree));
    }
    inline bool is_connected() const {
        if (pattern_size == 0) {
            return true;
        }
        // BFS
        bool is_visited[MAX_PATTERN_SIZE];
        memset(is_visited, 0, sizeof(is_visited));
        PVertexId num_visited_vertices = 1;
        PVertexId queue[MAX_PATTERN_SIZE];
        PVertexId queue_head = 0, queue_tail = 0;
        queue[queue_tail ++] = 0;
        is_visited[0] = true;
        while (queue_head < queue_tail) {
            PVertexId v = queue[queue_head ++];
            for (PVertexId u = 0; u < pattern_size; ++ u) {
                if (has_edge(v, u) && !is_visited[u]) {
                    is_visited[u] = true;
                    ++ num_visited_vertices;
                    queue[queue_tail ++] = u;
                }
            }
        }
        assert(num_visited_vertices <= pattern_size);
        return num_visited_vertices == pattern_size;
    }
    inline bool is_clique() const {
        PEdgeId expected_num_edges = (PEdgeId) pattern_size * (pattern_size - 1) / 2;
        return expected_num_edges == num_edges;
    }
    inline bool is_cycle() const {
        bool flag = is_connected() && pattern_size >= 4;
        for (PVertexId i = 0; i < pattern_size; ++ i) {
            flag = flag && (degree[i] == 2);
        }
        return flag;
    }
    inline int get_multiplicity() const {
        PermutationList permutation_list = PermutationGenerator::get_instance()->get_permutations(pattern_size);
        int multiplicity = 0;
        int * permutation = permutation_list.permutations;
        for (int i = 0; i < permutation_list.num_permutations; ++ i, permutation += permutation_list.length) {
            bool is_equal = true;
            for (int src = 0; src < pattern_size; ++ src) {
                for (int dst = 0; dst < src; ++ dst) {
                    if (has_edge(src, dst)) {
                        if (! has_edge(permutation[src], permutation[dst])) {
                            is_equal = false;
                            break;
                        }
                    }
                }
                if (! is_equal) {
                    break;
                }
            }
            if (is_equal) {
                ++ multiplicity;
            }
        }
        return multiplicity;
    }
    std::string to_string() const {
        std::string s = "Pattern: (size:" + std::to_string(pattern_size) + ", num_edges:" + std::to_string(num_edges)
            + ") ";
        for (PVertexId src = 0; src < pattern_size; ++ src) {
            for (PVertexId dst = src; dst < pattern_size; ++ dst) {
                if (has_edge(src, dst)) {
                    s += "(" + std::to_string(src) + ", " + std::to_string(dst) + ")";
                }
            }
        }
        return s;
    }

};

struct LabeledPattern {
    LabelId labels[MAX_PATTERN_SIZE];
    int size;
    bool operator==(const LabeledPattern &other) const {
        bool is_equal = size == other.size;
        for (int i = 0; i < size; ++ i) {
            is_equal = is_equal && labels[i] == other.labels[i];
        }
        return is_equal;
    }
} __attribute__((packed));

class LabeledPatternHashTable {
    private:
        const size_t hash_v = 17;
        const double max_occupancy = 0.75;

        size_t table_size_;
        size_t num_occupied_entries_;
        bool * is_valid_;
        LabeledPattern * entries_;

        size_t get_hash_value(const LabeledPattern &p) {
            size_t hash = 0;
            for (int i = 0; i < p.size; ++ i) {
                hash *= hash_v;
                hash += (size_t) p.labels[i];
            }
            hash %= table_size_;
            return hash;
        }

    public:
        void clear() {
            memset(is_valid_, 0, sizeof(bool) * table_size_);
            num_occupied_entries_ = 0;
        }

        LabeledPatternHashTable(size_t table_size = 1000000) {
            table_size_ = table_size;
            is_valid_ = new bool [table_size_];
            entries_ = new LabeledPattern [table_size_];
            clear();
        }
        ~LabeledPatternHashTable() {
            delete [] is_valid_;
            delete [] entries_;
        }

        void insert(const LabeledPattern p) {
            num_occupied_entries_ ++;
            double occupancy = (double) num_occupied_entries_ / table_size_;
            assert(occupancy < max_occupancy);

            size_t idx = get_hash_value(p);
            for (; is_valid_[idx]; idx = (idx + 1) % table_size_) {
                if (entries_[idx] == p) return ; // the entry already exists
            }
            is_valid_[idx] = true;
            entries_[idx] = p;
        }
        bool is_exist(const LabeledPattern p) {
            size_t idx = get_hash_value(p);
            for (; is_valid_[idx]; idx = (idx + 1) % table_size_) {
                if (entries_[idx] == p) {
                    return true;
                }
            }
            return false;
        }
};

class SharedMemoryPatternDomain {
    private:
        uint32_t valid_tag_; // used to support O(1) clearing 

        int max_pattern_size_;
        int num_threads_;
        int num_sockets_;
        int num_threads_per_socket_;
        VertexId num_vertices_;

        uint32_t * is_mapped_[MAX_PATTERN_SIZE][MAX_NUM_THREADS]; // uint32_t[max_pattern_size][num_threads][num_vertices]
        VertexId * mapped_vertices_[MAX_PATTERN_SIZE][MAX_NUM_THREADS]; // VertexId[max_pattern_size][num_threads][num_vertices]
        VertexId num_mapped_vertices_[MAX_PATTERN_SIZE][MAX_NUM_THREADS][32]; // padding to avoid false cache sharing
        NumaAwareBufferManager * buff_manager_;

        void allocate_buffer() {
            buff_manager_ = new NumaAwareBufferManager(num_sockets_);
            for (int i = 0; i < max_pattern_size_; ++ i) {
                for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                    int s_i = t_i / num_threads_per_socket_;
                    buff_manager_->register_numa_aware_buffer(
                            sizeof(uint32_t) * num_vertices_, 
                            (uint8_t**) &is_mapped_[i][t_i],
                            s_i
                            );
                    buff_manager_->register_numa_aware_buffer(
                            sizeof(VertexId) * num_vertices_,
                            (uint8_t**) &mapped_vertices_[i][t_i],
                            s_i
                            );
                }
            }
            buff_manager_->allocate_numa_aware_buffers(); // the buffer allocator will zero out the buffers
        }
        void release_buffer() {
            buff_manager_->deallocate_numa_aware_buffers();
            delete buff_manager_;
        }

    public:
        SharedMemoryPatternDomain(int max_pattern_size, int num_threads, int num_sockets, VertexId num_vertices): 
            max_pattern_size_(max_pattern_size), num_threads_(num_threads), num_sockets_(num_sockets), num_vertices_(num_vertices) {
                num_threads_per_socket_ = num_threads_ / num_sockets_;
                assert(num_threads_ % num_sockets_ == 0);
                valid_tag_ = 0;
                allocate_buffer();
            }
        ~SharedMemoryPatternDomain() {
            release_buffer();
        }

        inline void clear() {
            valid_tag_ += 1;
            if (valid_tag_ == 0) {
                valid_tag_ ++;
                for (int i = 0; i < max_pattern_size_; ++ i) {
                    for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                        memset(is_mapped_[i][t_i], 0, sizeof(uint32_t) * num_vertices_);
                    }
                }
            }
            for (int i = 0; i < max_pattern_size_; ++ i) {
                for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                    num_mapped_vertices_[i][t_i][0] = 0;
                }
            }
        }
        inline void insert(
                int pattern_vertex,
                int thread_id, 
                VertexId input_graph_vertex,
                bool enable // used to avoid branch
                ) {
            mapped_vertices_[pattern_vertex][thread_id][num_mapped_vertices_[pattern_vertex][thread_id][0]]
                = input_graph_vertex;
            num_mapped_vertices_[pattern_vertex][thread_id][0] += (enable && 
                    is_mapped_[pattern_vertex][thread_id][input_graph_vertex] < valid_tag_);
            is_mapped_[pattern_vertex][thread_id][input_graph_vertex] = valid_tag_ - 
                (! enable && is_mapped_[pattern_vertex][thread_id][input_graph_vertex] < valid_tag_);
        }
        inline VertexId get_support(int pattern_size) {
            VertexId support = num_vertices_;
            valid_tag_ += 1;
            assert(valid_tag_ != 0);
            for (int i = 0; i < pattern_size; ++ i) {
                VertexId mapped_vertices = 0;
                uint32_t * is_mapped = is_mapped_[i][0];
                for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                    VertexId * v_ptx = mapped_vertices_[i][t_i];
                    VertexId num_mapped_vertices = num_mapped_vertices_[i][t_i][0];
                    for (VertexId v_idx = 0; v_idx < num_mapped_vertices; ++ v_idx) {
                        VertexId v = *(v_ptx ++);
                        mapped_vertices += (is_mapped[v] < valid_tag_);
                        is_mapped[v] = valid_tag_;
                    }
                }
                support = std::min(support, mapped_vertices);
            }
            return support;
        }
};

class DistributedPatternDomain {
    private:
        uint32_t valid_tag_; // used to support O(1) clearing 

        int max_pattern_size_;
        int num_threads_;
        int num_sockets_;
        int num_threads_per_socket_;
        VertexId num_vertices_;

        uint32_t * is_mapped_[MAX_PATTERN_SIZE][MAX_NUM_THREADS]; // uint32_t[max_pattern_size][num_threads][num_vertices]
        VertexId * mapped_vertices_[MAX_PATTERN_SIZE][MAX_NUM_THREADS]; // VertexId[max_pattern_size][num_threads][num_vertices]
        VertexId num_mapped_vertices_[MAX_PATTERN_SIZE][MAX_NUM_THREADS][32]; // padding to avoid false cache sharing
        NumaAwareBufferManager * buff_manager_;

        void allocate_buffer() {
            buff_manager_ = new NumaAwareBufferManager(num_sockets_);
            for (int i = 0; i < max_pattern_size_; ++ i) {
                for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                    int s_i = t_i / num_threads_per_socket_;
                    buff_manager_->register_numa_aware_buffer(
                            sizeof(uint32_t) * num_vertices_, 
                            (uint8_t**) &is_mapped_[i][t_i],
                            s_i
                            );
                    buff_manager_->register_numa_aware_buffer(
                            sizeof(VertexId) * num_vertices_,
                            (uint8_t**) &mapped_vertices_[i][t_i],
                            s_i
                            );
                }
            }
            buff_manager_->allocate_numa_aware_buffers(); // the buffer allocator will zero out the buffers
        }
        void release_buffer() {
            buff_manager_->deallocate_numa_aware_buffers();
            delete buff_manager_;
        }
    public:
        DistributedPatternDomain(
                int max_pattern_size, int num_threads, int num_sockets, VertexId num_vertices
                ): max_pattern_size_(max_pattern_size), num_threads_(num_threads), num_sockets_(num_sockets), num_vertices_(num_vertices) {
            num_threads_per_socket_ = num_threads_ / num_sockets_;
            assert(num_threads_ % num_sockets_ == 0);
            valid_tag_ = 0;
            allocate_buffer();
        }
        ~DistributedPatternDomain() {
            release_buffer();
        }

        inline void clear() {
            valid_tag_ += 1;
            if (valid_tag_ == 0) {
                valid_tag_ ++;
                for (int i = 0; i < max_pattern_size_; ++ i) {
                    for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                        memset(is_mapped_[i][t_i], 0, sizeof(uint32_t) * num_vertices_);
                    }
                }
            }
            for (int i = 0; i < max_pattern_size_; ++ i) {
                for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                    num_mapped_vertices_[i][t_i][0] = 0;
                }
            }
        }
        inline void insert(
                Context context,
                int pattern_vertex,
                VertexId input_graph_vertex,
                bool enable
                ) {
            int t_i = context.socket_id * num_threads_per_socket_ + context.thread_id;
            mapped_vertices_[pattern_vertex][t_i][num_mapped_vertices_[pattern_vertex][t_i][0]]
                = input_graph_vertex;
            num_mapped_vertices_[pattern_vertex][t_i][0] += (enable && 
                    is_mapped_[pattern_vertex][t_i][input_graph_vertex] < valid_tag_);
            is_mapped_[pattern_vertex][t_i][input_graph_vertex] = valid_tag_ - 
                (! enable && is_mapped_[pattern_vertex][t_i][input_graph_vertex] < valid_tag_);
        }
        inline VertexId get_support(int pattern_size) {
            int node_id = DistributedSys::get_instance()->get_node_id();
            int num_nodes = DistributedSys::get_instance()->get_num_nodes();
            //printf("****** Node %d, Calculating support ******\n", node_id);
            VertexId support = num_vertices_;
            Context context;
            context.socket_id = context.thread_id = 0;
            VertexId * recv_buff;
            if (num_threads_ > 1) {
                recv_buff = mapped_vertices_[0][1];
            } else {
                recv_buff = new VertexId [num_vertices_];
            }
            double local_aggregation_time = 0.;
            double distributed_aggregation_time = 0.;
            double comm_t = 0.;
            double comp_t = 0.;
            for (int i = 0; i < pattern_size; ++ i) {
                // aggregate the results across all threads
                local_aggregation_time -= get_time();
                for (int t_i = 1; t_i < num_threads_; ++ t_i) {
                    VertexId * v_ptx = mapped_vertices_[i][t_i];
                    VertexId num_mapped_vertices = num_mapped_vertices_[i][t_i][0];
                    for (VertexId v_idx = 0; v_idx < num_mapped_vertices; ++ v_idx) {
                        VertexId v = *(v_ptx ++);
                        insert(context, i, v, true);
                    }
                }
                local_aggregation_time += get_time();
                // aggregate the results across all nodes
                // two possbile strategies: 
                // 1) all nodes send the data to node 0 for aggregation O(N)
                // 2) use an binary aggregation tree O(log N)
                distributed_aggregation_time -= get_time();
                int disp = 1;
                for (; disp < num_nodes; disp <<= 1) {
                    if (node_id % (disp << 1) == 0) { // the receiver
                        int source_node = node_id + disp;
                        if (source_node < num_nodes) {
                            VertexId num_mapped_vertices;
                            comm_t -= get_time();
                            MPI_Status status;
                            MPI_Recv(
                                    &num_mapped_vertices, 1, DistributedSys::get_mpi_data_type<VertexId>(),
                                    source_node, UserDataAggregation, MPI_COMM_WORLD, &status
                                    );
                            MPI_Recv(
                                    recv_buff, num_mapped_vertices, DistributedSys::get_mpi_data_type<VertexId>(),
                                    source_node, UserDataAggregation, MPI_COMM_WORLD, &status
                                    );
                            comm_t += get_time();
                            comp_t -= get_time();
                            VertexId * v_ptx = recv_buff;
                            for (VertexId v_idx = 0; v_idx < num_mapped_vertices; ++ v_idx) {
                                VertexId v = *(v_ptx ++);
                                insert(context, i, v, true);
                            }
                            comp_t += get_time();
                        }
                    } else { // the sender
                        comm_t -= get_time();
                        int target_node = node_id - disp;
                        assert(target_node >= 0);
                        MPI_Send(
                                &num_mapped_vertices_[i][0][0], 1, DistributedSys::get_mpi_data_type<VertexId>(),
                                target_node, UserDataAggregation, MPI_COMM_WORLD
                                );
                        MPI_Send(
                                mapped_vertices_[i][0], num_mapped_vertices_[i][0][0], DistributedSys::get_mpi_data_type<VertexId>(),
                                target_node, UserDataAggregation, MPI_COMM_WORLD
                                );
                        comm_t += get_time();
                        break;
                    }
                }
                comm_t -= get_time();
                MPI_Barrier(MPI_COMM_WORLD);
                comm_t += get_time();
                if (node_id == 0) {
                    support = std::min(support, num_mapped_vertices_[i][0][0]);
                }
                distributed_aggregation_time += get_time();
            }
            MPI_Bcast(
                    &support, 1, DistributedSys::get_mpi_data_type<VertexId>(), 
                    0, MPI_COMM_WORLD
                    );
            if (num_threads_ <= 1) {
                delete [] recv_buff;
            }
            //printf("node_id: %d, Local aggregation time: %.3f (s), distributed aggregation time: %.3f (comm: %.3f, comp: %.3f) (s)\n",
            //        node_id, local_aggregation_time, distributed_aggregation_time,comm_t, comp_t);
            return support;
        }
};

namespace Khuzdul {

    class AbstractEmbeddingExplorationEngine;
    class EmbeddingExplorationEngine;
    class DistributedApplication;
    struct ExtendableEmbedding;

    typedef void(*ApplyFunction)(ExtendableEmbedding&, const Context, AbstractEmbeddingExplorationEngine*);
    typedef std::function<void(const VertexId, const bool, const uint32_t, ExtendableEmbedding&, const Context, const bool)> ScatterFunction;

    inline void init_cached_objs(VertexId * start_ptx) {
        *start_ptx = *(start_ptx + 1) = 0;
    }
    
    inline VertexId* create_cached_obj(VertexId * start_ptx, VertexId size) {
        VertexId * data_ptx = start_ptx + MAX_NUM_INTERMEDIATE_OBJS + 2;
        VertexId num_objs = *start_ptx;
        assert(num_objs < MAX_NUM_INTERMEDIATE_OBJS);
        VertexId allocated_size = start_ptx[num_objs + 1];
        VertexId * new_obj_ptx = data_ptx + allocated_size;
        start_ptx[num_objs + 2] = allocated_size + size;
        (*start_ptx) ++;
        return new_obj_ptx;
    }
    
    inline VertexId* get_cached_obj(VertexId * start_ptx, int obj_id) {
        VertexId * data_ptx = start_ptx + MAX_NUM_INTERMEDIATE_OBJS + 2;
        VertexId num_objs = *start_ptx;
        assert(obj_id < num_objs);
        VertexId obj_disp = start_ptx[obj_id + 1];
        return data_ptx + obj_disp;
    }

    template<typename T>
        class Aggregator {
            private:
                const int padding = 128; // TO avoid false sharing
                int num_threads_;
                int num_sockets_;
                T * local_reducers_[MAX_NUM_SOCKETS][MAX_NUM_THREADS];
            public:
                inline void clear() {
                    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                        for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                            memset(local_reducers_[s_i][t_i], 0, sizeof(T));
                        }
                    }
                }
                inline T evaluate() {
                    T reducer = 0;
                    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                        for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                            reducer += *local_reducers_[s_i][t_i];
                        }
                    }

                    T global_reducer = 0;
                    MPI_Allreduce(
                            &reducer, &global_reducer, 1,
                            DistributedSys::get_mpi_data_type<T>(),
                            MPI_SUM, MPI_COMM_WORLD
                            );
                    return global_reducer;
                }
                inline void add(int s_i, int t_i, T delta) {
                    *local_reducers_[s_i][t_i] += delta;
                }
                inline void add(Context context, T delta) {
                    add(context.socket_id, context.thread_id, delta);
                }

                Aggregator(int num_sockets, int num_threads) {
                    num_sockets_ = num_sockets;
                    num_threads_ = num_threads;
                    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                        for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                            local_reducers_[s_i][t_i] = (T*) numa_alloc_onnode(
                                    sizeof(T) * (1 + padding / sizeof(T)), s_i
                                    );
                        }
                    }
                    clear();
                }
                ~Aggregator() {
                    for (int s_i = 0; s_i < num_sockets_; ++ s_i) {
                        for (int t_i = 0; t_i < num_threads_; ++ t_i) {
                            numa_free(local_reducers_[s_i][t_i], 
                                    sizeof(T) * (1 + padding / sizeof(T)));
                        }
                    }
                }
        };

    // communication threads
    // graph data request sender/receiver
    struct GraphDataRequestMetaData {
        lli graph_data_disp;
        lli num_requested_vertices;
        int request_sender_socket_id;
        int request_sender_node_id;
    } __attribute__((packed));

    // target: request receiver
    inline int get_request_meta_data_tag(int target_socket_id) {
        return GraphDataRequestMetaTag * MAX_NUM_SOCKETS * MAX_NUM_SOCKETS
            + target_socket_id;
    }

    // source: request sender
    // target: request receiver
    inline int get_request_vertices_data_tag(int source_socket_id, int target_socket_id) {
        return GraphDataRequestVerticesTag * MAX_NUM_SOCKETS * MAX_NUM_SOCKETS
            + MAX_NUM_SOCKETS * source_socket_id + target_socket_id;
    }

    // source: respond sender
    // target: respond receiver
    inline int get_respond_data_tag(int source_socket_id, int target_socket_id) {
        return GraphDataRespondTag * MAX_NUM_SOCKETS * MAX_NUM_SOCKETS
            + MAX_NUM_SOCKETS * source_socket_id + target_socket_id;
    }
   
    class GraphDataRequestSender { 
        private:
            const lli graph_data_size_per_request = DATA_SIZE_PER_REQUEST;
            const lli num_vertices_per_request = MAX_NUM_VERTICES_PER_REQUEST;

            EmbeddingExplorationEngine * engine_;
            int socket_id_;
            int num_sockets_;
            std::function<void()> thread_entry_;
            std::thread * thread_;
            volatile bool is_terminated_;

            // data structures used to filter away duplicated requests (horizontal data reuse)
            // some simplified calculation:
            // if hash_set_size = 1M
            // the amount of used memory is 1M * (4 + 4 + 8 + 4) = 1M * 16 = 20M 
            // acceptable
            // this parameter is also used in the graph data cache
            const int num_confused_bits = 4; // 2^num_confused_bits vertex ids shall the same hash value
            uint64_t hash_set_size_;
            VertexId vertex_id_mask_;

            NumaAwareBufferManager * buff_manager_;

            uint32_t curr_generation_[MAX_DEPTH]; // using the generation method to avoid btimap resetting cost
            uint32_t * previous_request_generation_[MAX_DEPTH]; // uint32_t[hash_set_size]
            VertexId * previous_request_vertex_id_[MAX_DEPTH]; // VertexId[has_set_size]
            VertexId ** previous_request_graph_buffer_[MAX_DEPTH]; // uint8_t*[hash_set_size]

            // the data structure used to maintain the cache without replacement
            double relative_cache_size_;
            VertexId cache_degree_threshold;
            uint64_t max_cached_graph_data_size_; // unit: sizeof(VertexId)
            uint64_t cached_graph_data_size_;
            VertexId * cached_graph_data_; // VertexId[max_cached_graph_data_size_]
            bool * is_cache_line_valid_; // bool [hash_set_size_]
            VertexId * cached_vertices_; // VertexId[hash_set_size]
            VertexId ** cache_line_data_ptx_; // VertexId*[hash_set_size]
            uint64_t num_cache_gets_;
            uint64_t num_cacge_hits_;

            int get_next_request_batch(
                    lli & num_requested_vertices, 
                    lli & requested_graph_data_size,
                    int & target_node_id,
                    int & previous_target_node_id,
                    lli & graph_data_disp,
                    lli graph_data_start_addr,
                    lli & curr_embedding_idx,
                    VertexId * requested_vertices
                    );
            void thread_main();

        public:
            inline VertexId get_cache_vertex_id_mask() {
                return vertex_id_mask_;
            }
            inline bool * get_is_cache_line_valid() {
                return is_cache_line_valid_;
            }
            inline VertexId * get_cached_vertices() {
                return cached_vertices_;
            }

            GraphDataRequestSender(EmbeddingExplorationEngine * engine, VertexId cache_degree_threshold, double relative_cache_size);
            ~GraphDataRequestSender();
            void clear_cache(); 
    };

    class GraphDataRequestHandler {
        private:
            const lli graph_data_size_per_request = DATA_SIZE_PER_REQUEST;
            const lli num_vertices_per_request = MAX_NUM_VERTICES_PER_REQUEST;
            const lli comm_batch_size = DATA_SIZE_PER_REQUEST / 32; 

            EmbeddingExplorationEngine * engine_;
            DistGraph * graph_;
            int socket_id_;
            int num_sockets_;
            std::function<void()> thread_entry_;
            std::thread * thread_;
            volatile bool is_terminated_;

            void thread_main();
        public:
            GraphDataRequestHandler(EmbeddingExplorationEngine * engine);
            ~GraphDataRequestHandler();
    };

    // in total takes 36 bytes:
    // L1-cache size = 36KB => can accommodate 32KB/36B=910 compressed embedding
    struct CompactExtendableEmbedding {
        CompactExtendableEmbedding * parent; // 8 byte
        // new vertex information
        VertexId * new_vertex_graph_data; // 8 byte
        uint32_t new_vertex; // 4 byte
        uint32_t new_vertex_degree; // 4 byte
        int32_t new_vertex_partition_id; // 4 byte, -1: the new vertex is not active
        // object caching related information
        uint32_t cached_object_offset; // 4 byte
        uint32_t cached_object_size; // 4 byte
    } __attribute__((packed));

    // the core internal abstraction separting pattern matching algorithms
    // with distributed execution
    struct ExtendableEmbedding {
        // the embedding itself && adjacent graph data of active vertices
        VertexId * matched_vertices_nbrs_ptx[MAX_PATTERN_SIZE];
        VertexId matched_vertices_num_nbrs[MAX_PATTERN_SIZE];
        VertexId matched_vertices[MAX_PATTERN_SIZE];
        int size;

        // cached objects
        void * cached_objects[MAX_PATTERN_SIZE];

        CompactExtendableEmbedding * compact_version;

        // users are only allow to use these functions to access an extendable embedding object
        int get_size() {
            return size;
        }
        VertexId get_matched_vertex(int idx) {
            assert(idx >= 0 && idx < size);
            return matched_vertices[idx];
        }
        VertexSet get_matched_vertex_nbrs(int idx) {
            assert(idx >= 0 && idx < size);
            assert(matched_vertices_nbrs_ptx[idx] != nullptr && "Requested the graph data of a vertex marked as inactive!");
            VertexSet nbrs(matched_vertices_nbrs_ptx[idx], 
                    matched_vertices_num_nbrs[idx]);
            return nbrs;
        }
        void * get_cached_obj(int dist) {
            return cached_objects[size - dist - 1];
        }
    };

    enum GlobalEmbeddingQueueState {
        Filling = 1,
        PartialReady = 2
    };

    class AbstractEmbeddingExplorationEngine {
        public: 
            virtual void scatter( 
                    const VertexId new_v, 
                    const bool is_new_v_active, 
                    const uint32_t cached_obj_size,
                    ExtendableEmbedding &parent,
                    const Context context,
                    const bool is_finalized = false // finalized embedding will no longer has new active vertices
                    ) = 0;
            virtual void* alloc_thread_local_scratchpad_buffer(const Context &context, const size_t &size) = 0;
            virtual void clear_thread_local_scratchpad_buffer(const Context &context) = 0;
            virtual LabelId get_vertex_label(const VertexId v) = 0;
    };

    //template<typedef ApplyFuncType>
    class EmbeddingExplorationEngine: AbstractEmbeddingExplorationEngine {
        private:
            friend class GraphDataRequestSender; 
            friend class GraphDataRequestHandler;
            friend class DistributedApplication;

            size_t local_scratchpad_size_;

            ApplyFunction apply_fun_;

            // engines existing on the same node across different sockets
            EmbeddingExplorationEngine ** engines_same_node_ = NULL; // EmbeddingExplorationEngine* [num_sockets]

            // some performance metrics
            double shuffle_time_;
            double comm_time_on_critical_path_;
            double actual_computation_time_;

            // graph 
            DistGraph * local_graph_; // The Graph partition resides in the local socket
            int num_sockets_;
            int socket_id_;
            int next_depth_;

            int num_partitions_;
            int partition_id_;

            // communication threads 
            GraphDataRequestSender * graph_data_request_sender_;
            MPI_Win win_; // used to perform one-sided graph data transferring
            GraphDataRequestHandler * graph_data_request_handler_;

            // threads
            int num_computation_threads_;

            // global embedding queues && managed buffers
            int max_depth_; // depth = 0...max_depth
            lli max_num_embeddings_global_queue_;
            lli global_intermediate_buffer_size_;
            lli global_graph_data_buffer_size_;

            GlobalEmbeddingQueueState global_queue_states_[MAX_DEPTH];

            // NUMA-aware embedding queue && buffer
            NumaAwareBufferManager * buff_manager_;

            CompactExtendableEmbedding * global_embedding_queues_[MAX_DEPTH];
            CompactExtendableEmbedding * global_shuffled_embedding_queues_[MAX_DEPTH];
            uint8_t * global_graph_data_buffer_[MAX_DEPTH];
            uint8_t * global_intermediate_buffer_[MAX_DEPTH];

            lli global_embedding_queue_size_[MAX_DEPTH];
            lli global_num_ready_embeddings_[MAX_DEPTH];

            lli global_used_graph_data_buffer_size_[MAX_DEPTH];
            lli global_used_intermediate_buffer_size_[MAX_DEPTH];

            // mutex protecting the global data (extendable embedding) layout
            std::mutex global_embedding_queue_mutex_;

            // thread-local data structures
            // the scatterred extendable embeddings are firstly stored in a local
            // queue, in order to reduce the number of accesses to the global queue
            // and hence avoid mutex contention
            lli max_num_embeddings_local_queue_; // this paramter should be carefully choosen
            lli local_graph_data_buffer_size_;
            lli local_intermediate_data_buffer_size_;

            // padding to avoid L1/L2-cache-line false sharing
            CompactExtendableEmbedding * local_embedding_queues_[MAX_NUM_THREADS][MAX_DEPTH + 128];
            lli local_embedding_queue_size_[MAX_NUM_THREADS][MAX_DEPTH + 128];
            lli local_needed_graph_data_buffer_size_[MAX_NUM_THREADS][MAX_DEPTH + 128];
            lli local_needed_intermediate_buffer_size_[MAX_NUM_THREADS][MAX_DEPTH + 128];

            // thread local scratchpad 
            uint8_t * thread_local_scratchpad_[MAX_NUM_THREADS][MAX_DEPTH];
            size_t allocated_scratchpad_size_[MAX_NUM_THREADS][MAX_DEPTH + 128];

            // the lightweight graph data cache
            VertexId cache_vertex_id_mask_;
            bool * is_cache_line_valid_;
            VertexId * cached_vertices_;

            // the size of level-0 extendable embeddings
            int level_0_emebeding_size_;

            // CV and mutex used to managing computation threads
            std::mutex num_suspended_comp_threads_mutex_[MAX_DEPTH];
            std::condition_variable num_suspended_comp_threads_cv_[MAX_DEPTH];
            volatile int num_suspended_threads_[MAX_DEPTH];
            volatile int global_phases_[MAX_DEPTH];
            int local_phases_[MAX_DEPTH][MAX_NUM_THREADS][128]; // there could be some false sharing concerning this array => however it is fine since the operations updating this array are rare
            // we do not use std::barrier since it is only supported after C++20
            pthread_barrier_t comp_thread_barrier_[MAX_DEPTH];

            size_t engine_data_size_;
            lli num_extensions_;

            int chunk_size_;

            void init_buffers();
            void release_buffer();
            void shuffle_global_embedding_queue(int depth);
            void clear_global_queue(int depth);
            int flush_local_embeddings(int thread_id, int depth);
            void change_global_queue_state_to_partial_ready(int depth);
            void extend_embeddings(int depth);

        public:
            void* alloc_thread_local_scratchpad_buffer(const Context &context, const size_t &size);
            void clear_thread_local_scratchpad_buffer(const Context &context);
            void clear_all_buffers();
            void clear_graph_data_cache();
            void set_engines_on_the_same_node(EmbeddingExplorationEngine ** engines) {
                engines_same_node_ = engines;
            }
            inline void set_level_0_embedding_size(int level_0_emebeding_size) {
                level_0_emebeding_size_ = level_0_emebeding_size;
            }
            inline VertexId get_degree(VertexId vtx) {
                return local_graph_->get_degree(vtx);
            }
            inline VertexId get_max_degree() {
                return local_graph_->get_max_degree();
            }
            inline VertexId get_num_hub_vertices() {
                return local_graph_->get_num_hub_vertices();
            }
            inline VertexId* get_hub_vertices() {
                return local_graph_->get_hub_vertices();
            }
            inline int get_vertex_master_partition(VertexId v) {
                return local_graph_->get_vertex_master_partition(v);
            }
            LabelId get_vertex_label(const VertexId v) {
                return local_graph_->get_vertex_label(v);
            }

            EmbeddingExplorationEngine(
                    //DistributedApplication * dist_app,
                    DistGraph * local_graph,
                    lli max_depth, 
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
                    );
            ~EmbeddingExplorationEngine();

            void flush_all_extendable_embeddings();
            void scatter_vertex_extendable_embedding(const VertexId v); // must be size-1
            // can be not size-1
            void scatter_entry_level_extendable_embedding(
                    const VertexId new_v, 
                    const bool is_new_v_active, 
                    const uint32_t cached_obj_size,
                    ExtendableEmbedding &parent,
                    const Context context,
                    const bool is_finalized = false // finalized embedding will no longer has new active vertices
                    );
            void suspend_thread(const int thread_id);
            void scatter( 
                    const VertexId new_v, 
                    const bool is_new_v_active, 
                    const uint32_t cached_obj_size,
                    ExtendableEmbedding &parent,
                    const Context context,
                    const bool is_finalized = false // finalized embedding will no longer has new active vertices
                    );
    };

    struct Workload {
        VertexId * v_0_list;
        VertexId num_v_0;
    } __attribute__((packed));

    // one per node shared by multiple execution engines
    class WorkloadDistributer {
        private:
            VertexId batch_size = MAX_WORKLOAD_BATCH_SIZE;
            VertexId * stolen_vertices_buff_[MAX_NUM_SOCKETS]; // one per socket

            int node_id_;
            int num_nodes_;
            int num_sockets_;
            int num_partitions_;
            int max_pattern_size_;

            // the progress of all execution engines on this node
            VertexId curr_progress_[MAX_NUM_SOCKETS]; 
            MPI_Win progress_win_;
            std::mutex progress_win_mutex_;

            // tracking the node && socket being stolen
            int remote_partition_id_[MAX_NUM_SOCKETS];

            DistGraph * graphs_[MAX_NUM_SOCKETS];
            VertexId num_local_vertices_[MAX_NUM_SOCKETS];
            VertexId num_local_vertices_per_partition_[MAX_NUM_SOCKETS * MAX_NUM_NODES];

            VertexId * local_vertices_[MAX_NUM_SOCKETS];
            MPI_Win local_vertices_win_[MAX_NUM_SOCKETS];

            // supporting labeled graphs
            VertexId * local_labeled_vertices_[MAX_NUM_SOCKETS];
            VertexId * local_labeled_vertices_idx_[MAX_NUM_SOCKETS];
            VertexId global_labeled_vertices_idx_[MAX_NUM_NODES][MAX_NUM_SOCKETS][MAX_NUM_LABELS + 1];
            MPI_Win local_labeled_vertices_win_[MAX_NUM_SOCKETS];
            bool restrict_starting_vertex_label_;
            LabelId starting_vertex_label_;

            void fetch_next_batch_local(int s_i, Workload &workload, bool disable_output = false);
            void next_remote_partition(int s_i);
            void fetch_next_batch_remote(int s_i, Workload &workload, bool disable_output = false);

        public:
            // remember to call this function before each run
            void reset_workload(bool restrict_starting_vertex_label = false, LabelId starting_vertex_label = 0);

            // return value:
            // true: there will be more local workload
            // false: no more local workload
            bool fetch_next_batch(int s_i, Workload &workload, bool disable_output = false);

            WorkloadDistributer(int node_id, int num_nodes, int num_sockets,
                    int max_pattern_size, DistGraph * graphs);
            ~WorkloadDistributer();
    };

    struct PerformanceMetric {
        std::string metric_name;
        std::string metric_unit;
        double value;
        bool operator<(const PerformanceMetric &other) const {
            int r = metric_name.compare(other.metric_name);
            return r < 0;
        }
    };

    class PerformanceMetricLogger {
        private:
            static std::vector<PerformanceMetric> collected_metrics;
        public:
            static void clear_metrics();
            static void report_metric(std::string metric_name,
                    std::string metric_unit, double value);
            static void print_metircs();
    };
    
    class LocalPerformanceMetric {
        public:
            static std::mutex mutex;
            static double aggregated_request_sender_bandwidth;
            static double aggregated_request_sender_volume;
            static double aggregated_request_sender_remote_socket_bandwidth;
            static double aggregated_request_sender_remote_socket_volume;
            static double aggregated_request_handler_bandwidth;
            static double aggregated_request_handler_volume;
            static double comm_time_on_critical_path;
            static double shuffle_time;
            static double request_sender_time;
            static double memory_consumption;
            static double load_graph_time;
            static double sender_thread_useful_work_time;
            static double sender_thread_communication_time;
            static double actual_computation_time;
            static double cpu_utilization;
            static double num_extensions;
            static double cache_hit_rate;

            static inline void init_local_metrics() {
                aggregated_request_sender_bandwidth = 0;
                aggregated_request_sender_volume = 0;
                aggregated_request_sender_remote_socket_bandwidth = 0;
                aggregated_request_sender_remote_socket_volume = 0;
                aggregated_request_handler_bandwidth = 0;
                aggregated_request_handler_volume = 0;
                comm_time_on_critical_path = 0;
                shuffle_time = 0;
                request_sender_time = 0;
                sender_thread_useful_work_time = 0;
                sender_thread_communication_time = 0;
                actual_computation_time = 0;
                cpu_utilization = 0;
                num_extensions = 0;
                cache_hit_rate = 0;
                PerformanceMetricLogger::clear_metrics();
            }

            static inline void print_metrics(int num_runs) {
                int node_id = DistributedSys::get_instance()->get_node_id();

                PerformanceMetricLogger::report_metric(
                        "RequestSenderNetworkBandwidth", "MBps",
                        aggregated_request_sender_bandwidth
                        );

                aggregated_request_sender_volume /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "RequestSenderNetworkVolume", "MB",
                        aggregated_request_sender_volume
                        );

                request_sender_time = aggregated_request_sender_volume / aggregated_request_sender_bandwidth * 1000.;
                PerformanceMetricLogger::report_metric(
                        "RequestSenderNetworkCommTime", "ms",
                        request_sender_time
                        );

                PerformanceMetricLogger::report_metric(
                        "RequestSenderRemoteSocketBandwidth", "MBps",
                        aggregated_request_sender_remote_socket_bandwidth
                        );

                aggregated_request_sender_remote_socket_volume /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "RequestSenderRemoteSocketVolume", "MB",
                        aggregated_request_sender_remote_socket_volume
                        );

                double request_sender_remote_socket_time = aggregated_request_sender_remote_socket_volume / aggregated_request_sender_remote_socket_bandwidth * 1000.;
                PerformanceMetricLogger::report_metric(
                        "RequestSenderRemoteSocketTime", "ms",
                        request_sender_remote_socket_time
                        );

                PerformanceMetricLogger::report_metric(
                        "RequestHandlerBandwidth", "MBps",
                        aggregated_request_handler_bandwidth
                        );

                aggregated_request_handler_volume /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "RequestHandlerVolume", "MB",
                        aggregated_request_handler_volume
                        );

                comm_time_on_critical_path /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "CommTimeCritalPath", "ms",
                        comm_time_on_critical_path
                        );

                shuffle_time /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "ShuffleTime", "ms",
                        shuffle_time
                        );
                PerformanceMetricLogger::report_metric(
                        "MemUsage", "MB",
                        memory_consumption
                        );
                PerformanceMetricLogger::report_metric(
                        "LoadGraphTime", "min",
                        load_graph_time
                        );

                sender_thread_useful_work_time /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "CommThreadUsefulWorkTime", "ms",
                        sender_thread_useful_work_time * 1000.
                        );
                sender_thread_communication_time /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "CommThreadNetworkTime", "ms",
                        sender_thread_communication_time * 1000.
                        );

#ifdef ENABLE_BREAKDOWN_PROFILING
                actual_computation_time /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                        "ActualComputationTime", "ms",
                        actual_computation_time
                        );
#endif

                cpu_utilization /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                    "CPUUtilization", "%",
                    cpu_utilization
                );

                num_extensions /= double(num_runs);
                PerformanceMetricLogger::report_metric(
                    "NumExtensions", "",
                    num_extensions
                );

                PerformanceMetricLogger::report_metric(
                    "CacheHitRate", "%",
                    cache_hit_rate 
                );

                PerformanceMetricLogger::print_metircs();
            }
    };

}

#endif






