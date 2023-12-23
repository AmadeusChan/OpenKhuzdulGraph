#ifndef GRAPH_LOADER_H
#define GRAPH_LOADER_H

#include "graph.h"

#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h> 
#include <sys/mman.h>
#include <string.h>
#include <numa.h>

#include <algorithm>
#include <string>
#include <random>
#include <vector>
#include <functional>

#include "utilities.h"
#include "shared_memory_sys.h"
#include "dwarves_debug.h"
#include "distributed_sys.h"
#include "timer.h"

#define MAX_NUM_LABELS 48
#define PAGESIZE 4096

/*
 * Graph Dataset Format
 * [Graph Meta Data] [Graph Node Labels] [Graph Edge List]
 * The edges are sorted by firstly src, then dst
 */

struct GraphMetaDataStructOnDisk {
    VertexId num_vertices;
    EdgeId num_edges;
    int has_node_label;
    int has_edge_label;
} __attribute__((packed));

// this class is used to distributed a large block of data 
// to all machines in the cluster from node 0
class BlockDataDistributer {
    private:
        uint8_t * data_block_;
        size_t block_size_;
    public:
        BlockDataDistributer(
                size_t block_size = (size_t) 1024 * 1024 * 1024 // the block size is 1GB by default
                ): block_size_(block_size) {
            data_block_ = new uint8_t [block_size];
        }
        ~BlockDataDistributer() {
            delete [] data_block_;
        }
        void distribute_data(
                uint8_t * src,
                size_t total_data_size,
                std::function<void(uint8_t*, size_t)> process_data
                ) {
            int num_nodes = DistributedSys::get_instance()->get_num_nodes();
            int node_id = DistributedSys::get_instance()->get_node_id();

            double start_time = get_time();

            size_t processed_data_size = 0;
            int block_id = 0;
            while (processed_data_size < total_data_size) {
                MPI_Barrier(MPI_COMM_WORLD);
                size_t data_size_to_read = std::min(
                        total_data_size - processed_data_size, block_size_);
                if (! node_id) {
                    printf("Distributing the %d-th block [%.3f GB, %.3f GB).\n", 
                            block_id, processed_data_size / 1024. / 1024. / 1024.,
                            (processed_data_size + data_size_to_read) / 1024. / 1024. / 1024.);
                    fflush(stdout);
                }
                block_id ++;
                // load the data to node 0
                if (node_id == 0) {
                    printf("    Loading to node 0...\n");
                    fflush(stdout);
                    memcpy(data_block_, src + processed_data_size, data_size_to_read);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                if (! node_id) {
                    printf("    Broadcasting the data block from node 0...\n");
                    fflush(stdout);
                }
                MPI_Bcast(data_block_, data_size_to_read, MPI_CHAR, 0, MPI_COMM_WORLD);
                if (! node_id) {
                    printf("    Finished broadcasting.\n");
                    printf("    Processing the data block...\n");
                    fflush(stdout);
                }
                // process the distributed data
                process_data(data_block_, data_size_to_read);
                processed_data_size += data_size_to_read;
                if (! node_id) {
                    double curr_time = get_time();
                    double time_elasped = curr_time - start_time;
                    double estimated_remained_time = time_elasped / processed_data_size * total_data_size - time_elasped;
                    printf("    Finished processing, time elasped %.3f seconds, estimated time left: %.3f seconds\n",
                            time_elasped, estimated_remained_time);
                }
            }
        }
};

template<typename NodeLabel, typename EdgeLabel>
class CSRGraphLoader {
    protected:
        template<typename EdgeLabelOnDisk>
            void construct_csr_format(
                    EdgeStruct<EdgeLabelOnDisk> * edge_list, 
                    EdgeId num_edges,
                    EdgeId * &csr_idx, 
                    VertexId * &csr_list_vtx,
                    EdgeLabel * &csr_list_label,
                    VertexId &max_degree_to_return,
                    VertexId num_vertices,
                    VertexId * degree
                    ) {
                if (! std::is_same<EdgeLabel, Empty>::value) {
                    assert((std::is_same<EdgeLabelOnDisk, EdgeLabel>::value));
                }
#pragma omp parallel for 
                for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
                    if (e_i == 0 || (e_i != 0 && edge_list[e_i].src != edge_list[e_i - 1].src)) {
                        csr_idx[edge_list[e_i].src] = e_i;
                    }
                    csr_list_vtx[e_i] = edge_list[e_i].dst;
                    if (! std::is_same<EdgeLabel, Empty>::value) { 
                        csr_list_label[e_i] = edge_list[e_i].label;
                    }
                }
                csr_idx[num_vertices] = num_edges;

                VertexId max_degree = 0;
#pragma omp parallel for reduction(max: max_degree) 
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    if (csr_idx[v_i + 1] - csr_idx[v_i] > max_degree) {
                        max_degree = csr_idx[v_i + 1] - csr_idx[v_i];
                    }
                    degree[v_i] = csr_idx[v_i + 1] - csr_idx[v_i];
                }
                max_degree_to_return = max_degree;
            }

        template<typename EdgeLabelOnDisk>
            void construct_csr_format_for_empty(
                    EdgeStruct<EdgeLabelOnDisk> * edge_list, 
                    EdgeId num_edges,
                    EdgeId * &csr_idx, 
                    VertexId * &csr_list_vtx,
                    EdgeLabel * &csr_list_label,
                    VertexId &max_degree_to_return,
                    VertexId num_vertices,
                    VertexId * degree
                    ) {
                assert((std::is_same<EdgeLabel, Empty>::value));

                Debug::get_instance()->log("constructing csr_idx");
                EdgeId edge_idx = 0;
                EdgeId e_seg = num_edges / 100;
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    for (; edge_idx < num_edges && edge_list[edge_idx].src < v_i; ++ edge_idx) {
                        if (edge_idx % e_seg == 0) {
                            printf("      Progress %.3f\n", 1. * (edge_idx + 1) / num_edges);
                            fflush(stdout);
                        }
                    }
                    csr_idx[v_i] = edge_idx;
                }
                csr_idx[num_vertices] = num_edges;

                Debug::get_instance()->log("copying edge data...");
                for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
                    csr_list_vtx[e_i] = edge_list[e_i].dst;
                    if (e_i % e_seg == 0) {
                        printf("      Progress %.3f\n", 1. * (e_i + 1) / num_edges);
                        fflush(stdout);
                    }
                }

                Debug::get_instance()->log("calculating degrees...");
                VertexId max_degree = 0;
#pragma omp parallel for reduction(max: max_degree) 
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    if (csr_idx[v_i + 1] - csr_idx[v_i] > max_degree) {
                        max_degree = csr_idx[v_i + 1] - csr_idx[v_i];
                    }
                    degree[v_i] = csr_idx[v_i + 1] - csr_idx[v_i];
                }
                max_degree_to_return = max_degree;
                printf("max degree: %u\n", max_degree);
                Debug::get_instance()->log("finished CSR construction");
            }


    public:
        // you are able to load a labeled graph datasets on disk to a unlabeled graph in memory
        // however, you cannot load a labeled in-memory graph from a unlabeled on-disk graph
        // which means, 
        // if NodeLabel is not Empty, the has_node_label flag in on-disk dataset should be 1
        // So as EdgeLabel
        void load_graph(const std::string &file_name, CSRGraph<NodeLabel, EdgeLabel> & graph, bool use_out_of_core_feature = false, std::string swap_files_path = "", bool avoid_copy_edgelist = false) {
            Debug::get_instance()->enter_function("CSRGraphLoader::load_graph");

            if (use_out_of_core_feature) {
                Debug::get_instance()->log("out-of-core feature is enabled.");
            }

            VertexId num_vertices;
            EdgeId num_edges;
            VertexId max_degree;
            EdgeId * csr_idx;
            VertexId * csr_list_vtx;
            VertexId * degree;
            EdgeLabel * csr_list_label = nullptr;
            NodeLabel * node_label_list = nullptr;
            VertexId num_labels;

            assert(file_exits(file_name));
            int f = open(file_name.c_str(), O_RDONLY);
            assert(f != -1);

            GraphMetaDataStructOnDisk graph_meta_data_on_disk;
            read_file(f, (uint8_t*) &graph_meta_data_on_disk, sizeof(GraphMetaDataStructOnDisk));
            if (! (std::is_same<NodeLabel, Empty>::value)) {
                assert(graph_meta_data_on_disk.has_node_label == 1);
                assert(sizeof(NodeLabel) == 4); // Only Support 32-bit Node Label
            }
            if (! (std::is_same<EdgeLabel, Empty>::value)) {
                assert(graph_meta_data_on_disk.has_edge_label == 1);
                assert(sizeof(EdgeLabel) == 4); // Only Support 32-bit Edge Label
            }
            int has_node_label = graph_meta_data_on_disk.has_node_label;
            int has_edge_label = graph_meta_data_on_disk.has_edge_label;

            num_vertices = graph_meta_data_on_disk.num_vertices;
            num_edges = graph_meta_data_on_disk.num_edges;
            num_labels = 1;

            if (has_node_label) {
                if (use_out_of_core_feature) {
                    printf("ERROR: out-of-core feature is not supported when node labes is presented.\n");
                    assert(false);
                }
                if (! std::is_same<NodeLabel, Empty>::value) {
                    node_label_list = new NodeLabel [num_vertices]; 
                    read_file(f, (uint8_t*) node_label_list, sizeof(NodeLabel) * num_vertices);
                } else { // if the application doesn't needed the node labels, just skip it
                    //Debug::get_instance()->log("skipping the node label list...");
                    assert(lseek(f, sizeof(VertexId) * num_vertices, SEEK_CUR) != -1);
                }
            }

            csr_idx = new EdgeId [num_vertices + 1];
            if (! use_out_of_core_feature) {
                csr_list_vtx = new VertexId [num_edges];
            } else {
                std::string swap_file = swap_files_path + "/csr_list_vtx";
                int f = open(swap_file.c_str(), O_CREAT| O_RDWR | O_TRUNC, 0644);
                assert(ftruncate(f, sizeof(VertexId) * num_edges) == 0);
                csr_list_vtx = (VertexId*) mmap(0, sizeof(VertexId) * num_edges, PROT_READ | PROT_WRITE, MAP_SHARED, f, 0);
                graph.csr_list_vtx_f_ = f;
            }
            degree = new VertexId [num_vertices];
            //madvise(csr_idx, sizeof(EdgeId) * (num_vertices + 1), MADV_HUGEPAGE);
            //madvise(csr_list_vtx, sizeof(VertexId) * num_edges, MADV_HUGEPAGE);

            if (has_edge_label) {
                if (use_out_of_core_feature) {
                    printf("ERROR: out-of-core feature is not supported when edge labes is presented.\n");
                    assert(false);
                }
                if (! std::is_same<EdgeLabel, Empty>::value) {
                    csr_list_label = new EdgeLabel[num_edges];
                    EdgeStruct<EdgeLabel> * edge_list = new EdgeStruct<EdgeLabel> [num_edges];
                    read_file(f, (uint8_t*) edge_list, sizeof(EdgeStruct<EdgeLabel>) * num_edges);
                    construct_csr_format(
                            edge_list,
                            num_edges,
                            csr_idx,
                            csr_list_vtx,
                            csr_list_label,
                            max_degree,
                            num_vertices,
                            degree
                            );
                    delete [] edge_list;
                } else {
                    EdgeStruct<VertexId> * edge_list = new EdgeStruct<VertexId> [num_edges];
                    read_file(f, (uint8_t*) edge_list, sizeof(EdgeStruct<VertexId>) * num_edges);
                    construct_csr_format_for_empty(
                            edge_list,
                            num_edges,
                            csr_idx,
                            csr_list_vtx,
                            csr_list_label,
                            max_degree,
                            num_vertices,
                            degree
                            );
                    delete [] edge_list;
                }
            } else {
                EdgeStruct<Empty> * edge_list;

                if (! use_out_of_core_feature && ! avoid_copy_edgelist) {
                    edge_list = new EdgeStruct<Empty> [num_edges];
                    read_file(f, (uint8_t*) edge_list, sizeof(EdgeStruct<Empty>) * num_edges);
                } else {
                    //std::string swap_file = swap_files_path + "/edge_list";
                    assert(close(f) == 0);

                    //f = open(swap_file.c_str(), O_CREAT| O_RDWR| O_TRUNC, 0644);
                    f = open(file_name.c_str(), O_RDONLY);
                    assert(f != -1);
                    //assert(ftruncate(f, sizeof(EdgeStruct<Empty>) * num_edges + (1 << 20)) == 0);
                    //printf("Successfully resize the file.\n");

                    long expected_file_size = (long) sizeof(EdgeStruct<Empty>) * num_edges + sizeof(GraphMetaDataStructOnDisk);
                    //if ()
                    //printf("Expected file size: %lld\n", expected_file_size);
                    //printf("Actual file size: %lld\n", file_size(file_name));
                    assert(sizeof(EdgeStruct<Empty>) * num_edges + sizeof(GraphMetaDataStructOnDisk) == file_size(file_name));
                    uint8_t * pt = (uint8_t*) mmap(
                            0, sizeof(EdgeStruct<Empty>) * num_edges + sizeof(GraphMetaDataStructOnDisk), PROT_READ, MAP_SHARED, f, 0
                            );
                    pt = pt + sizeof(GraphMetaDataStructOnDisk);
                    edge_list = (EdgeStruct<Empty>*) pt;
                    //edge_list = (EdgeStruct<Empty>*) mmap(
                    //    0, sizeof(EdgeStruct<Empty>) * num_edges + (1 << 20), PROT_READ | PROT_WRITE, MAP_SHARED, f, 0
                    //    );
                    if (edge_list == (void*) -1) {
                        printf("Error calling mmap: %s\n", strerror(errno));
                    }
                    //read_file(f, (uint8_t*) edge_list, sizeof(EdgeStruct<Empty>) * num_edges);
                }

                //for (EdgeId e_i = 0; e_i < num_edges; e_i ++) {
                //	Debug::get_instance()->log(e_i, ": ", edge_list[e_i].src, " ", edge_list[e_i].dst);
                //}
                construct_csr_format_for_empty(
                        edge_list,
                        num_edges,
                        csr_idx,
                        csr_list_vtx,
                        csr_list_label,
                        max_degree,
                        num_vertices,
                        degree
                        );
                if (! use_out_of_core_feature && ! avoid_copy_edgelist) {
                    delete [] edge_list;
                } else {
                    munmap(edge_list, sizeof(EdgeStruct<Empty>) * num_edges);
                }
            }

            graph.num_vertices_ = num_vertices;
            graph.num_edges_ = num_edges;
            graph.max_degree_ = max_degree;
            graph.csr_idx_ = csr_idx;
            graph.csr_list_vtx_ = csr_list_vtx;
            graph.csr_list_label_ = csr_list_label;
            graph.node_label_list_ = node_label_list;
            graph.degree_ = degree;
            graph.is_out_of_core_ = use_out_of_core_feature;
            //graph.num_labels_ = num_labels;

            assert(close(f) == 0);

            // dealing with labeled graphs
            if (! std::is_same<NodeLabel, Empty>::value) {
                // get the number of labes
                //memset(node_label_list, 0, sizeof(VertexId) * num_vertices); // TODO
                VertexId label_occurences[MAX_NUM_LABELS];
                memset(label_occurences, 0, sizeof(label_occurences));
                VertexId num_labels = 0;
                Debug::get_instance()->log("Calculating number of labels");
#pragma omp parallel for 
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    VertexId label = (VertexId) node_label_list[v_i];
                    __sync_fetch_and_add(&label_occurences[label], 1);
                }
                for (VertexId l_i = 0; l_i < MAX_NUM_LABELS; ++ l_i) {
                    if (label_occurences[l_i] > 0) {
                        ++ num_labels;
                    }
                }
                for (VertexId l_i = 0; l_i < num_labels; ++ l_i) {
                    assert(label_occurences[l_i] > 0);
                }

                // constructing labeled vertex CSR 
                Debug::get_instance()->log("Number of labels = ", num_labels);
                Debug::get_instance()->log("Constructing Labeled vertices CSR");
                VertexId * labeled_vtx_csr_idx = new VertexId[num_labels + 1];
                VertexId * tmp_labeled_vtx_csr_idx = new VertexId[num_labels + 1];
                VertexId * labeled_vtx_csr_list = new VertexId[num_vertices];
                memset(labeled_vtx_csr_idx, 0, sizeof(VertexId) * (num_labels + 1));
#pragma omp parallel for 
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    VertexId label = (VertexId) node_label_list[v_i];
                    //printf("label = %u\n", label);
                    __sync_fetch_and_add(&labeled_vtx_csr_idx[label + 1], 1);
                }
                //printf("A");
                for (VertexId l_i = 1; l_i <= num_labels; ++ l_i) {
                    labeled_vtx_csr_idx[l_i] += labeled_vtx_csr_idx[l_i - 1];
                }
                memcpy(tmp_labeled_vtx_csr_idx, labeled_vtx_csr_idx, sizeof(VertexId) * (num_labels + 1));
                //printf("B");
                //#pragma omp parallel for 
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    VertexId label = (VertexId) node_label_list[v_i];
                    VertexId pos = __sync_fetch_and_add(&tmp_labeled_vtx_csr_idx[label], 1);
                    labeled_vtx_csr_list[pos] = v_i;
                }
                //for (VertexId l_i = 0; l_i <= num_labels; ++ l_i) {
                //	printf("    l_i = %u, %u\n", l_i, labeled_vtx_csr_idx[l_i]);
                //}

                // constructing labeled neighbours sets
                Debug::get_instance()->log("Constructing Labeled neighbours vertices CSR");
                EdgeId ** labeled_nbr_csr_idx = new EdgeId* [num_vertices];
                {
                    EdgeId * tmp_mem_block = new EdgeId[num_vertices * (num_labels + 1)];
                    memset(tmp_mem_block, 0, sizeof(EdgeId) * num_vertices * (num_labels + 1));
#pragma omp parallel for 
                    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                        labeled_nbr_csr_idx[v_i] = tmp_mem_block + (v_i * (num_labels + 1));
                    }
                }
                VertexId * labeled_nbr_csr_list = new VertexId[num_edges];

                int num_threads = SharedMemorySys::get_instance()->get_num_threads();
#pragma omp parallel
                {
                    int thread_id = SharedMemorySys::get_instance()->get_current_thread_id();
                    VertexId delta = num_vertices / num_threads;
                    VertexId start_v_i = delta * thread_id;
                    VertexId end_v_i = delta * (thread_id + 1);
                    if (thread_id == num_threads - 1) {
                        end_v_i = num_vertices;
                    }
                    //printf("Thread ID = %d, start_v_i = %u, end_v_i = %u\n", thread_id, start_v_i, end_v_i);

                    EdgeId tmp_local_labeled_csr_idx[MAX_NUM_LABELS + 1];
                    for (VertexId v_i = start_v_i; v_i < end_v_i; ++ v_i) {
                        EdgeId * local_labeled_csr_idx = labeled_nbr_csr_idx[v_i];
                        VertexId * neighbours = csr_list_vtx + csr_idx[v_i];
                        EdgeId num_neighbours = csr_idx[v_i + 1] - csr_idx[v_i];
                        //printf("v_i = %u, number of neighbours = %u\n", v_i, num_neighbours);
                        for (EdgeId nbr_idx = 0; nbr_idx < num_neighbours; ++ nbr_idx) {
                            VertexId nbr_vtx = neighbours[nbr_idx];
                            VertexId label = (VertexId) node_label_list[nbr_vtx];
                            local_labeled_csr_idx[label + 1] ++;
                        }
                        //printf("A\n");
                        local_labeled_csr_idx[0] = csr_idx[v_i];
                        for (VertexId l_i = 1; l_i <= num_labels; ++ l_i) {
                            local_labeled_csr_idx[l_i] += local_labeled_csr_idx[l_i - 1];
                        }
                        //for (VertexId l_i = 0; l_i <= num_labels; ++ l_i) {
                        //	printf("    %lu\n", local_labeled_csr_idx[l_i]);
                        //}
                        //printf("B\n");
                        assert(local_labeled_csr_idx[num_labels] == csr_idx[v_i + 1]);
                        //printf("C\n");
                        memcpy(tmp_local_labeled_csr_idx, local_labeled_csr_idx, sizeof(EdgeId) * (num_labels + 1));
                        for (EdgeId nbr_idx = 0; nbr_idx < num_neighbours; ++ nbr_idx) {
                            VertexId nbr_vtx = neighbours[nbr_idx];
                            VertexId label = (VertexId) node_label_list[nbr_vtx];
                            //printf("label = %u, %lu\n", label, tmp_local_labeled_csr_idx[label]);
                            labeled_nbr_csr_list[tmp_local_labeled_csr_idx[label] ++] = nbr_vtx;
                        }
                        //printf("D\n");
                        for (VertexId l_i = 1; l_i <= num_labels; ++ l_i) {
                            assert(tmp_local_labeled_csr_idx[l_i - 1] == local_labeled_csr_idx[l_i]);
                        }
                    }
                }

                graph.num_labels_ = num_labels;
                graph.labeled_vtx_csr_idx_ = labeled_vtx_csr_idx;
                graph.labeled_vtx_csr_list_ = labeled_vtx_csr_list;
                graph.labeled_nbr_csr_idx_ = labeled_nbr_csr_idx;
                graph.labeled_nbr_csr_list_ = labeled_nbr_csr_list;
            }

            Debug::get_instance()->leave_function("CSRGraphLoader::load_graph");
        }

        // remember to call the function to destroy the CSRGraph object to avoid memory leaking
        void destroy_graph(CSRGraph<NodeLabel, EdgeLabel> &graph) {
            delete [] graph.csr_idx_;
            if (! graph.is_out_of_core_) {
                delete [] graph.csr_list_vtx_;
            } else {
                // TODO: release csr list mapping here
            }
            delete [] graph.degree_;
            if (graph.csr_list_label_ != nullptr) {
                delete [] graph.csr_list_label_;
            }
            if (graph.node_label_list_ != nullptr) {
                delete [] graph.node_label_list_;
            }
        }
};

class JointHashBasedNumaAwareDistGraphLoader {
    private:
        size_t graph_data_size_;
        int num_sockets_;

        void partition_graph_vertices(
                int num_partitions, 
                int partition_id,
                VertexId * &local_vertices,
                VertexId &num_local_vertices,
                VertexId * &num_local_vertices_per_partition,
                VertexId num_vertices,
                int socket_id
                ) {
            num_local_vertices_per_partition = (VertexId*) numa_alloc_onnode(
                    sizeof(VertexId) * num_partitions, socket_id
                    );
            memset(num_local_vertices_per_partition, 0, sizeof(VertexId) * num_partitions);
            graph_data_size_ += sizeof(VertexId) * num_partitions;

            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                int p_i = PART_ID_HASH(v_i, num_partitions);
                num_local_vertices_per_partition[p_i] ++;
            }
            num_local_vertices = num_local_vertices_per_partition[partition_id];

            printf("    Partition %d, Number of local vertices: %u\n",
                    partition_id, num_local_vertices);

            VertexId num_local_vertices_sum = 0;
            for (int p_i = 0; p_i < num_partitions; ++ p_i) {
                num_local_vertices_sum += num_local_vertices_per_partition[p_i];
            }
            assert(num_local_vertices_sum == num_vertices);

            // storing the local vertices
            local_vertices = (VertexId*) numa_alloc_onnode(
                    sizeof(VertexId) * num_local_vertices, socket_id
                    );
            graph_data_size_ = sizeof(VertexId) * num_local_vertices;
            num_local_vertices = 0;
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                int p_i = PART_ID_HASH(v_i, num_partitions);
                if (p_i == partition_id) {
                    local_vertices[num_local_vertices ++] = v_i;
                }
            }
            assert(num_local_vertices == num_local_vertices_per_partition[partition_id]);
        }

        void partition_graph_vertices(
                int num_nodes, int node_id, int num_sockets,
                VertexId ** local_vertices,
                VertexId * num_local_vertices,
                VertexId ** num_local_vertices_per_partition,
                VertexId num_vertices
                ) {
            printf("    Partitioning the graph vertices...\n");
            int num_partitions = num_nodes * num_sockets;
            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                int partition_id = node_id * num_sockets + socket_id;
                partition_graph_vertices(
                        num_partitions, partition_id,
                        local_vertices[socket_id],
                        num_local_vertices[socket_id],
                        num_local_vertices_per_partition[socket_id],
                        num_vertices,
                        socket_id
                        );
            }
            printf("Finished partitioning the graph.\n");
        }

        void calculate_degree(
                EdgeStruct<Empty> * edge_list, VertexId * degree,
                VertexId num_vertices, EdgeId num_edges
                ) {
            printf("Calculating degree...\n");

            memset(degree, 0, sizeof(VertexId) * num_vertices);

            auto process_edge_data = [&](uint8_t * data_block, size_t data_block_size) {
                EdgeStruct<Empty> * edge_block = (EdgeStruct<Empty> *) data_block;
                assert(data_block_size % sizeof(EdgeStruct<Empty>) == 0);
                EdgeId edge_block_size = data_block_size / sizeof(EdgeStruct<Empty>);
                for (EdgeId e_i = 0; e_i < edge_block_size; ++ e_i) {
                    VertexId src = edge_block[e_i].src;
                    degree[src] ++;
                }
            };

            BlockDataDistributer * graph_data_distributer = new BlockDataDistributer();
            graph_data_distributer->distribute_data(
                    (uint8_t*) edge_list,
                    num_edges * sizeof(EdgeStruct<Empty>),
                    process_edge_data
                    );
            delete graph_data_distributer;

            printf("Finished calculating degree.\n");
        }

        void construct_csr_format(
                VertexId num_vertices,
                VertexId * num_local_vertices,
                EdgeId num_edges,
                EdgeId * num_local_edges,
                CSRIndexTable ** csr_idx,
                VertexId ** csr_list_vtx,
                VertexId ** degree,
                VertexId ** local_vertices,
                EdgeStruct<Empty> * edge_list,
                int num_sockets, 
                int num_nodes,
                int node_id
                ) {
            printf("    Constructing the CSR format...\n");

            int num_partitions = num_sockets * num_nodes;

            printf("    Calculating the number of local edges...\n");
            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                num_local_edges[socket_id] = 0;
                for (VertexId v_i = 0; v_i < num_local_vertices[socket_id]; ++ v_i) {
                    num_local_edges[socket_id] += (EdgeId) degree[socket_id][local_vertices[socket_id][v_i]];
                }
                int partition_id = num_sockets * node_id + socket_id;
                printf("    Partition %d, Number of local edges: %lu\n",
                        partition_id, num_local_edges[socket_id]);
            }

            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                csr_idx[socket_id] = new CyclicCSRIndexTable(
                        num_vertices, num_local_vertices[socket_id], socket_id, num_partitions
                        );
                csr_list_vtx[socket_id] = (VertexId*) numa_alloc_onnode(
                        sizeof(VertexId) * num_local_edges[socket_id], socket_id
                        );
                memset(csr_list_vtx[socket_id], 0, sizeof(VertexId) * num_local_edges[socket_id]);
                graph_data_size_ += sizeof(EdgeId) * num_local_vertices[socket_id];
                graph_data_size_ += sizeof(VertexId) * num_local_edges[socket_id];
            }

            EdgeId num_added_edges[num_sockets];
            VertexId curr_v[num_sockets];
            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                num_added_edges[socket_id] = 0;
                curr_v[socket_id] = num_vertices;
            }

            auto process_edge_data = [&](uint8_t * data_block, size_t data_block_size) {
                EdgeStruct<Empty> * edge_block = (EdgeStruct<Empty> *) data_block;
                assert(data_block_size % sizeof(EdgeStruct<Empty>) == 0);
                EdgeId edge_block_size = data_block_size / sizeof(EdgeStruct<Empty>);
                for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                    int partition_id = num_sockets * node_id + socket_id;
                    for (EdgeId e_i = 0; e_i < edge_block_size; ++ e_i) {
                        VertexId src = edge_block[e_i].src;
                        assert(curr_v[socket_id] == num_vertices || curr_v[socket_id] <= src);
                        if (PART_ID_HASH(src, num_partitions) == partition_id) {
                            if (curr_v[socket_id] != src) {
                                curr_v[socket_id] = src;
                                csr_idx[socket_id]->set(src, num_added_edges[socket_id]);
                            }
                            VertexId dst = edge_block[e_i].dst;
                            csr_list_vtx[socket_id][num_added_edges[socket_id] ++] = dst;
                        }
                    }
                }
            };

            BlockDataDistributer * graph_data_distributer = new BlockDataDistributer();
            graph_data_distributer->distribute_data(
                    (uint8_t*) edge_list, 
                    num_edges * sizeof(EdgeStruct<Empty>),
                    process_edge_data
                    );
            delete graph_data_distributer;

            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                assert(num_added_edges[socket_id] == num_local_edges[socket_id]);
            }

            printf("    Finished constructing the CSR format.\n");
        }

    public:
        void load_graphs(
                const std::string &file_name,
                HashBasedNumaAwareDistributedCSRGraph * graphs, // [num sockets] 
                int num_sockets,
                bool shared_degree_array,
                bool load_vertex_labels
                ) {
            int node_id = DistributedSys::get_instance()->get_node_id();
            int num_nodes = DistributedSys::get_instance()->get_num_nodes();
            int num_partitions = num_nodes * num_sockets;

            printf("Loading graph...\n");

            graph_data_size_ = 0;

            // the graph representation
            VertexId num_vertices;
            VertexId num_local_vertices[num_sockets];
            EdgeId num_edges;
            EdgeId num_local_edges[num_sockets];
            VertexId max_degree;

            CSRIndexTable * csr_idx[num_sockets];
            VertexId * csr_list_vtx[num_sockets];
            VertexId * degree[num_sockets];
            VertexId * local_vertices[num_sockets];

            VertexId num_hub_vertices;
            VertexId * hub_vertices[num_sockets];

            VertexId * num_local_vertices_per_partition[num_sockets];

            VertexId hub_vertex_threshold = HUB_VERTEX_DEGREE_TH;

            // supporting vertex labels
            bool is_labeled_graph = load_vertex_labels;
            LabelId num_labels = 0;
            LabelId * vertex_labels[num_sockets];
            VertexId * labeled_vertices_csr_idx[num_sockets];
            VertexId * labeled_vertices_csr_list[num_sockets];

            memset(vertex_labels, 0, sizeof(vertex_labels));
            memset(labeled_vertices_csr_idx, 0, sizeof(labeled_vertices_csr_idx));
            memset(labeled_vertices_csr_list, 0, sizeof(labeled_vertices_csr_list));

            // open the graph dataset file
            assert(file_exits(file_name));
            int f = open(file_name.c_str(), O_RDONLY);
            assert(f != -1);

            long file_length = file_size(file_name.c_str());
            void * mmap_ptr = mmap(NULL, file_length, PROT_READ, MAP_PRIVATE, f, 0);

            // get graph meta data
            GraphMetaDataStructOnDisk graph_meta_data_on_disk;
            read_file(f, (uint8_t *) &graph_meta_data_on_disk, sizeof(GraphMetaDataStructOnDisk));

            num_vertices = graph_meta_data_on_disk.num_vertices;
            num_edges = graph_meta_data_on_disk.num_edges;

            printf("    Number of vertices: %u\n", num_vertices);
            printf("    Number of edges: %lu\n", num_edges);

            int has_node_label = graph_meta_data_on_disk.has_node_label;
            int has_edge_label = graph_meta_data_on_disk.has_edge_label;
            assert(has_edge_label == 0); // do not support edge label yet

            // getting the edge list && the vertex label list
            EdgeStruct<Empty> * edge_list;
            if (graph_meta_data_on_disk.has_node_label) {
                if (load_vertex_labels) {
                    printf("    Loading vertex labels...\n");
                    uint32_t * on_disk_vertex_labels = (uint32_t*) ((uint8_t*) mmap_ptr + sizeof(GraphMetaDataStructOnDisk));
                    for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                        vertex_labels[socket_id] = (LabelId*) 
                            numa_alloc_onnode(sizeof(LabelId) * num_vertices, socket_id);
                        graph_data_size_ += sizeof(LabelId) * num_vertices;
                        for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                            uint32_t l = on_disk_vertex_labels[v_i];
                            vertex_labels[socket_id][v_i] = l;
                            num_labels = std::max(num_labels, l + 1);
                        }
                    }
                    printf("    Number of labels: %u\n", num_labels);
                    printf("    Constructing the CSR format for labeled vertices...\n");
                    for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                        labeled_vertices_csr_idx[socket_id] = (VertexId*)
                            numa_alloc_onnode(sizeof(VertexId) * (num_labels + 1), socket_id);
                        labeled_vertices_csr_list[socket_id] = (VertexId*)
                            numa_alloc_onnode(sizeof(VertexId) * num_vertices, socket_id);
                        graph_data_size_ += sizeof(VertexId) * (num_labels + 1);
                        graph_data_size_ += sizeof(VertexId) * num_vertices;
                        for (LabelId l_i = 0; l_i <= num_labels; ++ l_i) {
                            labeled_vertices_csr_idx[socket_id][l_i] = 0;
                        }
                        for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                            uint32_t l = on_disk_vertex_labels[v_i];
                            labeled_vertices_csr_idx[socket_id][l + 1] ++;
                        }
                        for (LabelId l_i = 1; l_i <= num_labels; ++ l_i) {
                            if (labeled_vertices_csr_idx[socket_id][l_i] == 0) {
                                fprintf(stderr, "The vertex labels should be continuous and range from 0 to num_labels - 1.\n");
                                exit(-1);
                            }
                        }
                        for (LabelId l_i = 1; l_i <= num_labels; ++ l_i) {
                            labeled_vertices_csr_idx[socket_id][l_i] += labeled_vertices_csr_idx[socket_id][l_i - 1];
                        }
                        VertexId * tmp = new VertexId [num_labels + 1];
                        memcpy(tmp, labeled_vertices_csr_idx[socket_id], sizeof(VertexId) * (num_labels + 1));
                        for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                            uint32_t l = on_disk_vertex_labels[v_i];
                            labeled_vertices_csr_list[socket_id][tmp[l] ++] = v_i;
                        }
                        delete tmp;
                    }
                }
                edge_list = (EdgeStruct<Empty> *) ((uint8_t*) mmap_ptr + sizeof(GraphMetaDataStructOnDisk) + sizeof(uint32_t) * num_vertices);
            } else {
                assert(load_vertex_labels == false);
                edge_list = (EdgeStruct<Empty> *) ((uint8_t*) mmap_ptr + sizeof(GraphMetaDataStructOnDisk));
            }

            assert(close(f) == 0);

            partition_graph_vertices(
                    num_nodes, node_id, num_sockets, 
                    local_vertices, num_local_vertices, 
                    num_local_vertices_per_partition, 
                    num_vertices
                    );

            // calculating degree
            if (! shared_degree_array) {
                for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                    degree[socket_id] = (VertexId*) numa_alloc_onnode(
                            sizeof(VertexId) * num_vertices, socket_id
                            );
                    graph_data_size_ += sizeof(VertexId) * num_vertices;
                }
                calculate_degree(
                        edge_list, degree[0], num_vertices, num_edges
                        );
                for (int socket_id = 1; socket_id < num_sockets; ++ socket_id) {
                    memcpy(
                            degree[socket_id], degree[0], sizeof(VertexId) * num_vertices
                          );
                }
            } else {
                degree[0] = (VertexId*) numa_alloc_interleaved(sizeof(VertexId) * num_vertices);
                graph_data_size_ += sizeof(VertexId) * num_vertices;
                calculate_degree(
                        edge_list, degree[0], num_vertices, num_edges
                        );
                for (int socket_id = 1; socket_id < num_sockets; ++ socket_id) {
                    degree[socket_id] = degree[0];
                }
            }
            // calculating degree-related meta-data
            max_degree = 0;
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                max_degree = std::max(max_degree, degree[0][v_i]);
            }
            printf("    Max degree: %u\n", max_degree);

            num_hub_vertices = 0;
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                num_hub_vertices += (degree[0][v_i] > hub_vertex_threshold);
            }
            printf("    Number of hub vertices: %u\n", num_hub_vertices);

            for (int socket_id = 0; socket_id < num_sockets; socket_id ++) {
                hub_vertices[socket_id] = (VertexId*) numa_alloc_onnode(
                        sizeof(VertexId) * num_hub_vertices, socket_id
                        );
                graph_data_size_ += sizeof(VertexId) * num_hub_vertices;
                VertexId found_num_hub_vertices = 0;
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    if (degree[socket_id][v_i] > hub_vertex_threshold) {
                        hub_vertices[socket_id][found_num_hub_vertices ++] = v_i;
                    }
                }
                assert(num_hub_vertices == found_num_hub_vertices);
            }

            // sanity check
            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                EdgeId degree_sum = 0;
                for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                    degree_sum += degree[socket_id][v_i];
                }
                assert(degree_sum == num_edges);
            }

            // construct the CSR representation
            construct_csr_format(
                    num_vertices, 
                    num_local_vertices,
                    num_edges,
                    num_local_edges,
                    csr_idx,
                    csr_list_vtx,
                    degree,
                    local_vertices,
                    edge_list,
                    num_sockets,
                    num_nodes,
                    node_id
                    );

            assert(munmap(mmap_ptr, file_length) == 0);

            printf("    Shuffling the local vertices...\n");
            unsigned seed = 17;
            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                std::shuffle(
                        local_vertices[socket_id], local_vertices[socket_id] + num_local_vertices[socket_id],
                        std::default_random_engine(seed)
                        );
            }

            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                int partition_id = num_sockets * node_id + socket_id;

                graphs[socket_id].num_vertices_ = num_vertices;
                graphs[socket_id].num_local_vertices_ = num_local_vertices[socket_id];
                graphs[socket_id].num_edges_ = num_edges;
                graphs[socket_id].num_local_edges_ = num_local_edges[socket_id];
                graphs[socket_id].max_degree_ = max_degree;

                graphs[socket_id].csr_idx_ = csr_idx[socket_id];
                graphs[socket_id].csr_list_vtx_ = csr_list_vtx[socket_id];
                graphs[socket_id].degree_ = degree[socket_id];
                graphs[socket_id].local_vertices_ = local_vertices[socket_id];

                graphs[socket_id].num_partitions_ = num_partitions;
                graphs[socket_id].partition_id_ = partition_id;
                graphs[socket_id].socket_id_ = socket_id;
                graphs[socket_id].num_local_vertices_per_partition_ = num_local_vertices_per_partition[socket_id];

                graphs[socket_id].num_hub_vertices_ = num_hub_vertices;
                graphs[socket_id].hub_vertices_ = hub_vertices[socket_id];

                graphs[socket_id].graph_data_size_ = graph_data_size_ / num_sockets;

                graphs[socket_id].is_labeled_graph_ = is_labeled_graph;
                graphs[socket_id].num_labels_ = num_labels;
                graphs[socket_id].vertex_labels_ = vertex_labels[socket_id];
                graphs[socket_id].labeled_vertices_csr_idx_ = labeled_vertices_csr_idx[socket_id];
                graphs[socket_id].labeled_vertices_csr_list_ = labeled_vertices_csr_list[socket_id];
            }

            printf("*** The graph partitions takes %.3f (GB) space on node %d\n",
                    graph_data_size_ / 1024. / 1024. / 1024., node_id);
            printf("Finished graph loading.\n");
        }

        void destroy_graphs(HashBasedNumaAwareDistributedCSRGraph * graphs, int num_sockets) {
            for (int socket_id = 0; socket_id < num_sockets; ++ socket_id) {
                delete graphs[socket_id].csr_idx_;
                numa_free(graphs[socket_id].csr_list_vtx_, sizeof(VertexId) * graphs[socket_id].num_local_edges_);
                if (socket_id == 0 || graphs[socket_id].degree_ != graphs[0].degree_) {
                    numa_free(graphs[socket_id].degree_, sizeof(VertexId) * graphs[socket_id].num_vertices_);
                }
                numa_free(graphs[socket_id].local_vertices_, sizeof(VertexId) * graphs[socket_id].num_local_vertices_);
                numa_free(graphs[socket_id].num_local_vertices_per_partition_, sizeof(VertexId) * graphs[socket_id].num_partitions_);
                numa_free(graphs[socket_id].hub_vertices_, sizeof(VertexId) * graphs[socket_id].num_hub_vertices_);
            }
        }
};

#endif



