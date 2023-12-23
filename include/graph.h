#ifndef GRAPH_H
#define GRAPH_H

#include <mpi.h>
#include <pthread.h>
#include <numa.h>

#include <iostream>
#include <set>
#include <map>
#include <cstring>

#include "vertex_set.h"
#include "types.h"
#include "distributed_sys.h"
#include "shared_memory_sys.h"

#define PART_ID_HASH(vid, num_partitions) ((int)((vid) % (num_partitions)))
//#define HUB_VERTEX_DEGREE_TH (1024 * 1024) 
#define HUB_VERTEX_DEGREE_TH (1024 * 1024 * 1024) 

template<typename NodeLabel, typename EdgeLabel> 
class CSRGraphLoader;

template<typename NodeLabel>
class DistributedCSRGraphLoader;

class NumaAwareDistributedCSRGraphLoader;
class HashBasedNumaAwareDistributedCSRGraphLoader;
class JointHashBasedNumaAwareDistGraphLoader;

struct Empty {
    operator VertexId() const {
        return 0;
    }
} __attribute__((packed));

template<typename EdgeLabel> 
struct EdgeStruct {
    VertexId src;
    VertexId dst;
    EdgeLabel label;
} __attribute__((packed));

template<>
struct EdgeStruct<Empty> {
    VertexId src;
    union {
        VertexId dst;
        Empty label;
    };
} __attribute__((packed));

template<typename NodeLabel, typename EdgeLabel> 
class CSRGraph {
    private:
        VertexId num_vertices_;
        EdgeId num_edges_;
        VertexId max_degree_;
        VertexId num_labels_;

        EdgeId * csr_idx_; // EdgeId[num_vertices + 1]
        VertexId * csr_list_vtx_; // VertexId[num_edges]
        EdgeLabel * csr_list_label_; // EdgeLabel[num_edges]

        NodeLabel * node_label_list_; // NodeLabel[num_vertices]
        VertexId * degree_;

        // for labeled graphs
        // CSR format of labeled vertices
        VertexId * labeled_vtx_csr_idx_; // VertexId[num_labels + 1]
        VertexId * labeled_vtx_csr_list_; // VertexId[num_vertices]
        EdgeId ** labeled_nbr_csr_idx_; // EdgeId[num_vertices][num_labels + 1]
        VertexId * labeled_nbr_csr_list_; // VertexId[num_edges]

        bool is_out_of_core_;
        int csr_list_vtx_f_;

        friend class CSRGraphLoader<NodeLabel, EdgeLabel>;
    public: 
        CSRGraph(VertexId num_vertices, EdgeId num_edges, EdgeId * csr_idx, VertexId * csr_list_vtx, EdgeLabel * csr_list_label, NodeLabel * node_label_list) {
            num_vertices_ = num_vertices;
            num_edges_ = num_edges;
            csr_idx_ = csr_idx;
            csr_list_vtx_ = csr_list_vtx;
            node_label_list_ = node_label_list;
            is_out_of_core_ = false;
        }
        CSRGraph() {
        }
        ~CSRGraph() {
        }

        bool get_is_out_of_core() {
            return is_out_of_core_;
        }

        void set_is_out_of_core(bool is_out_of_core) {
            is_out_of_core_ = is_out_of_core;
        }

        const VertexId get_num_vertices() {
            return num_vertices_;
        }
        const EdgeId get_num_edges() {
            return num_edges_;
        }
        const VertexId get_num_labels() {
            return num_labels_;
        }
        const VertexSet get_labeled_vertices_set(LabelId label) {
            VertexId num_vtx = labeled_vtx_csr_idx_[label + 1] - labeled_vtx_csr_idx_[label];
            return VertexSet(
                    &labeled_vtx_csr_list_[labeled_vtx_csr_idx_[label]],
                    num_vtx
                    );
        }
        const VertexSet get_labeled_neighbour_vertices_set(VertexId vtx, LabelId label) {
            VertexId num_vtx = labeled_nbr_csr_idx_[vtx][label + 1] - labeled_nbr_csr_idx_[vtx][label];
            return VertexSet(
                    &labeled_nbr_csr_list_[labeled_nbr_csr_idx_[vtx][label]],
                    num_vtx
                    );
        }
        const VertexSet get_neighbour_vertices_set(VertexId vtx) {
            VertexSet vertex_set(&csr_list_vtx_[csr_idx_[vtx]], csr_idx_[vtx + 1] - csr_idx_[vtx]);
            return vertex_set;
        }
        const NodeLabel get_node_label(const VertexId vtx) {
            return node_label_list_[vtx];
        }
        const NodeLabel* get_node_label_list() {
            return node_label_list_;
        }
        const VertexId get_max_degree() {
            return max_degree_;
        }
        const inline VertexId get_degree(const VertexId vtx) {
            return degree_[vtx];
        }
};

// not support node labels
// 1-D graph partitioning
class NumaAwareDistributedCSRGraph {
    private:
        VertexId num_vertices_;
        VertexId num_local_vertices_;
        EdgeId num_edges_;
        EdgeId num_local_edges_;
        VertexId max_degree_;

        EdgeId * csr_idx_; // EdgeId [num_local_vertices + 1]
        VertexId * csr_list_vtx_; // VertexId [num_local_edges]
        VertexId * degree_; // VertexId [num_vertices]

        int num_partitions_;
        int partition_id_;
        int socket_id_;
        VertexId * partitions_offset_; // VertexId [num_partitions + 1]

        friend class NumaAwareDistributedCSRGraphLoader;

    public:
        NumaAwareDistributedCSRGraph(
                VertexId num_vertices, VertexId num_local_vertices,
                EdgeId num_edges, EdgeId num_local_edges,
                VertexId max_degree, 
                EdgeId * csr_idx, VertexId * csr_list_vtx, VertexId * degree,
                int num_partitions, int partition_id, int socket_id, VertexId * partitions_offset
                ) {
            num_vertices_ = num_vertices;
            num_local_vertices_ = num_local_vertices;
            num_edges_ = num_edges;
            num_local_edges_ = num_local_edges;
            max_degree_ = max_degree;

            csr_idx_ = csr_idx;
            csr_list_vtx_ = csr_list_vtx;
            degree_ = degree;

            num_partitions_ = num_partitions;
            partition_id_ = partition_id;
            socket_id_ = socket_id;
            partitions_offset_ = partitions_offset;
        }
        NumaAwareDistributedCSRGraph() {
        }
        ~NumaAwareDistributedCSRGraph() {
        }

        const inline bool is_local_vertex(VertexId vtx) {
            return vtx >= partitions_offset_[partition_id_] &&
                vtx < partitions_offset_[partition_id_ + 1];
        }
        const inline int get_vertex_master_partition(VertexId vtx) {
            for (int i = 0; i < num_partitions_; ++ i) {
                if (partitions_offset_[i] <= vtx && 
                        vtx < partitions_offset_[i + 1]) {
                    return i;
                }
            }
            assert(false);
        }
        const inline VertexId get_num_vertices() {
            return num_vertices_;
        }
        const inline VertexId get_num_local_vertices() {
            return num_local_vertices_;
        }
        const inline EdgeId get_num_edges() {
            return num_edges_;
        }
        const inline EdgeId get_num_local_edges() {
            return num_local_edges_;
        }
        inline VertexSet get_neighbour_vertices_set(VertexId vtx) {
            assert(is_local_vertex(vtx));
            VertexId num_vertices = csr_idx_[vtx + 1] - csr_idx_[vtx];
            return VertexSet(
                    &csr_list_vtx_[csr_idx_[vtx]], num_vertices
                    );
        }
        inline VertexId * get_neighbours_ptx(VertexId vtx) {
            assert(is_local_vertex(vtx));
            return &csr_list_vtx_[csr_idx_[vtx]];
        }
        const inline VertexId get_max_degree() {
            return max_degree_;
        }
        const inline VertexId get_degree(VertexId vtx) {
            return degree_[vtx];
        }
        const inline VertexId get_local_vertices_begin() {
            return partitions_offset_[partition_id_];
        }
        const inline VertexId get_local_vertices_end() {
            return partitions_offset_[partition_id_ + 1];
        }
};

class CSRIndexTable {
    public:
        virtual void set(VertexId key, EdgeId value) = 0;
        virtual EdgeId get(VertexId key) = 0;
};

class NaiveCSRIndexTable: public CSRIndexTable {
    private:
        VertexId num_vertices_;
        int socket_id_;
        EdgeId * table_;

    public:
        NaiveCSRIndexTable(VertexId num_vertices, int socket_id) {
            num_vertices_ = num_vertices;
            socket_id_ = socket_id;
            table_ = (EdgeId*) numa_alloc_onnode(
                    sizeof(EdgeId) * num_vertices_, socket_id_
                    );
        }
        ~NaiveCSRIndexTable() {
            numa_free(table_, sizeof(EdgeId) * num_vertices_);
        }
        void set(VertexId key, EdgeId value) {
            table_[key] = value;
        }
        EdgeId get(VertexId key) {
            return table_[key];
        }
};

class CyclicCSRIndexTable: public CSRIndexTable {
    private:
        VertexId num_vertices_;
        VertexId num_local_vertices_;
        int socket_id_;
        int num_partitions_;
        EdgeId * table_;

    public:
        CyclicCSRIndexTable(
                VertexId num_vertices,
                VertexId num_local_vertices,
                int socket_id,
                int num_partitions
                ) {
            num_vertices_ = num_vertices;
            num_local_vertices_ = num_local_vertices;
            socket_id_ = socket_id;
            num_partitions_ = num_partitions;

            table_ = (EdgeId*) numa_alloc_onnode(
                    sizeof(EdgeId) * num_local_vertices_, socket_id_
                    );
        }
        ~CyclicCSRIndexTable() {
            numa_free(table_, sizeof(EdgeId) * num_local_vertices_);
        }
        void set(VertexId key, EdgeId value) {
            table_[key / num_partitions_] = value;
        }
        EdgeId get(VertexId key) {
            return table_[key / num_partitions_];
        }
};

class HashBasedNumaAwareDistributedCSRGraph {
    private:
        VertexId num_vertices_;
        VertexId num_local_vertices_;
        EdgeId num_edges_; // including reverse edges
        EdgeId num_local_edges_; // including reverse edges
        VertexId max_degree_;

        CSRIndexTable * csr_idx_;
        VertexId * csr_list_vtx_; // VertexId [num_local_edges]
        VertexId * degree_; // VertexId [num_vertices]
        VertexId * local_vertices_; // VertexId [num_local_vertices]

        VertexId num_hub_vertices_;
        VertexId * hub_vertices_;

        int num_partitions_;
        int partition_id_;
        int socket_id_;
        VertexId * num_local_vertices_per_partition_; // VertexId [num_partitions]

        size_t graph_data_size_;

        // supporting vertex-labeled input graphs
        bool is_labeled_graph_;
        LabelId num_labels_;
        LabelId * vertex_labels_; // LabelId [num_vertices]
        VertexId * labeled_vertices_csr_idx_; // VertexId [num_labels + 1]
        VertexId * labeled_vertices_csr_list_; // VertexId [num_vertices]

        friend class HashBasedNumaAwareDistributedCSRGraphLoader;
        friend class JointHashBasedNumaAwareDistGraphLoader;

    public:
        HashBasedNumaAwareDistributedCSRGraph() {
        }
        ~HashBasedNumaAwareDistributedCSRGraph() {
        }

        const inline int get_vertex_master_partition(VertexId vtx) {
            int p_i = PART_ID_HASH(vtx, num_partitions_);
            assert(p_i >= 0 && p_i < num_partitions_);
            return p_i;
        }
        const inline bool is_local_vertex(VertexId vtx) {
            return get_vertex_master_partition(vtx) == partition_id_;
        }
        const inline VertexId get_num_vertices() {
            return num_vertices_;
        }
        const inline VertexId get_num_local_vertices() {
            return num_local_vertices_;
        }
        const inline EdgeId get_num_edges() {
            return num_edges_;
        }
        const inline EdgeId get_num_local_edges() {
            return num_local_edges_;
        }
        inline VertexSet get_neighbour_vertices_set(VertexId vtx) {
            assert(is_local_vertex(vtx));
            VertexId num_vertices = degree_[vtx];
            return VertexSet(
                    &csr_list_vtx_[csr_idx_->get(vtx)], num_vertices
                    );
        }
        inline VertexId * get_neighbours_ptx(VertexId vtx) {
            assert(is_local_vertex(vtx));
            return &csr_list_vtx_[csr_idx_->get(vtx)];
        }
        inline VertexId * get_neighbours_ptx_relaxed(VertexId vtx) {
            return is_local_vertex(vtx) ? &csr_list_vtx_[csr_idx_->get(vtx)]: nullptr;
        }
        const inline VertexId get_max_degree() {
            return max_degree_;
        }
        const inline VertexId get_degree(VertexId vtx) {
            return degree_[vtx];
        }
        VertexId * get_local_vertices() {
            return local_vertices_;
        }
        VertexId * get_num_local_vertices_per_partition() {
            return num_local_vertices_per_partition_;
        }
        int get_num_partitions() {
            return num_partitions_;
        }
        int get_partition_id() {
            return partition_id_;
        }
        VertexId get_num_hub_vertices() {
            return num_hub_vertices_;
        }
        VertexId* get_hub_vertices() {
            return hub_vertices_;
        }
        const size_t get_graph_data_size() {
            return graph_data_size_;
        }
        // supporting vertex-labeled graphs
        inline VertexSet get_labeled_vertices_set(const LabelId label) const {
            VertexId num_vtx = labeled_vertices_csr_idx_[label + 1] - labeled_vertices_csr_idx_[label];
            return VertexSet(
                    &labeled_vertices_csr_list_[labeled_vertices_csr_idx_[label]],
                    num_vtx
                    );
        }
        inline LabelId get_vertex_label(const VertexId v) const {
            return vertex_labels_[v];
        }
        inline LabelId get_num_labels() const {
            return num_labels_;
        }
        inline bool is_labeled_graph() const {
            return is_labeled_graph_;
        }
};

#endif


