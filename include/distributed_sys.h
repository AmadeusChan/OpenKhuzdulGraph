#ifndef DISTRIBUTED_SYS_H
#define DISTRIBUTED_SYS_H

#include <mpi.h>

#include <iostream>
#include <type_traits>

#define MAX_NUM_NODES 32

enum MessageType {
    RemoteAccessRequest,
    RemoteAccessRespond
};

enum CommunicationType {
    GraphDataRequestMetaTag = 0,
    GraphDataRequestVerticesTag = 1,
    GraphDataRespondTag = 2,
    BlockDataDistributerTag = 3,
    UserDataAggregation = 4
};

class DistributedSys {
    private:
	static DistributedSys * instance_;

	int node_id_;
	int num_nodes_;

	DistributedSys();

    public:
	static void init_distributed_sys();
	static void finalize_distributed_sys();
	static DistributedSys * get_instance();

	inline int get_node_id() {
	    return node_id_;
	}
	inline int get_num_nodes() {
	    return num_nodes_;
	}
    inline bool is_master_node() {
        return node_id_ == 0;
    }
	template<typename T>
	    static MPI_Datatype get_mpi_data_type() {
                if (std::is_same<T, char>::value) {
                    return MPI_CHAR;
                } else if (std::is_same<T, unsigned char>::value) {
                    return MPI_UNSIGNED_CHAR;
                } else if (std::is_same<T, int>::value) {
                    return MPI_INT;
                } else if (std::is_same<T, unsigned>::value) {
                    return MPI_UNSIGNED;
                } else if (std::is_same<T, long>::value) {
                    return MPI_LONG;
                } else if (std::is_same<T, unsigned long>::value) {
                    return MPI_UNSIGNED_LONG;
                } else if (std::is_same<T, float>::value) {
                    return MPI_FLOAT;
                } else if (std::is_same<T, double>::value) {
                    return MPI_DOUBLE;
                } else {
                    printf("type not supported\n");
                    exit(-1);
                }
	    }
};

#endif
