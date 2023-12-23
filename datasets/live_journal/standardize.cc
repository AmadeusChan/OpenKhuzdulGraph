#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include <string.h>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <utility>
#include <set>
#include <string>

using namespace std;

typedef unsigned int VertexId;
typedef unsigned long long EdgeId;

const VertexId num_vertices = 4847571;
const EdgeId num_edges = 68993773;

VertexId * node_labels;

struct EdgeStruct {
	VertexId src;
	VertexId dst;
} __attribute__((packed));

EdgeStruct * edge_list;
EdgeStruct * raw_edge_list;

int main(int argc, char ** argv) {
	// processing edge 
	{
		// read edge list 
		ifstream fin("./data/soc-LiveJournal1.txt");
		string tmp;
		getline(fin, tmp); // skip 
		getline(fin, tmp); // skip 
		getline(fin, tmp); // skip 
		getline(fin, tmp); // skip 

		EdgeId num_read_edges = 0;
		raw_edge_list = new EdgeStruct[num_edges]; 
		VertexId max_raw_vertex_id = 0;
		VertexId src, dst;
		while (!(fin >> src >> dst).eof()) {
			if (num_read_edges % 1000 == 0) {
				cout << "reading graph: " << 1. * num_read_edges / num_edges << endl;
				cout << "\033[F"; 
			}
			assert(num_read_edges < num_edges);
			raw_edge_list[num_read_edges].src = src;
			raw_edge_list[num_read_edges].dst = dst;
			++ num_read_edges;
			max_raw_vertex_id = max(max_raw_vertex_id, src);
			max_raw_vertex_id = max(max_raw_vertex_id, dst);
		}
		cout << endl;
		assert(num_read_edges == num_edges);
		fin.close();

		// mapping the vertex id to a continous space 
		bool * is_used = new bool[max_raw_vertex_id + 1];
		VertexId * new_vertex_id = new VertexId[max_raw_vertex_id + 1]; 
		memset(is_used, 0, sizeof(bool) * (max_raw_vertex_id + 1));
#pragma omp parallel for 
		for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
			if (raw_edge_list[e_i].src != raw_edge_list[e_i].dst) {
				is_used[raw_edge_list[e_i].src] = true;
				is_used[raw_edge_list[e_i].dst] = true;
			}
		}
		VertexId curr_vertex_id = 0;
		for (VertexId v_i = 0; v_i <= max_raw_vertex_id; ++ v_i) {
			if (is_used[v_i]) {
				new_vertex_id[v_i] = curr_vertex_id ++;
			}
		}
		if (is_used[580]) {
			cout << "yes" << endl;
		} else cout << "no" << endl;
		//ssert(curr_vertex_id == num_vertices);
		cout << "Number of vertices (after deleting isolated vertices) = " << curr_vertex_id << endl;
		delete [] is_used;

		EdgeId num_edges_remained = 0;
		edge_list = new EdgeStruct[num_edges]; 
		set<pair<VertexId, VertexId>> edge_set;
		edge_set.clear();
		for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
			if (e_i % 1000 == 0) {
				cout << "deleting duplicated edges && self-loops: " << 1. * e_i / num_edges << endl;
				cout << "\033[F"; 
			}
			VertexId src = new_vertex_id[raw_edge_list[e_i].src];
			VertexId dst = new_vertex_id[raw_edge_list[e_i].dst];
			if (src > dst) {
				swap(src, dst);
			}
			pair<VertexId, VertexId> edge_pair(src, dst); 
			if (src != dst && edge_set.find(edge_pair) == edge_set.end()) {
				edge_list[num_edges_remained].src = src;
				edge_list[num_edges_remained].dst = dst;
				edge_set.insert(edge_pair);
				++ num_edges_remained;
			}
		}
		cout << endl;

		cout << "number of edges(after standardization): " << num_edges_remained << endl;

		int f = open("./data/live_journal.biedgelist", O_CREAT | O_WRONLY | O_TRUNC, 0644);
		long total_write_bytes = 0;
		long total_bytes_to_write = sizeof(EdgeStruct) * num_edges_remained;
		while (total_write_bytes < total_bytes_to_write) {
			long write_bytes = write(f, ((uint8_t*) edge_list) + total_write_bytes, total_bytes_to_write - total_write_bytes);
			assert(write_bytes >= 0);
			total_write_bytes += write_bytes;
		}
		assert(close(f) == 0);
        printf("Done exporting the binary edge list data\n");

		{
			ofstream fout("./data/live_journal.edgelist");
			fout << num_vertices << " " << num_edges_remained << endl;
			for (EdgeId e_i = 0; e_i < num_edges_remained; ++ e_i) {
				fout << edge_list[e_i].src << " " << edge_list[e_i].dst << endl;
			}
			fout.close();
            printf("Done exporting the edge list data\n");
		}

		{
			ofstream fout("./data/live_journal.rs_edgelist");
			for (EdgeId e_i = 0; e_i < num_edges_remained; ++ e_i) {
				fout << edge_list[e_i].src << " " << edge_list[e_i].dst << endl;
				fout << edge_list[e_i].dst << " " << edge_list[e_i].src << endl;
			}
			fout.close();
            printf("Done exporting the rs_edgelist data\n");
		}

		{
			EdgeId * csr_idx = new EdgeId[num_vertices + 1];
			VertexId * csr_list_vtx = new VertexId[num_edges_remained * 2];
			memset(csr_idx, 0, sizeof(EdgeId) * (num_vertices + 1));
			for (EdgeId e_i = 0; e_i < num_edges_remained; ++ e_i) {
				++ csr_idx[edge_list[e_i].src + 1];
				++ csr_idx[edge_list[e_i].dst + 1];
			}
			for (VertexId v_i = 1; v_i <= num_vertices; ++ v_i) {
				csr_idx[v_i] += csr_idx[v_i - 1];
			}
			EdgeId * curr_pos = new EdgeId[num_vertices + 1];
			for (VertexId v_i = 0; v_i <= num_vertices; ++ v_i) {
				curr_pos[v_i] = csr_idx[v_i];
			}
			for (EdgeId e_i = 0; e_i < num_edges_remained; ++ e_i) {
				VertexId src = edge_list[e_i].src;
				VertexId dst = edge_list[e_i].dst;
				csr_list_vtx[curr_pos[src] ++] = dst;
				csr_list_vtx[curr_pos[dst] ++] = src;
			}
			for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
				assert(curr_pos[v_i] == csr_idx[v_i + 1]);
			}
			ofstream fout("./data/live_journal.rs_adjlist");
			for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
				//cout << v_i << endl;
				fout << v_i << " " << 0;
				for (EdgeId e_i = csr_idx[v_i]; e_i < csr_idx[v_i + 1]; ++ e_i) {
					fout << " " << csr_list_vtx[e_i];
				}
				fout << endl;
			}
			fout.close();
            printf("Done exporting the rs_adjlist data\n");
			delete [] csr_list_vtx;
			delete [] csr_idx;
			delete [] curr_pos;
		}

		delete [] edge_list;
		delete [] raw_edge_list;
		delete [] new_vertex_id;
	}
	return 0;
}
