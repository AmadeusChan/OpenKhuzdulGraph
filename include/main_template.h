#define INSERT_MAIN_FUNCTION(APP_NAME)\
int main(int argc, char ** argv) {\
    if (argc != 2) {\
        Debug::get_instance()->print("usage: " #APP_NAME " [graph dataset (dgraph format)]");\
        exit(-1);\
    }\
    SharedMemorySys::init_shared_memory_sys();\
    CSRGraph<Empty, Empty> graph;\
    std::string graph_file_name = argv[1];\
    CSRGraphLoader<Empty, Empty> graph_loader;\
    graph_loader.load_graph(graph_file_name, graph);\
    Debug::get_instance()->print("warming up...");\
    EdgeId count;\
    count = mining_graph(graph); \
    int runs = 1;\
    for (int run = 0; run < runs; ++ run) {\
	Debug::get_instance()->print("run = ", run);\
        Timer::timer_start("mining");\
        count = mining_graph(graph);\
        Timer::timer_stop("mining");\
	Debug::get_instance()->print("	cnt = ", count);\
    }\
    Debug::get_instance()->print(#APP_NAME" result: ", count);\
    graph_loader.destroy_graph(graph);\
    Timer::report_timers();\
    return 0;\
}

#define INSERT_MAIN_FUNCTION_MULTI_PATTERNS(APP_NAME)\
int main(int argc, char ** argv) {\
    if (argc != 2) {\
        Debug::get_instance()->print("usage: " #APP_NAME " [graph dataset (dgraph format)]");\
        exit(-1);\
    }\
    SharedMemorySys::init_shared_memory_sys();\
    CSRGraph<Empty, Empty> graph;\
    std::string graph_file_name = argv[1];\
    CSRGraphLoader<Empty, Empty> graph_loader;\
    graph_loader.load_graph(graph_file_name, graph);\
    int runs = 1;\
    for (int run = 0; run < runs; ++ run) {\
	Debug::get_instance()->log("run = ", run);\
        double runtime = - get_time();\
        Timer::timer_start("mining");\
        mining_graph(graph);\
        Timer::timer_stop("mining");\
        runtime += get_time();\
        printf("Runtime: %.3f (ms)\n", runtime * 1000.);\
    }\
    graph_loader.destroy_graph(graph);\
    Timer::report_timers();\
    return 0;\
}

#define INSERT_MINING_FUNCTION_INITIALIZATION(PATTERN_SIZE, CHUNK_SIZE)\
EdgeId cnt = 0;\
VertexId num_vertices = graph.get_num_vertices();\
VertexId max_degree = graph.get_max_degree();\
int num_threads = SharedMemorySys::get_instance()->get_num_threads();\
MemoryPool ** memory_pools = new MemoryPool* [num_threads];\
_Pragma ("omp parallel for")\
for (int i = 0; i < num_threads; ++ i) {\
	assert(i == SharedMemorySys::get_instance()->get_current_thread_id());\
	memory_pools[i] = new MemoryPool(max_degree, PATTERN_SIZE);\
}\
const VertexId kChunkSize = CHUNK_SIZE;\
VertexId * thread_begin = new VertexId[num_threads];\
VertexId * thread_end = new VertexId[num_threads];\
VertexId * thread_curr = new VertexId[num_threads];\
for (int t_i = 0; t_i < num_threads; ++ t_i) {\
	thread_begin[t_i] = num_vertices / num_threads * t_i;\
	thread_curr[t_i] = thread_begin[t_i];\
	thread_end[t_i] = num_vertices / num_threads * (t_i + 1);\
	if (t_i == num_threads - 1) {\
		thread_end[t_i] = num_vertices;\
	}\
}

#define INSERT_MINING_FUNCTION_FINALIZATION()\
for (int i = 0; i < num_threads; ++ i) {\
	delete memory_pools[i];\
}\
delete [] memory_pools;\
delete [] thread_begin;\
delete [] thread_end;\
delete [] thread_curr;\
return cnt;

#define INSERT_MINING_FUNCTION_INITIALIZATION_MULTI_PATTERNS(NUM_PATTERNS, MAX_PATTERN_SIZE, CHUNK_SIZE)\
VertexId num_vertices = graph.get_num_vertices();\
VertexId max_degree = graph.get_max_degree();\
EdgeId num_intersection = 0;\
EdgeId num_substraction = 0;\
int num_threads = SharedMemorySys::get_instance()->get_num_threads();\
MemoryPool ** memory_pools = new MemoryPool * [num_threads];\
_Pragma ("omp parallel for")\
	for (int i = 0; i < num_threads; ++ i) {\
		assert(i == SharedMemorySys::get_instance()->get_current_thread_id());\
		memory_pools[i] = new MemoryPool(max_degree, MAX_PATTERN_SIZE * NUM_PATTERNS);\
	}\
const VertexId kChunkSize = CHUNK_SIZE;\
VertexId * thread_begin = new VertexId [num_threads];\
VertexId * thread_end = new VertexId [num_threads];\
VertexId * thread_curr = new VertexId [num_threads];\
for (int t_i = 0; t_i < num_threads; ++ t_i) {\
	thread_begin[t_i] = num_vertices / num_threads * t_i;\
	thread_curr[t_i] = thread_begin[t_i];\
	thread_end[t_i] = num_vertices / num_threads * (t_i + 1);\
	if (t_i == num_threads - 1) {\
		thread_end[t_i] = num_vertices;\
	}\
}

#define INSERT_MINING_FUNCTION_FINALIZATION_MULTI_PATTERNS()\
for (int i = 0; i < num_threads; ++ i) {\
	delete memory_pools[i];\
}\
delete [] memory_pools;\
delete [] thread_begin;\
delete [] thread_end;\
delete [] thread_curr;\
return 0;


