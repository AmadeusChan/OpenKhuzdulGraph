# The Khuzdul Graph Pattern Mining Engine

## Compilation 

The engine is dependent on the following libraries:
1. GCC
2. OpenMPI
3. OpenMP
4. LibNUMA
5. CMake 

Note: 
- Using a MPI implementations except for OpenMPI might leads to a performance issue. we strongly recommend using OpenMPI.
- This engine relies on InfiniBand networks. It might not work well on other types of network like Ethernet.
- The code base is configured be used to analyze small- and median-scale graphs (e.g., graphs <= 10B edges). If you plan to evaluate the performance on massive-scale graphs (with > 10B edges), please reach out to us and we can provide another code base that is configured slightly differently for massive graphs (those evaluated in our papers).
- Since we haven't extensively tested our engine on vairous software/hardware environments, if you find any performance/correctness-related compatility issue, please feel free to reach out to us.

Compile the engine:
```bash
mkdir build
cd build 
cmake ..
make -j 4
```

## Dataset Preparation

We use an internal graph format named "*.dgraph". Below is the instruction (using the LiveJournal graph as an example) to convert a regular graph dataset in the edge-list format.

- Step 1: Download the original datasets, delete all isolated vertices (those connecting to no other vertices), delete all duplicated edges, and output the dataset as binary edge list format. Please refer to "./datasets/live_journal/standardize.cc" for this process.
- Step 2: Use the toolkits/converter to generate the dgraph format.

Example (LiveJournal):
```bash
% build the overall project firstly as instructed above using CMake

% download the dataset
cd datasets/live_journal/
sh download.sh

% preprocess the dataset (Step 1)
g++ standardize.cc -std=c++11 -O3 -fopenmp
./a.out

% convert the preprocessed dataset to the 
% *.dgraph format
cd ../../build/toolkits/
./converter ../../datasets/live_journal/data/live_journal ../../datasets/live_journal/data/live_journal 4846609 0 0 0
```

## Configure the Machine-Specific Parameters

One need to configure some machine-specific macros firstly before running the experiments.

- ./include/engine.h: MAX_MEMORY_SIZE: the maximum memory size the engine can use on a single machine;
- ./include/engine.h: L1_DCACHE_SIZE: the L1 d-cache size of your CPU;
- ./include/engine.h: L2_CACHE_SIZE: the L2 size;
- ./include/engine.h: LLC_CACHE_SIZE: the L3 cache size;

## Running the Distributed GPM Application

We use mpirun to launch the tasks.

```bash
% using 4 machines
mpirun -n 4 -N 1 -host <a list of your machine names> <application path (e.g., ./build/distributed_applications/kautomine/triangle)> <dataset in the *.dgraph format>
```

## Citation

If you use Khuzdul for your research, please cite our paper:
```bibtex
@inproceedings{chen2023khuzdul,
  title={Khuzdul: Efficient and Scalable Distributed Graph Pattern Mining Engine},
  author={Chen, Jingji and Qian, Xuehai},
  booktitle={Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
  pages={413--426},
  year={2023}
}
```



