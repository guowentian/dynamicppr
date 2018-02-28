# overview #
This repository is the implementation of [1]. 

* cpu: cpu based implementation using the parallel framework CILK_PLUS
* gpu: gpu based implementation using CUDA
* encoder: transform a SNAP graph data file into the randomized and compressed binary file
* workload: choose source vertices with specified features (e.g. top-10,top-1000) for a graph
* scripts: various useful scripts to run the experiments


# prerequisite #
## GPU ##
* CUDA 7.0 or newer
* CUB 1.6.3 or newer, which should be installed in the same directory as this repository (can be installed somewhere elase but need to modify the makefile in 'gpu' folder). 
https://nvlabs.github.io/cub/

## CPU ##
* GCC 5.4.0 or newer (GCC 4.8 with CILK_PLUS should be fine, but haven't been tested)
* boost 1.55 or newer

# how to compile #
To compile the program, enter into 'gpu', 'cpu', 'workload' and 'encoder' separately and then execute 'make'. 

For 'gpu', you may need to mofity CFLAGS and INCLUDE in makefile, depending on the compute capability of your gpu and the directory of your CUB.

There are some flags that you can add in Makefile for compilation

* PROFILE: report useful information for defined profiling metrices
* PAPI_PROFILE, NVPROFILE, VALIDATE : deprecated, used during development

# how to run #
## 1. Prepare the data set ##
The input graph data file should be encoded as our binary format. 

The program 'workload' reads in the SNAP data file, randomize the edges, calculate the 0-based vertex ids and generate a more compressed binary data file 

(The SNAP format is a list of edges per line, where one line contains two vertex ids separated by tab or space).

As an example, download the youtube graph file from https://snap.stanford.edu/data/com-Youtube.html.
```
wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz 
``` 
Then, REMEMBER to delete any line of comments starting with '#', 
as the input file to encoder should be made up of only edges.
```
./encoder input_filename reverse
```
The argument 'reverse' means whether to reverse the direction of the edges, which is normally set to 0.
For our example, run:
```
./encoder $DATA_DIR/com-youtube.ungraph.txt 0
```
After executing this command, you could see an output file in the 'encoder' folder, i.e. com-youtube.ungraph.bin.


## 2. Prepare the source vertex ##
We run our experiment on the same set of source vertices. 

'workload' can generate the source vertex ids with top-10, top-1000, and top-1000000 degrees in the graph.
If you have chosen your own set of source vertices, you can just skip this step. 

For convenience, **you could directly use the pre-generated source vertices in the 'scripts/exp_vids' folder**.
```
./workload filename directed is_window is_choose_outdegree
```

* directed: 1 for directed graph and 0 for undirected graph
* is_window: whether evaluating the degree of a vertex in the window of a graph, normally set to 0
* is_choose_outdegree: evaluate the out-degree or in-degree of a vertex
As an example, run:
```
./workload $DATA_DIR/com-youtube.ungraph.bin 0 0 1
```

## 3. Run ##
The gpu and cpu implmentation has similar arguments, which we would explain as follows.

* -d: gDataFileName (the input graph file encoded as our binary form)
* -a: gAppType
* 0:rev push, 1:monte carlo
* -i: gIsDirected (1 for directed graph, 0 otherwise)
* -y: gIsDynamic (1 for execution on streaming graph, 0 is deprecated)
* -w: gWindowRatio (#edges in the sliding window, by default 0.1 of the all edges)
* -n: gWorkloadConfigType 0: SLIDE_WINDOW_RATIO, 1: SLIDE_BATCH_SIZE (two modes for the dynamic graph: in mode 0, should set -r and -b; in mode 1, should set -c and -l)
* -r: gStreamUpdateCountVersusWindowRatio (the batch size is set to be the ratio of the window size, by default 1% of the window) 
* -b: gStreamBatchCount (the number of batches)
* -c: gStreamUpdateCountPerBatch (the number of edges per batch)
* -l: gStreamUpdateCountTotal (the total number of edges)
* -s: gSourceVertexId
* -t: gThreadNum (by default 1)
* -o: gVariant (by default set to 0) 0: optimized, 1: fast frontier, 2: eager, 3: VANILLA
* -e: error tolerance (by default 1e-9)
```
EXAMPLE: ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 1 -n 0 -r 0.01 -b 100 -s 1
```
```
EXAMPLE: ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 1 -n 1 -c 100 -l 10000 -s 1
```
Normally, we use mode 0 with '-r 0.01 -b 100' and change different source vertices. 

For your convenience, we provide useful scripts in 'cpu' and 'gpu' folder. 

# reference #
[1. Parallel Personalized PageRank on Dynamic Graphs. Wentian Guo, Yuchen Li, Mo Sha, Kian-Lee Tan. VLDB 2017](http://www.vldb.org/pvldb/vol11/p93-guo.pdf) .
