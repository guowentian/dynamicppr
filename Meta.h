#ifndef __META_H__
#define __META_H__

#include <iostream>
#include <string>

enum AlgoType{
	REVERSE_PUSH = 0,
    kAlgoTypeSize
};
enum VariantType{
    OPTIMIZED = 0,
    FAST_FRONTIER = 1,
    EAGER = 2,
    VANILLA= 3,
    kVariantTypeSize
};

enum WorkloadConfigType{
    // the slide size is a ratio of the window size, e.g. 1%, 0.1%, 0.01%
    SLIDE_WINDOW_RATIO,
    // the slide size is given as a number of edges, e.g. 10^5
    SLIDE_BATCH_SIZE
};
typedef double ValueType;
typedef int IndexType;

const static IndexType kMinVertexId = 0;
const static IndexType kMaxVertexId = 1000000000;

const static ValueType ALPHA = 0.15; // which is REST_PROB

const static size_t kMasterThreadId = 0;

   
const static IndexType DEFAULT_SOURCE_VERTEX_ID = 1;
//GPU
const static size_t THREADS_PER_BLOCK = 256;
const static size_t THREADS_PER_WARP = 32;
const static size_t MAX_BLOCKS_NUM = 96 * 8;
const static size_t MAX_THREADS_NUM = MAX_BLOCKS_NUM * THREADS_PER_BLOCK;
//CPU
const static size_t VERTEX_DEGREE_THRESHOLD = 512;

// ============== parameter ==============
// general
extern std::string gDataFileName;
extern int gAppType;
extern int gIsDirected;
extern int gIsDynamic;
// workload
extern double gWindowRatio;
extern int gWorkloadConfigType; 
// in either workload, gStreamBatchCount, gStreamUpdateCountPerBatch and gStreamUpdateCountTotal will be used in execution
// for SLIDE_WINDOW_RATIO
extern double gStreamUpdateCountVersusWindowRatio;
extern size_t gStreamBatchCount;
// for SLIDE_BATCH_SIZE
extern size_t gStreamUpdateCountPerBatch;
extern size_t gStreamUpdateCountTotal;

extern int gSourceVertexId;
// execution
extern int gThreadNum;
extern int gVariant;
// epsilon
extern ValueType gTolerance;

#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) MAX_BLOCKS_NUM < ((CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1) ? MAX_BLOCKS_NUM : ((CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1)
//#define CALC_THREAD_BIN_SIZE(vertex_count, max_threads_num) ((vertex_count - 1) / max_threads_num + 1)  


#endif
