#ifndef __SLIDING_GRAPH_BUILDER_H__
#define __SLIDING_GRAPH_BUILDER_H__

#include "GPUUtil.cuh"
#include "Meta.h"
#include "EdgeBatch.h"
#include "GPUTimer.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

template<typename IndexType>
__global__ void EdgePairScatter(IndexType *input, IndexType *output, IndexType length, IndexType scale, IndexType offset){
    // normally, scale=2 and input is of the form (edge1, edge2)
    // scatter from edge1, edge2 (input) to edgepair(output)
    // length is length of input
    IndexType cta_offset = blockIdx.x * blockDim.x;
    while (cta_offset < length){
        IndexType idx = cta_offset + threadIdx.x;
        if (idx < length){
            IndexType output_idx = idx * scale + offset;
            output[output_idx] = input[idx];
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}
template<typename IndexType>
__global__ void EdgePairGather(IndexType *input, IndexType *output, IndexType length, IndexType scale, IndexType offset){
    // gather from edgepair(input) to edge1, edge2(output)
    // length is length of output
    IndexType cta_offset = blockIdx.x * blockDim.x;
    while (cta_offset < length){
        IndexType idx = cta_offset + threadIdx.x;
        if (idx < length){
            IndexType input_idx = idx * scale + offset;
            output[idx] = input[input_idx];
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}

struct EdgePair{
    IndexType x;
    IndexType y;
    __host__ __device__ bool operator< (const EdgePair& r) const{
        return (x < r.x) || (x == r.x && y < r.y);
    }
};

__global__ void CollectOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length,
    IndexType *deg, IndexType vertex_count){
    IndexType cta_offset = blockIdx.x * blockDim.x;
    while (cta_offset < edge_batch_length){
        if (cta_offset + threadIdx.x < edge_batch_length){
            IndexType u = edge_batch1[cta_offset + threadIdx.x];
            assert(u < vertex_count);
            atomicAdd(deg + u, 1);
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}

class SlidingGraphBuilder{
public:
	SlidingGraphBuilder(IndexType vcount, IndexType wsize, bool directed) : vertex_count(vcount), sliding_window_size(wsize), is_directed(directed){
	 	cusparse_handle = 0;
        cusparse_descr = 0;
        CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));
        CUSPARSE_ERROR(cusparseCreateMatDescr(&cusparse_descr));
        cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);
		
        total_edge_length = is_directed ? sliding_window_size : sliding_window_size * 2;
		CUDA_ERROR(cudaMalloc(&total_edge1, sizeof(IndexType) * total_edge_length));
		CUDA_ERROR(cudaMalloc(&total_edge2, sizeof(IndexType) * total_edge_length));
        CUDA_ERROR(cudaMalloc(&edge_pairs, sizeof(IndexType) * total_edge_length * 2));
    }
	~SlidingGraphBuilder(){
        CUDA_ERROR(cudaFree(total_edge1));
        CUDA_ERROR(cudaFree(total_edge2));
        CUDA_ERROR(cudaFree(edge_pairs));
    }
    
    void IncBuildOutGraph(EdgeBatch *new_stream, IndexType *row_ptr, IndexType *col_ind){
#if defined(PROFILE)
        GPUTimer timer1, timer2;
        timer1.StartTimer();
#endif
        IncCopyStreamFromCPU(total_edge1, total_edge2, sliding_window_size, new_stream);
#if defined(PROFILE)
        timer1.EndTimer();
        timer2.StartTimer();
#endif
        BuildCSRGraph(total_edge1, total_edge2, total_edge_length, row_ptr, col_ind);
#if defined(PROFILE)
        timer2.EndTimer();
        std::cout << "gpu memcpy time=" << timer1.GetElapsedMilliSeconds() << "ms, rebuild time=" << timer2.GetElapsedMilliSeconds() << std::endl;
#endif
    }

    void BuildOutGraph(EdgeBatch *window_stream, IndexType *row_ptr, IndexType *col_ind){
#if defined(PROFILE)
        GPUTimer timer1, timer2;
        timer1.StartTimer();
#endif
        CopyStreamFromCPU(total_edge1, total_edge2, window_stream);
#if defined(PROFILE)
        timer1.EndTimer();
        timer2.StartTimer();
#endif
        BuildCSRGraph(total_edge1, total_edge2, total_edge_length, row_ptr, col_ind);
#if defined(PROFILE)
        timer2.EndTimer();
        std::cout << "gpu memcpy time=" << timer1.GetElapsedMilliSeconds() << "ms, rebuild time=" << timer2.GetElapsedMilliSeconds() << std::endl;
#endif
    }
       
    void IncBuildInGraph(EdgeBatch *new_stream, IndexType *in_row_ptr, IndexType *in_col_ind, IndexType *row_ptr){
#if defined(PROFILE)
        GPUTimer timer1, timer2;
        timer1.StartTimer();
#endif
        IncCopyStreamFromCPU(total_edge1, total_edge2, sliding_window_size, new_stream);
#if defined(PROFILE)
        timer1.EndTimer();
        timer2.StartTimer();
#endif
        BuildCSRGraph(total_edge2, total_edge1, total_edge_length, in_row_ptr, in_col_ind);
        BuildRowPtr(total_edge1, total_edge2, total_edge_length, row_ptr);
#if defined(PROFILE)
        timer2.EndTimer();
        std::cout << "gpu memcpy time=" << timer1.GetElapsedMilliSeconds() << "ms, rebuild time=" << timer2.GetElapsedMilliSeconds() << std::endl;
#endif
    }

    void BuildInGraph(EdgeBatch *window_stream, IndexType *in_row_ptr, IndexType *in_col_ind, IndexType *row_ptr){
#if defined(PROFILE)
        GPUTimer timer1, timer2;
        timer1.StartTimer();
#endif
        CopyStreamFromCPU(total_edge1, total_edge2, window_stream);
#if defined(PROFILE)
        timer1.EndTimer();
        timer2.StartTimer();
#endif
        BuildCSRGraph(total_edge2, total_edge1, total_edge_length, in_row_ptr, in_col_ind);
        BuildRowPtr(total_edge1, total_edge2, total_edge_length, row_ptr);
#if defined(PROFILE)
        timer2.EndTimer();
        std::cout << "gpu memcpy time=" << timer1.GetElapsedMilliSeconds() << "ms, rebuild time=" << timer2.GetElapsedMilliSeconds() << std::endl;
#endif
    }

    void CopyStreamFromCPU(IndexType *d_edge1, IndexType *d_edge2, EdgeBatch *edge_stream){
        //edge_stream does not embrase the direction
        assert(edge_stream->length == sliding_window_size);
		CUDA_ERROR(cudaMemcpy(d_edge1, edge_stream->edge1, sizeof(IndexType) * edge_stream->length, cudaMemcpyHostToDevice));
		CUDA_ERROR(cudaMemcpy(d_edge2, edge_stream->edge2, sizeof(IndexType) * edge_stream->length, cudaMemcpyHostToDevice));
		if (!is_directed){
			CUDA_ERROR(cudaMemcpy(d_edge1 + edge_stream->length, edge_stream->edge2, sizeof(IndexType) * edge_stream->length, cudaMemcpyHostToDevice));
			CUDA_ERROR(cudaMemcpy(d_edge2 + edge_stream->length, edge_stream->edge1, sizeof(IndexType) * edge_stream->length, cudaMemcpyHostToDevice));
		}
    }
    void IncCopyStreamFromCPU(IndexType *d_edge1, IndexType *d_edge2, IndexType window_size, EdgeBatch *new_stream){
        // generate [1..N...2N], [1...N] are in reverse direction of [N+1...2N] if undirected graph
        // new_stream is newly inserted edges without direction
        // remove stale edges
        IndexType *tmp = edge_pairs; // edge_pairs as temporary memory
        size_t cur_size = window_size - new_stream->length;
        CUDA_ERROR(cudaMemcpy(tmp, d_edge1 + new_stream->length, sizeof(IndexType) * cur_size, cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaMemcpy(d_edge1, tmp, sizeof(IndexType) * cur_size, cudaMemcpyHostToHost));
        CUDA_ERROR(cudaMemcpy(tmp, d_edge2 + new_stream->length, sizeof(IndexType) * cur_size, cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaMemcpy(d_edge2, tmp, sizeof(IndexType) * cur_size, cudaMemcpyHostToHost));
        // add new edges
        CUDA_ERROR(cudaMemcpy(d_edge1 + cur_size, new_stream->edge1, sizeof(IndexType) * new_stream->length, cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_edge2 + cur_size, new_stream->edge2, sizeof(IndexType) * new_stream->length, cudaMemcpyHostToDevice));
        
        if (!is_directed){
            CUDA_ERROR(cudaMemcpy(d_edge1 + window_size, d_edge2, sizeof(IndexType) * window_size, cudaMemcpyDeviceToDevice));
            CUDA_ERROR(cudaMemcpy(d_edge2 + window_size, d_edge1, sizeof(IndexType) * window_size, cudaMemcpyDeviceToDevice));
        }
    }
    void InitWindowStream(EdgeBatch *window_stream){
        // copy from cpu memory
        // <edge_stream1, edge_stream2> does not embrase edge direction
        assert(window_stream->length == sliding_window_size);
        CUDA_ERROR(cudaMemcpy(total_edge1, window_stream->edge1, sizeof(IndexType) * window_stream->length, cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(total_edge2, window_stream->edge2, sizeof(IndexType) * window_stream->length, cudaMemcpyHostToDevice));
        if (!is_directed){
            CUDA_ERROR(cudaMemcpy(total_edge2 + window_stream->length, window_stream->edge1, sizeof(IndexType) * window_stream->length, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(total_edge1 + window_stream->length, window_stream->edge2, sizeof(IndexType) * window_stream->length, cudaMemcpyHostToDevice));
        }
    }
    void BuildRowPtr(IndexType *edge1, IndexType *edge2, IndexType edge_length, IndexType *row_ptr){
        CUDA_ERROR(cudaMemset(row_ptr, 0, sizeof(IndexType) * (vertex_count+1)));
        size_t blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, edge_length);
        CollectOutDegree<<<blocks_num, THREADS_PER_BLOCK>>>(edge1, edge2, edge_length,
            row_ptr, vertex_count);

        thrust::device_ptr<IndexType> deg_ptr(row_ptr);
        thrust::exclusive_scan(deg_ptr, deg_ptr + vertex_count + 1, deg_ptr);
    }

    void BuildCSRGraph(IndexType *edge1, IndexType *edge2, IndexType edge_length, IndexType *row_ptr, IndexType *col_ind){
        // edge1, edge2 cannot be modified !! because we want to keep the order of edges to enable sliding window
        // scatter
        size_t blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, edge_length);
        EdgePairScatter<IndexType><<<blocks_num, THREADS_PER_BLOCK>>>(edge1, edge_pairs, edge_length, 2, 0);
        EdgePairScatter<IndexType><<<blocks_num, THREADS_PER_BLOCK>>>(edge2, edge_pairs, edge_length, 2, 1);

        // sort (x,y)
        EdgePair *pairs = (EdgePair*)edge_pairs;
        thrust::device_ptr<EdgePair> pairs_ptr(pairs);
        thrust::sort(pairs_ptr, pairs_ptr + edge_length);
       
        // row_ptr
        EdgePairGather<IndexType><<<blocks_num, THREADS_PER_BLOCK>>>(edge_pairs, col_ind, edge_length, 2, 0);
        CUSPARSE_ERROR(cusparseXcoo2csr(cusparse_handle, col_ind, edge_length, vertex_count, row_ptr, CUSPARSE_INDEX_BASE_ZERO));
        
        // col_ind
        EdgePairGather<IndexType><<<blocks_num, THREADS_PER_BLOCK>>>(edge_pairs, col_ind, edge_length, 2, 1);
    }
    
public:
	// cusparse context
	cusparseStatus_t cusparse_status;
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t cusparse_descr;
    // gpu memory
    IndexType total_edge_length;
    // <total_edge1, total_edge2>, embrase edge direction, the order is the same as edge stream arriving order
    // so as to implement incremental window slide in gpu memory
    IndexType *total_edge1;
    IndexType *total_edge2;

    // assist building csr graph
    IndexType *edge_pairs;

	IndexType vertex_count;
	IndexType sliding_window_size;
	bool is_directed;

};

#endif
