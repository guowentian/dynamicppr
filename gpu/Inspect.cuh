#ifndef __INSPECT_CUH__
#define __INSPECT_CUH__

#include "PPRCommon.cuh"
#include <cub/cub.cuh>

// identify the vertices that can be pushed, put them into vertex_ft 
template<typename IndexType, typename ValueType>
__global__ void InspectPureRev(IndexType *vertex_ft, IndexType *vertex_ft_cnt, 
    IndexType vertex_count,
    ValueType *residual, const size_t phase_id, const double tolerance){
    typedef cub::BlockScan<IndexType, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    volatile __shared__ IndexType output_cta_offset;

    IndexType cta_offset = blockDim.x * blockIdx.x;
    IndexType thread_count = 0;
    while (cta_offset < vertex_count){
        IndexType u = cta_offset + threadIdx.x;
        if (u < vertex_count){
            if (IsLegalRevPush(residual[u], phase_id, tolerance)){
                thread_count++;
            }
        }
        cta_offset += blockDim.x * gridDim.x;
    }
    __syncthreads();
    IndexType scatter, total;
    BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);

    __syncthreads();
    if (threadIdx.x == 0){
        output_cta_offset = atomicAdd(vertex_ft_cnt, total);
    }

    __syncthreads();
    cta_offset = blockDim.x * blockIdx.x;
    while (cta_offset < vertex_count){
        IndexType u = cta_offset + threadIdx.x;
        if (u < vertex_count){
            if (IsLegalRevPush(residual[u], phase_id, tolerance)){
                vertex_ft[output_cta_offset + scatter] = u;
                scatter++;
            }
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}

// extract residual from frontiers, and updage pagerank and residual corrspondingly
template<typename IndexType, typename ValueType>
__global__ void InspectExtra(IndexType *vertex_ft, ValueType *vertex_ft_r, IndexType *vertex_ft_cnt, 
    ValueType *residual, ValueType *pagerank, const size_t phase_id){
    size_t cta_offset = blockIdx.x * blockDim.x;
    IndexType total_ft_cnt = *vertex_ft_cnt;
    while (cta_offset < total_ft_cnt){
        if (cta_offset + threadIdx.x < total_ft_cnt){
            IndexType u = vertex_ft[cta_offset + threadIdx.x];
            vertex_ft_r[cta_offset + threadIdx.x] = residual[u];
            pagerank[u] += ALPHA * residual[u];
            residual[u] = 0.0;
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}



#endif
