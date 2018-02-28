#ifndef __PPR_COMMON_H__
#define __PPR_COMMON_H__

#include "Meta.h"

inline __host__ __device__ bool IsLegalRevPush(ValueType r, const size_t phase_id, const double tolerance){
    if ((phase_id == 0 && r > tolerance) || (phase_id == 1 && r < -tolerance)){
        return true;
    }
    return false;
}
template<typename IndexType, typename ValueType>
__global__ void Init(ValueType *pagerank, ValueType *residual, int *status, IndexType vertex_count, IndexType source_vertex_id){
	size_t active_threads = gridDim.x * blockDim.x;
	size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (IndexType s = thread_id; s < vertex_count; s += active_threads){
		residual[s] = (source_vertex_id == s) ? 1.0 : 0.0;
		pagerank[s] = 0.0;
        status[s] = -1;
	}
}
template<typename IndexType, typename ValueType>
__global__ void UpdateFrontierStatus(IndexType *vertex_ft, IndexType *vertex_ft_cnt, int *status, int level){
    IndexType cta_offset = blockIdx.x * blockDim.x;
    IndexType total_ft_cnt = *vertex_ft_cnt;
    while (cta_offset < total_ft_cnt){
        if (cta_offset + threadIdx.x < total_ft_cnt){
            IndexType u = vertex_ft[cta_offset + threadIdx.x];
            status[u] = level;
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}

#endif
