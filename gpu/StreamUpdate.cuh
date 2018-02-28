#ifndef __STREAM_UPDATE_H__
#define __STREAM_UPDATE_H__

#include "PPRCommon.cuh"
#include <cub/cub.cuh>

template<typename IndexType, typename ValueType>
__global__ void CopyOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, IndexType batch_length, IndexType *row_ptr, IndexType *pre_deg){
    size_t cta_offset = blockIdx.x * blockDim.x;
    while (cta_offset < batch_length){
        if (cta_offset + threadIdx.x < batch_length){
            IndexType u = edge_batch1[cta_offset + threadIdx.x];
            pre_deg[u] = row_ptr[u+1] - row_ptr[u];
        }
        cta_offset += blockDim.x * gridDim.x;
    }    
}
template<typename IndexType, typename ValueType>
__global__ void RevertOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType batch_length, IndexType *deg){
    size_t cta_offset = blockIdx.x * blockDim.x;
    while (cta_offset < batch_length){
        if (cta_offset + threadIdx.x < batch_length){
            IndexType u = edge_batch1[cta_offset + threadIdx.x];
            if (is_insert[cta_offset + threadIdx.x]){
            	atomicAdd(deg + u, -1);
            }
            else{
            	atomicAdd(deg + u, 1);
            }
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}
template<typename IndexType, typename ValueType>
__global__ void RevStreamUpdateOriginal(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType batch_length, 
    IndexType *deg, ValueType *pagerank, ValueType *residual, int* locks, const IndexType target_vertex_id){
    size_t thread_id = threadIdx.x;
    size_t cta_offset = blockIdx.x * blockDim.x;
    IndexType u, v;
    bool ins;
    bool valid;
    while (cta_offset < batch_length){
        valid = false;
        if (cta_offset + thread_id < batch_length){
            u = edge_batch1[cta_offset + thread_id];
            v = edge_batch2[cta_offset + thread_id];
            ins = is_insert[cta_offset + thread_id];
            valid = true;
        }
        while (cub::WarpAny(valid)){
            if (valid){
                int lock_val = atomicCAS(locks + u, 0, 1);
                if (lock_val == 0){
                    // lock[u] imposed on residual[u] and deg[u]
                    valid = false;
                    if (ins){
                        deg[u]++;
                        ValueType add_value = (1.0 - ALPHA) * pagerank[v] - pagerank[u] - ALPHA * residual[u] + ALPHA * (u == target_vertex_id ? 1.0 : 0.0);
                        add_value = add_value / (deg[u] + 1) / ALPHA;
                        residual[u] += add_value;
                    }
                    else{
                        deg[u]--;
                        ValueType add_value = (1.0 - ALPHA) * pagerank[v] - pagerank[u] - ALPHA * residual[u] + ALPHA * (u == target_vertex_id ? 1.0 : 0.0);
                        add_value = add_value / (deg[u] + 1) / ALPHA;
                        residual[u] -= add_value;
                    }
                    
                    atomicExch(locks + u, 0);    
                }
                __threadfence();
            }
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}

#endif
