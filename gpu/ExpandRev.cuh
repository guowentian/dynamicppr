#ifndef __EXPAND_REV_CUH__
#define __EXPAND_REV_CUH__

#include "Meta.h"


// optimized version: perform all tasks in one kernel: eager propagation + fast frontier 
template<typename IndexType, typename ValueType>
__global__ void ExpandUnifiedRev(IndexType *vertex_ft, ValueType *vertex_ft_r, IndexType *vertex_ft_cnt, IndexType *vertex_ft2, IndexType *vertex_ft_cnt2,
    IndexType *in_row_ptr, IndexType *in_col_ind, IndexType *row_ptr, IndexType vertex_count, 
    ValueType *residual, ValueType *pagerank, const size_t phase_id, const double tolerance){
    size_t thread_id = threadIdx.x;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP;
    
    typedef cub::BlockScan<IndexType, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    typedef cub::WarpScan<IndexType> WarpScan;
    __shared__ typename WarpScan::TempStorage warp_temp_storage[THREADS_PER_BLOCK / THREADS_PER_WARP];

    volatile __shared__ IndexType comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK / THREADS_PER_WARP];
    volatile __shared__ IndexType comm2[THREADS_PER_BLOCK];
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    volatile __shared__ size_t output_cta_offset;
    volatile __shared__ size_t output_warp_offset[THREADS_PER_BLOCK / THREADS_PER_WARP];

    IndexType total_ft_cnt = *vertex_ft_cnt;
    IndexType cta_offset = blockDim.x * blockIdx.x;
    IndexType row_start, row_end;
    IndexType u;
    ValueType ru;
    while (cta_offset < total_ft_cnt){
        if (cta_offset + thread_id < total_ft_cnt){
            u = vertex_ft[cta_offset + thread_id];
            row_start = in_row_ptr[u];
            row_end = in_row_ptr[u+1];

            ru = residual[u];
            vertex_ft_r[cta_offset + thread_id] = ru;
            pagerank[u] += ALPHA * ru;
        }
        else{
            row_start = 0;
            row_end = 0;
        }

        // cta-based coarse grained gathering
        while (__syncthreads_or((row_end - row_start) >= THREADS_PER_BLOCK)){
            // if there is any vertex with big out-degree
            if (row_end - row_start >= THREADS_PER_BLOCK){
                comm[0][0] = thread_id;
            }
            __syncthreads();
            if (comm[0][0] == thread_id){
                // winner
                comm[0][1] = row_start;
                comm[0][2] = row_end;
                commr[0] = ru;
                row_start = row_end;
            }
            __syncthreads();

            size_t gather_st = comm[0][1] + thread_id;
            size_t gather_ed = comm[0][2];
            while (__syncthreads_or(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[0] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(prer, phase_id, tolerance) == false && IsLegalRevPush(curr, phase_id, tolerance)){
                        thread_count = 1;
                    }
                }
                if (__syncthreads_or(thread_count > 0)){
                    IndexType scatter, total;
                    BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);
                    __syncthreads();
                    if (thread_id == 0){
                        output_cta_offset = atomicAdd(vertex_ft_cnt2, total); 
                    }
                    __syncthreads();
                    if (thread_count > 0){
                        vertex_ft2[output_cta_offset + scatter] = v; 
                    }
                }
                gather_st += THREADS_PER_BLOCK;
            }
        }

        // warp-based medium grained gathering
        while (cub::WarpAny((row_end - row_start) >= THREADS_PER_WARP)){
            if (row_end - row_start >= THREADS_PER_WARP){
                comm[warp_id][0] = lane_id;
            }
            if (comm[warp_id][0] == lane_id){
                comm[warp_id][1] = row_start;
                comm[warp_id][2] = row_end;
                commr[warp_id] = ru;
                row_start = row_end;
            }
            size_t gather_st = comm[warp_id][1] + lane_id;
            size_t gather_ed = comm[warp_id][2];
            while (cub::WarpAny(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[warp_id] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(prer, phase_id, tolerance) == false && IsLegalRevPush(curr, phase_id, tolerance)){
                        thread_count = 1;
                    }
                }
                if (cub::WarpAny(thread_count > 0)){
                    IndexType scatter, total;
                    WarpScan(warp_temp_storage[warp_id]).ExclusiveSum(thread_count, scatter, total);
                    if (lane_id == 0){
                        output_warp_offset[warp_id] = atomicAdd(vertex_ft_cnt2, total);
                    }
                    if (thread_count > 0){
                        vertex_ft2[output_warp_offset[warp_id] + scatter] = v;
                    }
                }
                gather_st += THREADS_PER_WARP;
            }
        }
        
        IndexType thread_count = row_end - row_start;
        IndexType rsv_rank, total;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_count, rsv_rank, total);
        __syncthreads();
        
        size_t cta_progress = 0;
        while (cta_progress < total){
            size_t remain = total - cta_progress;
            while (rsv_rank < cta_progress + THREADS_PER_BLOCK && row_start < row_end){
                comm2[rsv_rank - cta_progress] = row_start;
                commr2[rsv_rank - cta_progress] = ru;
                rsv_rank++;
                row_start++;
            } 
            __syncthreads();
            size_t cur_batch_count = MIN(remain, THREADS_PER_BLOCK);
            thread_count = 0;
            IndexType v;
            if (thread_id < cur_batch_count){
                v = in_col_ind[comm2[thread_id]];
                IndexType degv = row_ptr[v+1] - row_ptr[v];
                ValueType add = (1.0 - ALPHA) * commr2[thread_id] / (degv + 1);
                ValueType prer = atomicAdd(residual + v, add);
                ValueType curr = prer + add;
                if (IsLegalRevPush(prer, phase_id, tolerance) == false && IsLegalRevPush(curr, phase_id, tolerance)){
                    thread_count = 1;
                }
            }
           if (__syncthreads_or(thread_count > 0)){
                IndexType scatter, total2;
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total2);
                __syncthreads();
                if (thread_id == 0){
                    output_cta_offset = atomicAdd(vertex_ft_cnt2, total2);
                }
                __syncthreads();
                if (thread_count > 0){
                    assert(thread_id < cur_batch_count);
                    vertex_ft2[output_cta_offset + scatter] = v;
                }
            }
            
            cta_progress += THREADS_PER_BLOCK;
        }
        
        cta_offset += blockDim.x * gridDim.x;
    }
}

// eager propagation kernel
template<typename IndexType, typename ValueType>
__global__ void ExpandEagerRev(IndexType *vertex_ft, ValueType *vertex_ft_r, IndexType *vertex_ft_cnt, IndexType *vertex_ft2, IndexType *vertex_ft_cnt2,
    IndexType *in_row_ptr, IndexType *in_col_ind, IndexType *row_ptr, IndexType vertex_count, 
    ValueType *residual, ValueType *pagerank, int *status, const int level, const size_t phase_id, const double tolerance){
    size_t thread_id = threadIdx.x;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP;
    
    typedef cub::BlockScan<IndexType, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    typedef cub::WarpScan<IndexType> WarpScan;
    __shared__ typename WarpScan::TempStorage warp_temp_storage[THREADS_PER_BLOCK / THREADS_PER_WARP];

    volatile __shared__ IndexType comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK / THREADS_PER_WARP];
    volatile __shared__ IndexType comm2[THREADS_PER_BLOCK];
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    volatile __shared__ size_t output_cta_offset;
    volatile __shared__ size_t output_warp_offset[THREADS_PER_BLOCK / THREADS_PER_WARP];


    IndexType total_ft_cnt = *vertex_ft_cnt;
    IndexType cta_offset = blockDim.x * blockIdx.x;
    IndexType row_start, row_end;
    IndexType u;
    ValueType ru;
    while (cta_offset < total_ft_cnt){
        if (cta_offset + thread_id < total_ft_cnt){
            u = vertex_ft[cta_offset + thread_id];
            row_start = in_row_ptr[u];
            row_end = in_row_ptr[u+1];

            ru = residual[u];
            vertex_ft_r[cta_offset + thread_id] = ru;
            pagerank[u] += ALPHA * ru;
        }
        else{
            row_start = 0;
            row_end = 0;
        }

        // cta-based coarse grained gathering
        while (__syncthreads_or((row_end - row_start) >= THREADS_PER_BLOCK)){
            // if there is any vertex with big out-degree
            if (row_end - row_start >= THREADS_PER_BLOCK){
                comm[0][0] = thread_id;
            }
            __syncthreads();
            if (comm[0][0] == thread_id){
                // winner
                comm[0][1] = row_start;
                comm[0][2] = row_end;
                commr[0] = ru;
                row_start = row_end;
            }
            __syncthreads();

            size_t gather_st = comm[0][1] + thread_id;
            size_t gather_ed = comm[0][2];
            while (__syncthreads_or(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[0] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(curr, phase_id, tolerance) == true){
                        int old_level = atomicExch(status + v, level);
                        if (old_level < level) thread_count = 1;
                    }
                }
                IndexType scatter, total;
                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);
                __syncthreads();
                if (thread_id == 0){
                    output_cta_offset = atomicAdd(vertex_ft_cnt2, total); 
                }
                __syncthreads();
                if (thread_count > 0){
                    vertex_ft2[output_cta_offset + scatter] = v; 
                }
                gather_st += THREADS_PER_BLOCK;
            }
        }

        // warp-based medium grained gathering
        while (cub::WarpAny((row_end - row_start) >= THREADS_PER_WARP)){
            if (row_end - row_start >= THREADS_PER_WARP){
                comm[warp_id][0] = lane_id;
            }
            // no need to sync since thread execution inside warp is synchronous
            if (comm[warp_id][0] == lane_id){
                comm[warp_id][1] = row_start;
                comm[warp_id][2] = row_end;
                commr[warp_id] = ru;
                row_start = row_end;
            }
            size_t gather_st = comm[warp_id][1] + lane_id;
            size_t gather_ed = comm[warp_id][2];
            while (cub::WarpAny(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[warp_id] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(curr, phase_id, tolerance) == true){
                        int old_level = atomicExch(status + v, level);
                        if (old_level < level) thread_count = 1;
                    }
                }
                IndexType scatter, total;
                WarpScan(warp_temp_storage[warp_id]).ExclusiveSum(thread_count, scatter, total);
                if (lane_id == 0){
                    output_warp_offset[warp_id] = atomicAdd(vertex_ft_cnt2, total);
                }
                if (thread_count > 0){
                    vertex_ft2[output_warp_offset[warp_id] + scatter] = v;
                }
                gather_st += THREADS_PER_WARP;
            }
        }
        
        IndexType thread_count = row_end - row_start;
        IndexType rsv_rank, total;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_count, rsv_rank, total);
        __syncthreads();
        
        size_t cta_progress = 0;
        while (cta_progress < total){
            size_t remain = total - cta_progress;
            while (rsv_rank < cta_progress + THREADS_PER_BLOCK && row_start < row_end){
                comm2[rsv_rank - cta_progress] = row_start;
                commr2[rsv_rank - cta_progress] = ru;
                rsv_rank++;
                row_start++;
            } 
            __syncthreads();
            size_t cur_batch_count = MIN(remain, THREADS_PER_BLOCK);
            thread_count = 0;
            IndexType v;
            if (thread_id < cur_batch_count){
                v = in_col_ind[comm2[thread_id]];
                IndexType degv = row_ptr[v+1] - row_ptr[v];
                ValueType add = (1.0 - ALPHA) * commr2[thread_id] / (degv + 1);
                ValueType prer = atomicAdd(residual + v, add);
                ValueType curr = prer + add;
                if (IsLegalRevPush(curr, phase_id, tolerance) == true){
                    int old_level = atomicExch(status + v, level);
                    if (old_level < level) thread_count = 1;
                }
            }
            IndexType scatter, total2;
            __syncthreads();
            BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total2);
            __syncthreads();
            if (thread_id == 0){
                output_cta_offset = atomicAdd(vertex_ft_cnt2, total2);
            }
            __syncthreads();
            if (thread_count > 0){
                vertex_ft2[output_cta_offset + scatter] = v;
            }

            cta_progress += THREADS_PER_BLOCK;
        }
        
        cta_offset += blockDim.x * gridDim.x;
    }
}
// fast frontier kernel
template<typename IndexType, typename ValueType>
__global__ void ExpandFastFrontierRev(IndexType *vertex_ft, ValueType *vertex_ft_r, IndexType *vertex_ft_cnt, IndexType *vertex_ft2, IndexType *vertex_ft_cnt2,
    IndexType *in_row_ptr, IndexType *in_col_ind, IndexType *row_ptr, IndexType vertex_count, 
    ValueType *residual, const size_t phase_id, const double tolerance){
    size_t thread_id = threadIdx.x;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP;
    
    typedef cub::BlockScan<IndexType, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    typedef cub::WarpScan<IndexType> WarpScan;
    __shared__ typename WarpScan::TempStorage warp_temp_storage[THREADS_PER_BLOCK / THREADS_PER_WARP];

    volatile __shared__ IndexType comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK / THREADS_PER_WARP];
    volatile __shared__ IndexType comm2[THREADS_PER_BLOCK];
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    volatile __shared__ size_t output_cta_offset;
    volatile __shared__ size_t output_warp_offset[THREADS_PER_BLOCK / THREADS_PER_WARP];


    IndexType total_ft_cnt = *vertex_ft_cnt;
    IndexType cta_offset = blockDim.x * blockIdx.x;
    IndexType row_start, row_end;
    IndexType u;
    ValueType ru;
    while (cta_offset < total_ft_cnt){
        if (cta_offset + thread_id < total_ft_cnt){
            u = vertex_ft[cta_offset + thread_id];
            ru = vertex_ft_r[cta_offset + thread_id];
            row_start = in_row_ptr[u];
            row_end = in_row_ptr[u+1];
        }
        else{
            row_start = 0;
            row_end = 0;
        }

        // cta-based coarse grained gathering
        while (__syncthreads_or((row_end - row_start) >= THREADS_PER_BLOCK)){
            // if there is any vertex with big out-degree
            if (row_end - row_start >= THREADS_PER_BLOCK){
                comm[0][0] = thread_id;
            }
            __syncthreads();
            if (comm[0][0] == thread_id){
                // winner
                comm[0][1] = row_start;
                comm[0][2] = row_end;
                commr[0] = ru;
                row_start = row_end;
            }
            __syncthreads();

            size_t gather_st = comm[0][1] + thread_id;
            size_t gather_ed = comm[0][2];
            while (__syncthreads_or(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[0] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(prer, phase_id, tolerance) == false && IsLegalRevPush(curr, phase_id, tolerance) == true){
                        thread_count = 1;
                    }
                }
                IndexType scatter, total;
                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);
                __syncthreads();
                if (thread_id == 0){
                    output_cta_offset = atomicAdd(vertex_ft_cnt2, total); 
                }
                __syncthreads();
                if (thread_count > 0){
                    vertex_ft2[output_cta_offset + scatter] = v; 
                }
                gather_st += THREADS_PER_BLOCK;
            }
        }

        // warp-based medium grained gathering
        while (cub::WarpAny((row_end - row_start) >= THREADS_PER_WARP)){
            if (row_end - row_start >= THREADS_PER_WARP){
                comm[warp_id][0] = lane_id;
            }
            // no need to sync since thread execution inside warp is synchronous
            if (comm[warp_id][0] == lane_id){
                comm[warp_id][1] = row_start;
                comm[warp_id][2] = row_end;
                commr[warp_id] = ru;
                row_start = row_end;
            }
            size_t gather_st = comm[warp_id][1] + lane_id;
            size_t gather_ed = comm[warp_id][2];
            while (cub::WarpAny(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[warp_id] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(prer, phase_id, tolerance) == false && IsLegalRevPush(curr, phase_id, tolerance) == true){
                        thread_count = 1;
                    }
                }
                IndexType scatter, total;
                WarpScan(warp_temp_storage[warp_id]).ExclusiveSum(thread_count, scatter, total);
                if (lane_id == 0){
                    output_warp_offset[warp_id] = atomicAdd(vertex_ft_cnt2, total);
                }
                if (thread_count > 0){
                    vertex_ft2[output_warp_offset[warp_id] + scatter] = v;
                }
                gather_st += THREADS_PER_WARP;
            }
        }
        
        IndexType thread_count = row_end - row_start;
        IndexType rsv_rank, total;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_count, rsv_rank, total);
        __syncthreads();
        
        size_t cta_progress = 0;
        while (cta_progress < total){
            size_t remain = total - cta_progress;
            while (rsv_rank < cta_progress + THREADS_PER_BLOCK && row_start < row_end){
                comm2[rsv_rank - cta_progress] = row_start;
                commr2[rsv_rank - cta_progress] = ru;
                rsv_rank++;
                row_start++;
            } 
            __syncthreads();
            size_t cur_batch_count = MIN(remain, THREADS_PER_BLOCK);
            thread_count = 0;
            IndexType v;
            if (thread_id < cur_batch_count){
                v = in_col_ind[comm2[thread_id]];
                IndexType degv = row_ptr[v+1] - row_ptr[v];
                ValueType add = (1.0 - ALPHA) * commr2[thread_id] / (degv + 1);
                ValueType prer = atomicAdd(residual + v, add);
                ValueType curr = prer + add;
                if (IsLegalRevPush(prer, phase_id, tolerance) == false && IsLegalRevPush(curr, phase_id, tolerance)){
                    thread_count = 1;
                }
            }
            IndexType scatter, total2;
            __syncthreads();
            BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total2);
            __syncthreads();
            if (thread_id == 0){
                output_cta_offset = atomicAdd(vertex_ft_cnt2, total2);
            }
            __syncthreads();
            if (thread_count > 0){
                vertex_ft2[output_cta_offset + scatter] = v;
            }

            cta_progress += THREADS_PER_BLOCK;
        }
        
        cta_offset += blockDim.x * gridDim.x;
    }

}    
template<typename IndexType, typename ValueType>
__global__ void ExpandVanillaRev(IndexType *vertex_ft, ValueType *vertex_ft_r, IndexType *vertex_ft_cnt, IndexType *vertex_ft2, IndexType *vertex_ft_cnt2,
    IndexType *in_row_ptr, IndexType *in_col_ind, IndexType *row_ptr, IndexType vertex_count, 
    ValueType *residual, int *status, const int level, const size_t phase_id, const double tolerance){
    size_t thread_id = threadIdx.x;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP;
    
    typedef cub::BlockScan<IndexType, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    typedef cub::WarpScan<IndexType> WarpScan;
    __shared__ typename WarpScan::TempStorage warp_temp_storage[THREADS_PER_BLOCK / THREADS_PER_WARP];

    volatile __shared__ IndexType comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK / THREADS_PER_WARP];
    volatile __shared__ IndexType comm2[THREADS_PER_BLOCK];
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    volatile __shared__ size_t output_cta_offset;
    volatile __shared__ size_t output_warp_offset[THREADS_PER_BLOCK / THREADS_PER_WARP];


    IndexType total_ft_cnt = *vertex_ft_cnt;
    IndexType cta_offset = blockDim.x * blockIdx.x;
    IndexType row_start, row_end;
    IndexType u;
    ValueType ru;
    while (cta_offset < total_ft_cnt){
        if (cta_offset + thread_id < total_ft_cnt){
            u = vertex_ft[cta_offset + thread_id];
            ru = vertex_ft_r[cta_offset + thread_id];
            row_start = in_row_ptr[u];
            row_end = in_row_ptr[u+1];
        }
        else{
            row_start = 0;
            row_end = 0;
        }

        // cta-based coarse grained gathering
        while (__syncthreads_or((row_end - row_start) >= THREADS_PER_BLOCK)){
            // if there is any vertex with big out-degree
            if (row_end - row_start >= THREADS_PER_BLOCK){
                comm[0][0] = thread_id;
            }
            __syncthreads();
            if (comm[0][0] == thread_id){
                // winner
                comm[0][1] = row_start;
                comm[0][2] = row_end;
                commr[0] = ru;
                row_start = row_end;
            }
            __syncthreads();

            size_t gather_st = comm[0][1] + thread_id;
            size_t gather_ed = comm[0][2];
            while (__syncthreads_or(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[0] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(curr, phase_id, tolerance) == true){
                        int old_level = atomicExch(status + v, level);
                        if (old_level < level) thread_count = 1;
                    }
                }
                IndexType scatter, total;
                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);
                __syncthreads();
                if (thread_id == 0){
                    output_cta_offset = atomicAdd(vertex_ft_cnt2, total); 
                }
                __syncthreads();
                if (thread_count > 0){
                    vertex_ft2[output_cta_offset + scatter] = v; 
                }
                gather_st += THREADS_PER_BLOCK;
            }
        }

        // warp-based medium grained gathering
        while (cub::WarpAny((row_end - row_start) >= THREADS_PER_WARP)){
            if (row_end - row_start >= THREADS_PER_WARP){
                comm[warp_id][0] = lane_id;
            }
            if (comm[warp_id][0] == lane_id){
                comm[warp_id][1] = row_start;
                comm[warp_id][2] = row_end;
                commr[warp_id] = ru;
                row_start = row_end;
            }
            size_t gather_st = comm[warp_id][1] + lane_id;
            size_t gather_ed = comm[warp_id][2];
            while (cub::WarpAny(gather_st < gather_ed)){
                IndexType thread_count = 0;
                IndexType v;
                if (gather_st < gather_ed){
                    v = in_col_ind[gather_st];
                    IndexType degv = row_ptr[v+1] - row_ptr[v];
                    ValueType add = (1.0 - ALPHA) * commr[warp_id] / (degv + 1);
                    ValueType prer = atomicAdd(residual + v, add);
                    ValueType curr = prer + add;
                    if (IsLegalRevPush(curr, phase_id, tolerance) == true){
                        int old_level = atomicExch(status + v, level);
                        if (old_level < level) thread_count = 1;
                    }
                }
                IndexType scatter, total;
                WarpScan(warp_temp_storage[warp_id]).ExclusiveSum(thread_count, scatter, total);
                if (lane_id == 0){
                    output_warp_offset[warp_id] = atomicAdd(vertex_ft_cnt2, total);
                }
                if (thread_count > 0){
                    vertex_ft2[output_warp_offset[warp_id] + scatter] = v;
                }
                gather_st += THREADS_PER_WARP;
            }
        }
        
        IndexType thread_count = row_end - row_start;
        IndexType rsv_rank, total;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_count, rsv_rank, total);
        __syncthreads();
        
        size_t cta_progress = 0;
        while (cta_progress < total){
            size_t remain = total - cta_progress;
            while (rsv_rank < cta_progress + THREADS_PER_BLOCK && row_start < row_end){
                comm2[rsv_rank - cta_progress] = row_start;
                commr2[rsv_rank - cta_progress] = ru;
                rsv_rank++;
                row_start++;
            } 
            __syncthreads();
            size_t cur_batch_count = MIN(remain, THREADS_PER_BLOCK);
            thread_count = 0;
            IndexType v;
            if (thread_id < cur_batch_count){
                v = in_col_ind[comm2[thread_id]];
                IndexType degv = row_ptr[v+1] - row_ptr[v];
                ValueType add = (1.0 - ALPHA) * commr2[thread_id] / (degv + 1);
                ValueType prer = atomicAdd(residual + v, add);
                ValueType curr = prer + add;
                if (IsLegalRevPush(curr, phase_id, tolerance) == true){
                    int old_level = atomicExch(status + v, level);
                    if (old_level < level) thread_count = 1;
                }
            }
            IndexType scatter, total2;
            __syncthreads();
            BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total2);
            __syncthreads();
            if (thread_id == 0){
                output_cta_offset = atomicAdd(vertex_ft_cnt2, total2);
            }
            __syncthreads();
            if (thread_count > 0){
                vertex_ft2[output_cta_offset + scatter] = v;
            }

            cta_progress += THREADS_PER_BLOCK;
        }
        
        cta_offset += blockDim.x * gridDim.x;
    }

}    
// repair residuals of frontier vertices by taking the residual amount that should be taken
template<typename IndexType, typename ValueType>
__global__ void RepairFrontierRev(IndexType *vertex_ft, ValueType *vertex_ft_r, IndexType *vertex_ft_cnt, 
    IndexType *vertex_ft2, IndexType *vertex_ft_cnt2, 
    ValueType *residual, const size_t phase_id, const double tolerance){
    size_t thread_id = threadIdx.x;
    typedef cub::BlockScan<IndexType, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    volatile __shared__ IndexType output_cta_offset; 

    IndexType cta_offset = blockIdx.x * blockDim.x;
    IndexType total_ft_cnt = *vertex_ft_cnt;
    IndexType u;
    while (cta_offset < total_ft_cnt){
        IndexType thread_count = 0;
        if (cta_offset + thread_id < total_ft_cnt){
            u = vertex_ft[cta_offset + thread_id];
            residual[u] -= vertex_ft_r[cta_offset + thread_id];
            if (IsLegalRevPush(residual[u], phase_id, tolerance)){
                thread_count = 1;
            }
        }
        if (__syncthreads_or(thread_count > 0)){
            IndexType scatter, total;
            BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);
            __syncthreads();
            if (thread_id == 0){
                output_cta_offset = atomicAdd(vertex_ft_cnt2, total); 
            }
            __syncthreads();
            if (thread_count > 0){
                vertex_ft2[output_cta_offset + scatter] = u; 
            }
        }
        cta_offset += blockDim.x * gridDim.x;
    }
}



#endif
