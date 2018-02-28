#ifndef __GPU_EDGE_BATCH_CUH__
#define __GPU_EDGE_BATCH_CUH__

#include "Meta.h"
#include "EdgeBatch.h"
#include "GPUUtil.cuh"

class GPUEdgeBatch{
public:
    GPUEdgeBatch(IndexType sz) : size(sz), length(0){
        CUDA_ERROR(cudaMalloc(&edge1, sizeof(IndexType) * size));
        CUDA_ERROR(cudaMalloc(&edge2, sizeof(IndexType) * size));
        CUDA_ERROR(cudaMalloc(&is_insert, sizeof(bool) * size));
    }
    ~GPUEdgeBatch(){
        CUDA_ERROR(cudaFree(edge1));
        CUDA_ERROR(cudaFree(edge2));
        CUDA_ERROR(cudaFree(is_insert));
    }
    void CudaMemcpy(EdgeBatch *edge_batch, cudaMemcpyKind kind){
        if (kind == cudaMemcpyHostToDevice){
            CUDA_ERROR(cudaMemcpy(edge1, edge_batch->edge1, sizeof(IndexType) * edge_batch->length, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(edge2, edge_batch->edge2, sizeof(IndexType) * edge_batch->length, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(is_insert, edge_batch->is_insert, sizeof(bool) * edge_batch->length, cudaMemcpyHostToDevice));
            length = edge_batch->length; 
        }
        else if (kind == cudaMemcpyDeviceToHost){
            CUDA_ERROR(cudaMemcpy(edge_batch->edge1, edge1, sizeof(IndexType) * length, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(edge_batch->edge2, edge2, sizeof(IndexType) * length, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(edge_batch->is_insert, is_insert, sizeof(bool) * length, cudaMemcpyDeviceToHost));
            edge_batch->length = length;
        }
    }
public:
    IndexType *edge1;
    IndexType *edge2;
    bool *is_insert;

    IndexType length, size;
};

#endif

