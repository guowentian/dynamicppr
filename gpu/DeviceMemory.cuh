#ifndef __DEVICE_MEMORY_CUH__
#define __DEVICE_MEMORY_CUH__

#include "GPUUtil.cuh"
#include "Meta.h"
#include "GraphVec.h"
#include "GPUEdgeBatch.cuh"

class DeviceMemory{
    public:
        DeviceMemory(IndexType vcount, IndexType ecount) : vertex_count(vcount), edge_count(ecount){
            row_ptr = NULL;
            col_ind = NULL;
            in_row_ptr = NULL;
            in_col_ind = NULL;

            pagerank = NULL;
            residual = NULL;

            vertex_ft = NULL;
            vertex_ft_cnt = NULL;
            vertex_ft2 = NULL;
            vertex_ft_cnt2 = NULL;
            vertex_ft_r = NULL;
            status = NULL;

            locks = NULL;
            predeg = NULL;
            edge_batch = NULL;
        }
        void FreeGPUMemory(){
            if (row_ptr) CUDA_ERROR(cudaFree(row_ptr));
            if (col_ind) CUDA_ERROR(cudaFree(col_ind));
            if (in_row_ptr) CUDA_ERROR(cudaFree(in_row_ptr));
            if (in_col_ind) CUDA_ERROR(cudaFree(in_col_ind));
            
            if (pagerank) CUDA_ERROR(cudaFree(pagerank));
            if (residual) CUDA_ERROR(cudaFree(residual));

            if (vertex_ft) CUDA_ERROR(cudaFree(vertex_ft));
            if (vertex_ft_cnt) CUDA_ERROR(cudaFree(vertex_ft_cnt));
            if (vertex_ft2) CUDA_ERROR(cudaFree(vertex_ft2));
            if (vertex_ft_cnt2) CUDA_ERROR(cudaFree(vertex_ft_cnt2));
            if (vertex_ft_r) CUDA_ERROR(cudaFree(vertex_ft_r));
            if (status) CUDA_ERROR(cudaFree(status));

            if (locks) CUDA_ERROR(cudaFree(locks));
            if (predeg) CUDA_ERROR(cudaFree(predeg));
            if (edge_batch) CUDA_ERROR(cudaFree(edge_batch));
        }
        void CudaAllocGraph(bool out_deg){
            CUDA_ERROR(cudaMalloc((out_deg ? &row_ptr : &in_row_ptr), sizeof(IndexType) * (vertex_count + 1)));
            CUDA_ERROR(cudaMalloc((out_deg ? &col_ind : &in_col_ind), sizeof(IndexType) * edge_count));
        }
        void CudaAllocAppData(){
            // application
            CUDA_ERROR(cudaMalloc(&pagerank, sizeof(ValueType) * (vertex_count + 1)));
            CUDA_ERROR(cudaMalloc(&residual, sizeof(ValueType) * (vertex_count + 1)));

            // frontier
            CUDA_ERROR(cudaMalloc(&vertex_ft, sizeof(IndexType) * vertex_count));
            CUDA_ERROR(cudaMalloc(&vertex_ft_cnt, sizeof(IndexType)));
            CUDA_ERROR(cudaMalloc(&vertex_ft2, sizeof(IndexType) * vertex_count));
            CUDA_ERROR(cudaMalloc(&vertex_ft_cnt2, sizeof(IndexType)));
            CUDA_ERROR(cudaMalloc(&vertex_ft_r, sizeof(ValueType) * vertex_count));
            CUDA_ERROR(cudaMalloc(&status, sizeof(IndexType) * vertex_count));
            CUDA_ERROR(cudaMalloc(&predeg, sizeof(IndexType) * vertex_count));
        }
        void InitForDynamicGraph(){
            edge_batch = new GPUEdgeBatch(gStreamUpdateCountPerBatch * 4);
            CUDA_ERROR(cudaMalloc(&predeg, sizeof(IndexType) * vertex_count));
            CUDA_ERROR(cudaMalloc(&locks, sizeof(int) * vertex_count));
            CUDA_ERROR(cudaMemset(locks, 0, sizeof(int) * vertex_count));
        }

        void CudaMemcpyRowPtr(GraphVec *graph, bool out_deg){
            if (out_deg && row_ptr == NULL) CUDA_ERROR(cudaMalloc(&row_ptr, sizeof(IndexType)*(vertex_count+1)));
            if (!out_deg && in_row_ptr == NULL) CUDA_ERROR(cudaMalloc(&in_row_ptr, sizeof(IndexType)*(vertex_count+1)));
            IndexType *rptr = new IndexType[vertex_count + 1];
            std::vector<std::vector<IndexType> >& col_ind_vec = out_deg ? graph->col_ind : graph->in_col_ind;
            for (IndexType i = 0; i < vertex_count; ++i) rptr[i] = col_ind_vec[i].size();
            for (IndexType i = 0, prefix = 0; i < vertex_count+1; ++i){
                IndexType tmp = rptr[i];
                rptr[i] = prefix;
                prefix += tmp;
            }
            CUDA_ERROR(cudaMemcpy(out_deg ? row_ptr : in_row_ptr, rptr, sizeof(IndexType) * (vertex_count+1), cudaMemcpyHostToDevice));
            delete[] rptr;
            rptr = NULL;
        }
        void CudaMemcpyColInd(GraphVec *graph, bool out_deg){
            if (out_deg && col_ind == NULL) CUDA_ERROR(cudaMalloc(&col_ind, sizeof(IndexType)*edge_count));
            if (!out_deg && in_col_ind == NULL) CUDA_ERROR(cudaMalloc(&in_col_ind, sizeof(IndexType)*edge_count));
            std::vector<std::vector<IndexType> >& col_ind_vec = out_deg ? graph->col_ind : graph->in_col_ind;
            IndexType *cind = new IndexType[edge_count];
            IndexType prev_edge_num = 0;
            for (IndexType i = 0; i < vertex_count; ++i){
                for (size_t j = 0; j < col_ind_vec[i].size(); ++j){
                    IndexType off = prev_edge_num + j;
                    cind[off] = col_ind_vec[i][j];
                }
                std::sort(cind + prev_edge_num, cind + prev_edge_num + col_ind_vec[i].size());
                prev_edge_num += col_ind_vec[i].size();
            }
            CUDA_ERROR(cudaMemcpy(out_deg ? col_ind : in_col_ind, cind, sizeof(IndexType) * edge_count, cudaMemcpyHostToDevice));
            delete[] cind;
            cind = NULL;
        }
        void CudaMemcpyGraph(GraphVec* graph, bool out_deg){
            CudaMemcpyRowPtr(graph, out_deg);
            CudaMemcpyColInd(graph, out_deg);
        }
        
    public:
        //graph
        IndexType *row_ptr;
        IndexType *col_ind;
        IndexType *in_row_ptr;
        IndexType *in_col_ind;
        IndexType vertex_count, edge_count;
        // app
        ValueType *pagerank;
        ValueType *residual;
        //frontier
        IndexType *vertex_ft;
        IndexType *vertex_ft_cnt;
        IndexType *vertex_ft2;
        IndexType *vertex_ft_cnt2;
        ValueType *vertex_ft_r;
        int *status;
        //dynamic graph
        int *locks;
        IndexType *predeg;
        GPUEdgeBatch *edge_batch;
};

#endif

