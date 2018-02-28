#ifndef __PPR_REV_PUSH_GPU_H__
#define __PPR_REV_PUSH_GPU_H__

#include "PPRGPU.cuh"
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/reduce.h>

class PPRRevPushGPU : public PPRGPU{
public:
	PPRRevPushGPU(GraphVec *g) : PPRGPU(g){
        std::cout << "initialize gpu graph..." << std::endl;
        device_memory->CudaAllocGraph(false);
        device_memory->CudaMemcpyGraph(graph, false);
        // to get the out-degree
        device_memory->CudaMemcpyRowPtr(graph, true);
    }
	~PPRRevPushGPU(){
    }

    virtual void IncrementalBatchUpdate(){
        size_t blks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, device_memory->edge_batch->length);
        CopyOutDegree<IndexType, ValueType><<<blks_num, THREADS_PER_BLOCK>>>(device_memory->edge_batch->edge1, device_memory->edge_batch->edge2, device_memory->edge_batch->length, device_memory->row_ptr, device_memory->predeg);            
        RevertOutDegree<IndexType, ValueType><<<blks_num, THREADS_PER_BLOCK>>>(device_memory->edge_batch->edge1, device_memory->edge_batch->edge2, device_memory->edge_batch->is_insert, device_memory->edge_batch->length, device_memory->predeg);
        RevStreamUpdateOriginal<IndexType, ValueType><<<blks_num, THREADS_PER_BLOCK>>>(device_memory->edge_batch->edge1, device_memory->edge_batch->edge2, device_memory->edge_batch->is_insert, device_memory->edge_batch->length, 
            device_memory->predeg, device_memory->pagerank, 
            device_memory->residual, device_memory->locks, source_vertex_id);
    }
    
    void BuildSlidingGraphScratch(){
        SlidingGraphVec *dg = (SlidingGraphVec*)graph;
        EdgeBatch *edge_stream = new EdgeBatch(dg->edge_count);
        dg->SerializeEdgeStream(edge_stream);
        sliding_graph_builder->BuildInGraph(edge_stream, device_memory->in_row_ptr, device_memory->in_col_ind, device_memory->row_ptr);
        delete edge_stream;
        edge_stream = NULL;
    }
    void BuildSlidingGraphIncremental(){
        SlidingGraphVec *dg = (SlidingGraphVec*)graph;
        sliding_graph_builder->IncBuildInGraph(dg->new_stream, device_memory->in_row_ptr, device_memory->in_col_ind, device_memory->row_ptr);
    }
    virtual void GPUBuildSlidingGraph(){
        BuildSlidingGraphIncremental();
       // BuildSlidingGraphScratch();
#if defined(VALIDATE)
        SlidingGraphVec *dg = (SlidingGraphVec*)graph;
        dg->ScratchConstructWindowGraph();
        std::vector<std::vector<IndexType> > &cind_vec = dg->in_col_ind;
        
        IndexType *rptr = new IndexType[graph->vertex_count + 1];
        IndexType *cind = new IndexType[graph->edge_count];
        for (IndexType i = 0; i < graph->vertex_count; ++i) rptr[i] = cind_vec[i].size();
        IndexType prefix = 0;
        for (IndexType i = 0; i < graph->vertex_count; ++i){
            for (size_t j = 0; j < cind_vec[i].size(); ++j){
                size_t off = prefix + j;
                cind[off] = cind_vec[i][j];
            }
            IndexType tmp = rptr[i];
            rptr[i] = prefix;
            prefix += tmp;
        }
        rptr[graph->vertex_count] = prefix;
        for (IndexType i = 0; i < graph->vertex_count; ++i){
            assert(rptr[i+1]>=rptr[i]);
            std::sort(cind + rptr[i], cind + rptr[i+1]);
        }

        IndexType *in_row_ptr_copy = new IndexType[graph->vertex_count+1];
        IndexType *in_col_ind_copy = new IndexType[graph->edge_count];
        CUDA_ERROR(cudaMemcpy(in_row_ptr_copy, device_memory->in_row_ptr, sizeof(IndexType) * (graph->vertex_count + 1), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(in_col_ind_copy, device_memory->in_col_ind, sizeof(IndexType) * graph->edge_count, cudaMemcpyDeviceToHost));
        for (IndexType i = 0; i < graph->vertex_count; ++i){
            assert(in_row_ptr_copy[i] == rptr[i]);   
        }
        for (IndexType i = 0; i < graph->vertex_count; ++i){
            for (IndexType j = rptr[i]; j < rptr[i+1]; ++j){
                assert(cind[j] == in_col_ind_copy[j]);
            }
        }

        delete[] in_row_ptr_copy;
        in_row_ptr_copy = NULL;
        delete[] in_col_ind_copy;
        in_col_ind_copy = NULL;
        delete[] rptr;
        rptr = NULL;
        delete[] cind;
        cind = NULL;
#endif
    }

    virtual void ExecuteMainLoop(const size_t phase_id = 0){
        ExecuteOptimized(phase_id);
    }

    void ExecuteOptimized(const size_t phase_id = 0){
        IndexType blocks_num, vertex_frontier_count;
        GPUProfiler::StartTimer(INSPECT_TIME);
        CUDA_ERROR(cudaMemset(device_memory->vertex_ft_cnt, 0, sizeof(IndexType)));
        blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, graph->vertex_count);
        InspectPureRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_cnt,
            graph->vertex_count, device_memory->residual, phase_id, gTolerance);
        GPUProfiler::EndTimer(INSPECT_TIME);

        while (1){
            CUDA_ERROR(cudaMemcpy(&vertex_frontier_count, device_memory->vertex_ft_cnt, sizeof(IndexType), cudaMemcpyDeviceToHost));
            if (vertex_frontier_count == 0) break;            
#if defined(PROFILE)
            std::cout << "phase_id=" << phase_id << ",iteration_id=" << iteration_id << ",frontier_count=" << vertex_frontier_count << std::endl;
#endif
            CUDA_ERROR(cudaMemset(device_memory->vertex_ft_cnt2, 0, sizeof(IndexType)));

            GPUProfiler::StartTimer(EXPAND_TIME);
            blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, vertex_frontier_count);
            ExpandUnifiedRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, device_memory->vertex_ft2, device_memory->vertex_ft_cnt2,
            device_memory->in_row_ptr, device_memory->in_col_ind, device_memory->row_ptr, graph->vertex_count, 
            device_memory->residual, device_memory->pagerank, phase_id, gTolerance);
            GPUProfiler::EndTimer(EXPAND_TIME);

            GPUProfiler::StartTimer(REPAIR_FRONTIER_TIME);
            RepairFrontierRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, 
                device_memory->vertex_ft2, device_memory->vertex_ft_cnt2, 
                device_memory->residual, phase_id, gTolerance);
            GPUProfiler::EndTimer(REPAIR_FRONTIER_TIME);
            
            std::swap(device_memory->vertex_ft, device_memory->vertex_ft2);
            std::swap(device_memory->vertex_ft_cnt, device_memory->vertex_ft_cnt2);
            ++iteration_id;
        }
    }
            

    virtual void ValidateResult(){
        graph->ConstructGraph();
        ValueType *residual_copy = new ValueType[graph->vertex_count];
        ValueType *pagerank_copy = new ValueType[graph->vertex_count];
        CUDA_ERROR(cudaMemcpy(residual_copy, device_memory->residual, sizeof(ValueType) * graph->vertex_count, cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(pagerank_copy, device_memory->pagerank, sizeof(ValueType) * graph->vertex_count, cudaMemcpyDeviceToHost));

        for (IndexType u = 0; u < graph->vertex_count; ++u){
            assert (residual_copy[u] < gTolerance && residual_copy[u] > -gTolerance);
        }

        PPRCPUPowVec *ppr_pow = new PPRCPUPowVec(graph);
		ppr_pow->CalPPRRev(source_vertex_id);
		double *ans = ppr_pow->pagerank;
        const ValueType bound = gTolerance*100;
        for (IndexType u = 0; u < graph->vertex_count; ++u){
            ValueType err = ans[u] - pagerank_copy[u];
            if (err < 0) err = -err;
            if (err > bound){
                std::cout << err << "," << bound << std::endl;
            }
            assert(err < bound);
        }
        
        delete ppr_pow;
        ppr_pow = NULL;
        delete[] residual_copy;
        residual_copy = NULL;
        delete[] pagerank_copy;
        pagerank_copy = NULL;
    }

};


#endif
