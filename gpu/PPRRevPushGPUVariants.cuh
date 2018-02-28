#ifndef __PPR_REV_PUSH_GPU_VARIANTS_CUH__
#define __PPR_REV_PUSH_GPU_VARIANTS_CUH__

#include "PPRRevPushGPU.cuh"

class PPRRevPushGPUEager : public PPRRevPushGPU{
public:
    PPRRevPushGPUEager(GraphVec *graph) : PPRRevPushGPU(graph){}
    ~PPRRevPushGPUEager(){}

    virtual void ExecuteMainLoop(const size_t phase_id = 0){
        ExecuteEager(phase_id);
    }

    void ExecuteEager(const size_t phase_id = 0){
        IndexType blocks_num, vertex_frontier_count;
        GPUProfiler::StartTimer(INSPECT_TIME);

        CUDA_ERROR(cudaMemset(device_memory->vertex_ft_cnt, 0, sizeof(IndexType)));
        blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, graph->vertex_count);
        InspectPureRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_cnt,
        graph->vertex_count,
        device_memory->residual, phase_id, gTolerance);
        GPUProfiler::EndTimer(INSPECT_TIME);

        while (1){
            CUDA_ERROR(cudaMemcpy(&vertex_frontier_count, device_memory->vertex_ft_cnt, sizeof(IndexType), cudaMemcpyDeviceToHost));
            if (vertex_frontier_count == 0)
                break;
#if defined(VALIDATE)
            std::cout << "phase_id=" << phase_id << ",iteration_id=" << iteration_id << ",frontier_count=" << vertex_frontier_count << std::endl;
#endif
            blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, vertex_frontier_count);
            UpdateFrontierStatus<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_cnt, device_memory->status, iteration_id);

            GPUProfiler::StartTimer(EXPAND_TIME);
            CUDA_ERROR(cudaMemset(device_memory->vertex_ft_cnt2, 0, sizeof(IndexType)));
            ExpandEagerRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, device_memory->vertex_ft2, device_memory->vertex_ft_cnt2, 
                device_memory->in_row_ptr, device_memory->in_col_ind, device_memory->row_ptr, graph->vertex_count, 
                device_memory->residual, device_memory->pagerank, device_memory->status, iteration_id, phase_id, gTolerance);
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

};

class PPRRevPushGPUFF : public PPRRevPushGPU{
public:
    PPRRevPushGPUFF(GraphVec *graph) : PPRRevPushGPU(graph){}
    ~PPRRevPushGPUFF(){}

    virtual void ExecuteMainLoop(const size_t phase_id = 0){
        ExecuteFastFrontier(phase_id);
    }

    void ExecuteFastFrontier(const size_t phase_id = 0){
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
#if defined(VALIDATE)
            std::cout << "phase_id=" << phase_id << ",iteration_id=" << iteration_id << ",frontier_count=" << vertex_frontier_count << std::endl;
#endif
            GPUProfiler::StartTimer(INSPECT_TIME);
            blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, vertex_frontier_count);
            InspectExtra<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, 
            device_memory->residual, device_memory->pagerank, phase_id);
            GPUProfiler::EndTimer(INSPECT_TIME);

            GPUProfiler::StartTimer(EXPAND_TIME);
            CUDA_ERROR(cudaMemset(device_memory->vertex_ft_cnt2, 0, sizeof(IndexType)));
            blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, vertex_frontier_count);
            ExpandFastFrontierRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, device_memory->vertex_ft2, device_memory->vertex_ft_cnt2,
            device_memory->in_row_ptr, device_memory->in_col_ind, device_memory->row_ptr, graph->vertex_count, 
            device_memory->residual, phase_id,  gTolerance);
            GPUProfiler::EndTimer(EXPAND_TIME);

            std::swap(device_memory->vertex_ft, device_memory->vertex_ft2);
            std::swap(device_memory->vertex_ft_cnt, device_memory->vertex_ft_cnt2);
            ++iteration_id;
        }
    }

};

class PPRRevPushGPUVanilla : public PPRRevPushGPU{
public:
    PPRRevPushGPUVanilla(GraphVec *graph) : PPRRevPushGPU(graph){}
    ~PPRRevPushGPUVanilla(){}

    virtual void ExecuteMainLoop(const size_t phase_id = 0){
        ExecuteVanilla(phase_id);
    }

    void ExecuteVanilla(const size_t phase_id = 0){
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
#if defined(VALIDATE)
            std::cout << "phase_id=" << phase_id << ",iteration_id=" << iteration_id << ",frontier_count=" << vertex_frontier_count << std::endl;
#endif
            GPUProfiler::StartTimer(INSPECT_TIME);
            blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, vertex_frontier_count);
            InspectExtra<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, 
            device_memory->residual, device_memory->pagerank, phase_id);
            GPUProfiler::EndTimer(INSPECT_TIME);

            GPUProfiler::StartTimer(EXPAND_TIME);
            CUDA_ERROR(cudaMemset(device_memory->vertex_ft_cnt2, 0, sizeof(IndexType)));
            blocks_num = CALC_BLOCKS_NUM(THREADS_PER_BLOCK, vertex_frontier_count);
            ExpandVanillaRev<IndexType, ValueType><<<blocks_num, THREADS_PER_BLOCK>>>(device_memory->vertex_ft, device_memory->vertex_ft_r, device_memory->vertex_ft_cnt, device_memory->vertex_ft2, device_memory->vertex_ft_cnt2,
            device_memory->in_row_ptr, device_memory->in_col_ind, device_memory->row_ptr, graph->vertex_count, 
            device_memory->residual, device_memory->status, iteration_id, phase_id, gTolerance);
            GPUProfiler::EndTimer(EXPAND_TIME);

            std::swap(device_memory->vertex_ft, device_memory->vertex_ft2);
            std::swap(device_memory->vertex_ft_cnt, device_memory->vertex_ft_cnt2);
            ++iteration_id;
        }
    }
       
};

#endif

