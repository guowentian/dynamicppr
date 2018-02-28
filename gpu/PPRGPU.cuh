#ifndef __PPR_GPU_H__
#define __PPR_GPU_H__

#include "GraphVec.h"
#include "SlidingGraphVec.h"
#include "GPUUtil.cuh"
#include "GPUProfiler.cuh"
#include "PPRCommon.cuh"
#include "StreamUpdate.cuh"
#include "Inspect.cuh"
#include "ExpandRev.cuh"
#include "Meta.h"
#include <set>
#include <algorithm>
#include "SlidingGraphBuilder.cuh"
#include "PPRCPUPowVec.h"
#include "DeviceMemory.cuh"
#if defined(NVPROFILE)
#include <cuda_profiler_api.h>
#endif
class PPRGPU{
public:
	PPRGPU(GraphVec *g) : graph(g){
        source_vertex_id = gSourceVertexId;
        std::cout << "choose " << source_vertex_id << " as source vertex id" << std::endl;

		device_memory = new DeviceMemory(graph->vertex_count, graph->edge_count);
        device_memory->CudaAllocAppData();

        std::cout << "init sliding graph.." << std::endl;
        SlidingGraphVec *sliding_graph = (SlidingGraphVec*)g;
        sliding_graph_builder = new SlidingGraphBuilder(sliding_graph->vertex_count, sliding_graph->sliding_window_size, g->directed);
       
    }
    
	~PPRGPU(){
        device_memory->FreeGPUMemory();
        delete device_memory;
        device_memory = NULL;
    }
   
    // execution on static graph 
    virtual void Execute(){
    	std::cout << "start Execute..." << std::endl;

		GPUTimer timer;
		timer.StartTimer();
        GPUProfiler::StartTimer(TOTAL_TIME);
		// init
		Init<IndexType, ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(device_memory->pagerank, device_memory->residual, device_memory->status, graph->vertex_count, source_vertex_id);
		iteration_id = 0;
        // local push procedure, only positive residual in static graphs 
        ExecuteMainLoop(0);

        GPUProfiler::EndTimer(TOTAL_TIME);
		timer.EndTimer();
		std::cout << "elapsed time=" << timer.GetElapsedMilliSeconds() * 1.0 << "ms" << std::endl;
#if defined(VALIDATE)
        ValidateResult();
#endif
    }
   
    // execution on dynamic graphs
    virtual void DynamicExecute(){
        // prepare application-related data structure for dynamic graph
        device_memory->InitForDynamicGraph();
        EdgeBatch *init_stream = new EdgeBatch(graph->edge_count);
        // initialize streaming graph in device
        SlidingGraphVec *sliding_graph = (SlidingGraphVec*)graph;
        sliding_graph->SerializeEdgeStream(init_stream);
        sliding_graph_builder->InitWindowStream(init_stream);
        delete init_stream;
        init_stream = NULL;
        // time measure
        ppr_time = 0;

        std::cout << "start..." << std::endl;
        GPUProfiler::StartTimer(TOTAL_TIME);
       
        GPUProfiler::StartTimer(INIT_GRAPH_CALC_TIME);
#if defined(NVPROFILE)
        cudaProfilerStart();
#endif
        // init
		Init<IndexType, ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(device_memory->pagerank, device_memory->residual, device_memory->status, graph->vertex_count, source_vertex_id);
        iteration_id = 0;
       
        // first execute to init the estimate and residual vector
		ExecuteMainLoop(0);
#if defined(VALIDATE)
        ValidateResult();
#endif
#if defined(NVPROFILE)
        cudaProfilerStop();
#endif
        GPUProfiler::EndTimer(INIT_GRAPH_CALC_TIME);
        
        GPUProfiler::StartTimer(DYNA_GRAPH_CALC_TIME);
        SlidingWindowExecuteMainLoop();
        GPUProfiler::EndTimer(DYNA_GRAPH_CALC_TIME);
        
        GPUProfiler::EndTimer(TOTAL_TIME);
        
        std::cout << "finish!" << std::endl;
#if defined(VALIDATE)
        ValidateResult();
#endif
    }
    virtual void SlidingWindowExecuteMainLoop(){
        SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec*>(graph);
        size_t stream_batch_count = 0;
        while (stream_batch_count++ < gStreamBatchCount){
            // ======= sliding graph update time in the device is excluded from the elapsed time of PPR update time
            GPUProfiler::StartTimer(EXCLUDE_GRAPH_UPDATE_TIME);
            // report the throughput at some frequency
            if (gStreamUpdateCountPerBatch > 100 || stream_batch_count % 100 == 0){
                std::cout << "coming stream_batch_count=" << stream_batch_count << std::endl;
                long long cur_edge_num = gStreamUpdateCountPerBatch;
                cur_edge_num *= (stream_batch_count - 1);
                std::cout << "ppr_time " << ppr_time << std::endl;
                std::cout << "edge_num " << cur_edge_num << std::endl;
                std::cout << "ppr_latency " << ((stream_batch_count - 1 > 0) ? ppr_time / (stream_batch_count-1) : 0) << std::endl; 
                std::cout << "ppr_throughput " << cur_edge_num / ppr_time * 1000.0 << std::endl;
            }

            bool is_edge_stream_over = dg->StreamUpdates(gStreamUpdateCountPerBatch);
            // the remaining #edges < gStreamUpdateCountPerBatch, choose not to process this batch and exit the loop directly
            if (is_edge_stream_over) 
                break;
            device_memory->edge_batch->CudaMemcpy(dg->edge_batch, cudaMemcpyHostToDevice);


            // update the sliding graph in the device
            GPUBuildSlidingGraph(); 
            GPUProfiler::EndTimer(EXCLUDE_GRAPH_UPDATE_TIME);

            // ======= start updating PPR vector 
            GPUTimer timer;
            timer.StartTimer();
            GPUProfiler::StartTimer(PPR_TIME);
            ++iteration_id;

            GPUProfiler::StartTimer(INC_UPDATE_TIME);
#if defined(NVPROFILE)
            cudaProfilerStart();
#endif
            IncrementalBatchUpdate();
#if defined(NVPROFILE)
            cudaProfilerStop();
#endif
            GPUProfiler::EndTimer(INC_UPDATE_TIME);
            GPUProfiler::StartTimer(PUSH_TIME);
#if defined(NVPROFILE)
            cudaProfilerStart();
#endif
            ExecuteMainLoop(0);
            ExecuteMainLoop(1);
#if defined(NVPROFILE)
            cudaProfilerStop();
#endif
            GPUProfiler::EndTimer(PUSH_TIME);
            GPUProfiler::EndTimer(PPR_TIME);
            timer.EndTimer();
            ppr_time += timer.GetElapsedMilliSeconds();
#if defined(VALIDATE) 
            ValidateResult();
#endif
            
        }
        std::cout << "coming stream_batch_count=" << stream_batch_count << std::endl;
        long long cur_edge_num = gStreamUpdateCountPerBatch;
        cur_edge_num *= (stream_batch_count - 1);
        std::cout << "ppr_time " << ppr_time << std::endl;
        std::cout << "edge_num " << cur_edge_num << std::endl;
        std::cout << "ppr_latency " << ((stream_batch_count - 1 > 0) ? ppr_time / (stream_batch_count-1) : 0) << std::endl; 
        std::cout << "ppr_throughput " << cur_edge_num / ppr_time * 1000.0 << std::endl;
    }
    
    virtual void IncrementalBatchUpdate() = 0;
    virtual void ExecuteMainLoop(const size_t phase_id = 0) = 0;
    virtual void ValidateResult() = 0;
    virtual void GPUBuildSlidingGraph() = 0;

public:
	// ===========allocated in GPU memory
    DeviceMemory *device_memory;
    SlidingGraphBuilder* sliding_graph_builder;
    
    // ===========cpu memory
	GraphVec *graph;

    // application parameter
	IndexType source_vertex_id;
    // global control
    int iteration_id;
    // measure time
    float ppr_time;
};

#endif
