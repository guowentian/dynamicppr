#ifndef __PPR_CPU_MT_CILK_H__
#define __PPR_CPU_MT_CILK_H__

#include "GraphVec.h"
#include "SlidingGraphVec.h"
#include "TimeMeasurer.h"
#include "Profiler.h"
#include "Barrier.h"
#include "SpinLock.h"
#include "CilkUtil.h"
#include "PPRCPUPowVec.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <queue>
#include <map>
#include <atomic>
#include <algorithm>
#if defined(PAPI_PROFILE)
#include "PapiProfiler.h"
#endif

class PPRCPUMTCilk{
public:
	PPRCPUMTCilk(GraphVec *g) : graph(g){
		vertex_count = g->vertex_count;
        edge_count = g->edge_count;

        pagerank = new ValueType[vertex_count + 1];
        residual = new ValueType[vertex_count + 1];
        iteration_id = 0;
        
        source_vertex_id = gSourceVertexId;

        predeg = new IndexType[vertex_count];

		thread_count = gThreadNum;
        SetWorkers(thread_count);

        locks = new SpinLock[vertex_count];
        for (IndexType i = 0; i < vertex_count; ++i) locks[i].Init();
        is_over = false;

        ppr_time = 0;
    }
	~PPRCPUMTCilk(){
        delete[] pagerank;
        pagerank = NULL;
        delete[] residual;
        residual = NULL;
        delete[] predeg;
        predeg = NULL;
        delete[] locks;
        locks = NULL;
    }
    
	virtual void Execute(){
		std::cout << "start processing..." << std::endl;
        Profiler::StartTimer(TOTAL_TIME);
        TimeMeasurer timer;
        timer.StartTimer();
        
        ExecuteImpl();

		timer.EndTimer();
        Profiler::EndTimer(TOTAL_TIME);
        std::cout << std::endl << "elapsed time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
#if defined(VALIDATE)
        Validate();
#endif
	}

	virtual void DynamicExecute(){
		std::cout << "start processing..." << std::endl;
#if defined(PAPI_PROFILE)
        PapiProfiler::InitPapiProfiler();
#endif
        Profiler::StartTimer(TOTAL_TIME);
		
        Profiler::StartTimer(INIT_GRAPH_CALC_TIME);
        TimeMeasurer timer;
        timer.StartTimer();
        ExecuteImpl();
		timer.EndTimer();
        std::cout << std::endl << "elapsed time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
        Profiler::EndTimer(INIT_GRAPH_CALC_TIME);
#if defined(VALIDATE)
        Validate();
#endif
        Profiler::StartTimer(DYNA_GRAPH_CALC_TIME);
        DynamicMainLoop();
        Profiler::EndTimer(DYNA_GRAPH_CALC_TIME);
        
        Profiler::EndTimer(TOTAL_TIME);
#if defined(PAPI_PROFILE)
        PapiProfiler::ReportProfile();
#endif
    }
	
    virtual void DynamicMainLoop(){
		SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec*>(graph);
        // on streaming updates
		size_t stream_batch_count = 0;
        while (stream_batch_count++ < gStreamBatchCount){
            Profiler::StartTimer(EXCLUDE_GRAPH_UPDATE_TIME);
            // report throughput from time to time
            if (gStreamUpdateCountPerBatch >= 100 || stream_batch_count % 1000 == 0){ 
                std::cout << "stream_batch_count=" << stream_batch_count << std::endl;
                double cur_ppr_time = ppr_time / 1000.0;
                long long cur_edge_count = gStreamUpdateCountPerBatch;
                cur_edge_count *= (stream_batch_count-1);
                std::cout << "ppr_time " << cur_ppr_time << " ms" << std::endl;
                std::cout << "edge_count "<< cur_edge_count << std::endl;
                std::cout << "ppr_latency " << ((stream_batch_count - 1 > 0) ? cur_ppr_time / (stream_batch_count-1) : 0) << " ms" << std::endl;
                std::cout << "ppr_throughput " << cur_edge_count / cur_ppr_time * 1000.0 << " edge/s" << std::endl;
            }

            bool is_edge_stream_over = dg->StreamUpdates(gStreamUpdateCountPerBatch);
            if (is_edge_stream_over)
            	break;
            
            // update graph structure
            dg->IncConstructWindowGraph();
            //dg->ConstructGraph();

            Profiler::EndTimer(EXCLUDE_GRAPH_UPDATE_TIME);
#if defined(PAPI_PROFILE)
            PapiProfiler::BeginProfile();
#endif
            Profiler::StartTimer(PPR_TIME);
            TimeMeasurer timer;
            timer.StartTimer();
            IncExecuteImpl();
            timer.EndTimer();
            ppr_time += timer.GetElapsedMicroSeconds();
            Profiler::EndTimer(PPR_TIME);
#if defined(PAPI_PROFILE)
            PapiProfiler::EndProfile();
#endif

#if defined(VALIDATE)
            Validate();
#endif
        }
        std::cout << "stream_batch_count=" << stream_batch_count << std::endl;
        double cur_ppr_time = ppr_time / 1000.0;
        long long cur_edge_count = gStreamUpdateCountPerBatch;
        cur_edge_count *= (stream_batch_count-1);
        std::cout << "ppr_time " << cur_ppr_time << " ms" << std::endl;
        std::cout << "edge_count "<< cur_edge_count << std::endl;
        std::cout << "ppr_latency " << ((stream_batch_count - 1 > 0) ? cur_ppr_time / (stream_batch_count-1) : 0) << " ms" << std::endl;
        std::cout << "ppr_throughput " << cur_edge_count / cur_ppr_time * 1000.0 << " edge/s" << std::endl;
	}

	
	void RevertOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType edge_batch_length, IndexType *deg){
		parallel_for (IndexType i = 0; i < edge_batch_length; ++i){
            IndexType u = edge_batch1[i];
            locks[u].Lock();
            if (is_insert[i]) deg[u]--;
            else deg[u]++;
            locks[u].Unlock();
        }
	}
	void CopyOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length, IndexType *predeg){
        std::vector<IndexType> &deg = graph->deg;
        parallel_for (IndexType i = 0; i < edge_batch_length; ++i){
            IndexType u = edge_batch1[i];
            IndexType v = edge_batch2[i];
            predeg[u] = deg[u];
            predeg[v] = deg[v];
        }
	}

    virtual void ExecuteImpl() = 0;

    virtual void IncExecuteImpl() = 0;
	
    virtual void Validate() = 0;

public:
	GraphVec *graph;
	IndexType vertex_count;
    IndexType edge_count;
// app data
    ValueType *pagerank;
	ValueType *residual;
    IndexType iteration_id;
// app parameter
	IndexType source_vertex_id;
	// dynamic graph
    IndexType *predeg;

    // multi thread
    size_t thread_count;
    volatile bool is_over;
    SpinLock *locks;
    
    // measurement
    double ppr_time;
};


#endif
