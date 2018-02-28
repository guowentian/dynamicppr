#ifndef __PPR_CPU_H__
#define __PPR_CPU_H__

#include "GraphVec.h"
#include "TimeMeasurer.h"
#include "Profiler.h"
#include "SlidingGraphVec.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <queue>
#include <map>

// deprecated: single-threaded implementation
class PPRCPU{
public:
	PPRCPU(GraphVec *g) : graph(g){
		vertex_count = g->vertex_count;
        pagerank = new ValueType[vertex_count + 1];
        residual = new ValueType[vertex_count + 1];
        predeg = new IndexType[vertex_count + 1];
        source_vertex_id = gSourceVertexId;
        ppr_time = 0;
        std::cout << "choose " << source_vertex_id << " as source vertex" << std::endl;
	}
	~PPRCPU(){
	    delete[] pagerank;
        pagerank = NULL;
        delete[] residual;
        residual = NULL;
        delete[] predeg;
        predeg = NULL;
    }

	virtual void Execute(const size_t queue_type = FIFO){
        std::cout << "start.." << std::endl;
        // static graph execution

		TimeMeasurer timer;
		timer.StartTimer();
        Profiler::StartTimer(TOTAL_TIME);

        ExecuteImpl();

        Profiler::EndTimer(TOTAL_TIME);
		timer.EndTimer();
		std::cout << "finish, elapsed time=" << timer.GetElapsedMicroSeconds() * 1.0 / 1000.0 << "ms" << std::endl;
#if defined(VALIDATE)
		Validate();
#endif
	}

	virtual void DynamicExecute(const size_t queue_type = FIFO){
        std::cout << "start..." << std::endl;

        Profiler::StartTimer(TOTAL_TIME);
        
        Profiler::StartTimer(INIT_GRAPH_CALC_TIME);
        ExecuteImpl();
        Profiler::EndTimer(INIT_GRAPH_CALC_TIME);
#if defined(VALIDATE)
            Validate();
#endif

        Profiler::StartTimer(DYNA_GRAPH_CALC_TIME);
        DynamicMainLoop();
        Profiler::EndTimer(DYNA_GRAPH_CALC_TIME);

        Profiler::EndTimer(TOTAL_TIME);
#if defined(VALIDATE)
        Validate();
#endif
	}

    virtual void DynamicMainLoop(){
		SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec*>(graph);
        // on streaming updates
		size_t stream_batch_count = 0;
        while (stream_batch_count++ < gStreamBatchCount){
            Profiler::StartTimer(EXCLUDE_GRAPH_UPDATE_TIME);
#if defined(PROFILE)
            TimeMeasurer exclude_timer;
            exclude_timer.StartTimer();
#endif
            if (gStreamUpdateCountPerBatch >= 100 || stream_batch_count % 1000 == 0){ 
                std::cout << "stream_batch_count=" << stream_batch_count << std::endl;
                double cur_ppr_time = ppr_time / 1000.0;
                long long cur_edge_count = gStreamUpdateCountPerBatch;
                cur_edge_count *= (stream_batch_count-1);
                std::cout << "ppr_time " << cur_ppr_time << " ms" << std::endl;
                std::cout << "edge_count "<< cur_edge_count << std::endl;
                std::cout << "ppr_latency " << ((stream_batch_count-1 > 0) ? cur_ppr_time / (stream_batch_count-1) : 0) << " ms" << std::endl;
                std::cout << "ppr_throughput " << cur_edge_count / cur_ppr_time * 1000.0 << " edge/s" << std::endl;
            }
            bool is_edge_stream_over = dg->StreamUpdates(gStreamUpdateCountPerBatch);
            if (is_edge_stream_over)
            	break;
           
            // update graph structure
            dg->IncConstructWindowGraph();
            //dg->ConstructGraph();

#if defined(PROFILE)
            exclude_timer.EndTimer();
            //std::cout << "exluce graph update time=" << exclude_timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
#endif
            Profiler::EndTimer(EXCLUDE_GRAPH_UPDATE_TIME);
    
            Profiler::StartTimer(PPR_TIME);
            
            IncExecuteImpl();

            Profiler::EndTimer(PPR_TIME);
#if defined(VALIDATE)
            Validate();
#endif
        }
	}

    void RevertOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType edge_batch_length){
		for (IndexType i = 0; i < edge_batch_length; ++i){
			IndexType u = edge_batch1[i];
			if (is_insert[i]){
				predeg[u]--;
			}
			else{
				predeg[u]++;
			}
		}
	}
	void CopyOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length, IndexType *deg){
        std::vector<IndexType> &gdeg = graph->deg;
        for (IndexType i = 0; i < edge_batch_length; ++i){
            IndexType u = edge_batch1[i];
            IndexType v = edge_batch2[i];
            deg[u] = gdeg[u];
            deg[v] = gdeg[v];
        }
	}

    virtual void IncExecuteImpl() = 0;
    
    virtual void ExecuteImpl() = 0;

	virtual void Validate() = 0;
	
public:
	GraphVec *graph;
	IndexType vertex_count;
// app data
    ValueType *pagerank;
	ValueType *residual;
// dynamic graph
    IndexType *predeg;
// app parameter
	IndexType source_vertex_id;

    double ppr_time;

};

#endif
