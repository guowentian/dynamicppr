#ifndef __PPR_CPU_REV_H__
#define __PPR_CPU_REV_H__

#include "PPRCPU.h"
#include "PPRCPUPowVec.h"

// deprecated: single threaded implementation
class PPRCPURev : public PPRCPU{
public:
	PPRCPURev(GraphVec *g) : PPRCPU(g){
        status = new int[vertex_count];
        q = new std::queue<IndexType>();
	}
	~PPRCPURev(){
        delete[] status;
        status = NULL;
        delete q;
        q = NULL;
    }

    virtual void Init(){
		for (IndexType u = 0; u < vertex_count; ++u){
			pagerank[u] = 0.0;
			residual[u] = (source_vertex_id == u) ? 1.0 : 0.0;
        }
        memset(status, 0, sizeof(int) * vertex_count);
        status[source_vertex_id] = 1;
	    q->push(source_vertex_id);
    }

    virtual void ExecuteImpl(){
        Init();
        MainLoopFIFO(0);
    }

    virtual void IncExecuteImpl(){
        SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec*>(graph);
        Profiler::StartTimer(INC_UPDATE_TIME);
        EdgeBatch *edge_batch = dg->edge_batch;
        
        // get pre degree, which is helpful to update app data
        CopyOutDegree(edge_batch->edge1, edge_batch->edge2, edge_batch->length, predeg);
        RevertOutDegree(edge_batch->edge1, edge_batch->edge2, edge_batch->is_insert, edge_batch->length);

        // update stream app data
        for (IndexType i = 0; i < edge_batch->length; ++i){
            // direction of graph has already been considered in edge_batch
            StreamUpdateAppData(edge_batch->edge1[i], edge_batch->edge2[i], edge_batch->is_insert[i]);
        }
        Profiler::EndTimer(INC_UPDATE_TIME);

        // phase_id = 0
        DynPushInit(0, edge_batch->edge1, edge_batch->edge2, edge_batch->length);
        MainLoopFIFO(0);

        // phase_id = 1
        DynPushInit(1, edge_batch->edge1, edge_batch->edge2, edge_batch->length);
        MainLoopFIFO(1);

    }

	virtual void StreamUpdateAppData(IndexType u, IndexType v, bool is_insert){
		if (is_insert) predeg[u]++;
        else predeg[u]--;
        if (is_insert){
            ValueType add = (1.0 - ALPHA) * pagerank[v] - pagerank[u] - ALPHA * residual[u] + ALPHA * (source_vertex_id == u ? 1.0 : 0.0);
            residual[u] += add / (predeg[u] + 1) / ALPHA;
        }
        else{
            ValueType add = (1.0 - ALPHA) * pagerank[v] - pagerank[u] - ALPHA * residual[u] + ALPHA * (source_vertex_id == u ? 1.0 : 0.0);
            residual[u] -= add / (predeg[u] + 1) / ALPHA;
        }
    }

    inline bool IsLegalPush(ValueType r, size_t phase_id){
        if ((phase_id == 0 && r > gTolerance) || (phase_id == 1 && r < -gTolerance)){
            return true;
        }
        return false;
    }
	
	virtual void MainLoopFIFO(const size_t phase_id = 0){
        std::vector<std::vector<IndexType> > &in_col_ind = graph->in_col_ind;
        std::vector<IndexType> &deg = graph->deg;
        while (!q->empty()){
			IndexType u = q->front();
			q->pop();
			status[u] = 0;
#if defined(VALIDATE)
			assert(IsLegalPush(residual[u], phase_id));
			assert(u < vertex_count);
#endif
#if defined(PROFILE)
            Profiler::AggCount(EXPAND_COUNT, 1);
            Profiler::AggCount(TRAVERSE_COUNT, in_col_ind[u].size());
#endif
			for (size_t j = 0; j < in_col_ind[u].size(); ++j){
                IndexType v = in_col_ind[u][j];
                if (v >= vertex_count) continue;
                residual[v] += (1.0 - ALPHA) * residual[u] / (deg[v] + 1);
				if (IsLegalPush(residual[v], phase_id) && status[v] == 0){
                    q->push(v);
                    status[v] = 1;
                }
            }
			pagerank[u] += ALPHA * residual[u];
			residual[u] = 0.0;
		}
	}
    
    virtual void DynPushInit(const size_t phase_id, IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length){
        for (IndexType i = 0; i < edge_batch_length; ++i){
			IndexType u = edge_batch1[i];
			IndexType v = edge_batch2[i];
            if (IsLegalPush(residual[u], phase_id) && status[u] == 0){
                q->push(u);
                status[u] = 1;
            }
            if (IsLegalPush(residual[v], phase_id) && status[v] == 0){
                q->push(v);
                status[v] = 1;
            }
		}
	}

	
	virtual void Validate(){
        graph->ConstructGraph();
		for (IndexType u = 0; u < vertex_count; ++u){
            assert(residual[u] < gTolerance && residual[u] > -gTolerance);
        }
        PPRCPUPowVec *ppr_pow = new PPRCPUPowVec(graph);
		ppr_pow->CalPPRRev(source_vertex_id);
		double *ans = ppr_pow->pagerank;
		const double bound = gTolerance;
        for (IndexType u = 0; u < vertex_count; ++u){
            ValueType err = ans[u] - pagerank[u];
            if (err < 0) err = -err;
            assert(err < bound);
		}
		delete ppr_pow;
		ppr_pow = NULL;
	}
public:
	int *status;
	std::queue<IndexType> *q;
};

#endif
