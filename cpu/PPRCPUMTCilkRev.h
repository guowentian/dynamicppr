#ifndef __PPR_CPU_MT_CILK_REV_H__
#define __PPR_CPU_MT_CILK_REV_H__

#include "PPRCPUMTCilk.h"

class PPRCPUMTCilkRev : public PPRCPUMTCilk{
public:
    PPRCPUMTCilkRev(GraphVec *g) : PPRCPUMTCilk(g){
        edge_ind = new IndexType[edge_count+1];
        edge_flag = new bool[edge_count+1];
        vertex_offset = new IndexType[vertex_count+1];
        status = new IndexType[vertex_count + 1];
        memset(status, -1, sizeof(IndexType) * (vertex_count + 1));

        global_ft = new IndexType[vertex_count];
        global_ft2 = new IndexType[vertex_count];
        global_ft_r = new ValueType[vertex_count];
    }
    ~PPRCPUMTCilkRev(){
        delete[] edge_ind;
        edge_ind = NULL;
        delete[] edge_flag;
        edge_flag = NULL;
        delete[] vertex_offset;
        vertex_offset = NULL;
        delete[] status;
        status = NULL;

        delete[] global_ft;
        global_ft = NULL;
        delete[] global_ft2;
        global_ft2 = NULL;
        delete[] global_ft_r;
        global_ft_r = NULL;
    }
 
     
    virtual void ExecuteImpl(){
        Init();
        MainLoop(0);
    }

    virtual void IncExecuteImpl(){
		SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec*>(graph);
#if defined(PROFILE)
        Profiler::StartTimer(INC_UPDATE_TIME);
#endif
        EdgeBatch *edge_batch = dg->edge_batch;
        
        // get pre degree, which is helpful to update app data
        CopyOutDegree(edge_batch->edge1, edge_batch->edge2, edge_batch->length, predeg);
        RevertOutDegree(edge_batch->edge1, edge_batch->edge2, edge_batch->is_insert, edge_batch->length, predeg);

        // update stream app data
        StreamUpdateAppData(edge_batch->edge1, edge_batch->edge2, edge_batch->is_insert, edge_batch->length);
#if defined(PROFILE)      
        Profiler::EndTimer(INC_UPDATE_TIME);
        Profiler::StartTimer(PUSH_TIME);
#endif
        // phase_id = 0
        ++iteration_id;
        DynPushInit(0, edge_batch->edge1, edge_batch->edge2, edge_batch->length, edge_ind, edge_flag, predeg);
        MainLoop(0);

        // phase_id = 1
        ++iteration_id;
        DynPushInit(1, edge_batch->edge1, edge_batch->edge2, edge_batch->length, edge_ind, edge_flag, predeg);
        MainLoop(1);
#if defined(PROFILE)
        Profiler::EndTimer(PUSH_TIME);
#endif

    }
	
    inline bool IsLegalPush(ValueType r, size_t phase_id){
        if ((phase_id == 0 && r > gTolerance) || (phase_id == 1 && r < -gTolerance)){
            return true;
        }
        return false;
    }

    inline ValueType AtomicAddResidual(IndexType u, ValueType add){
        if (sizeof(ValueType) == 8){
            volatile ValueType old_val, new_val;
            do{
                old_val = residual[u];
                new_val = old_val + add;
            }while (!__sync_bool_compare_and_swap((long long*)(residual + u), *((long long*)&old_val), *((long long*)&new_val)));
            return old_val;
        }
        else{
            std::cout << "CAS bad length" << std::endl;
            exit(-1);
        }
    }
    inline bool AtomicUpdateStatus(IndexType u, IndexType iter_id){
        // for global duplicate detect
        bool ret = false;
        locks[u].Lock();
        if (status[u] < iteration_id){
            status[u] = iteration_id;
            ret = true;
        }
        locks[u].Unlock();
        return ret;
    }

    virtual void StreamUpdateAppData(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType edge_batch_length){
        parallel_for(IndexType i = 0; i < edge_batch_length; ++i){
            IndexType u = edge_batch1[i];
			IndexType v = edge_batch2[i];
			locks[u].Lock();
            ValueType add = (1.0 - ALPHA) * pagerank[v] - pagerank[u] - ALPHA * residual[u] + ALPHA * (source_vertex_id == u ? 1.0 : 0.0);
		    if (is_insert[i]){
                predeg[u]++;
                residual[u] += add / (predeg[u] + 1) / ALPHA;
            }
            else{
                predeg[u]--;
                residual[u] -= add / (predeg[u] + 1) / ALPHA;
            }
			locks[u].Unlock();
        }
    }

    virtual void DynPushInit(const size_t phase_id, IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length, IndexType *vertex_ind, bool *vertex_tag, IndexType *vertex_map) {
        IndexType *vertex_set[2] = { edge_batch1, edge_batch2 };
        // init
        parallel_for (IndexType i = 0; i < 2; ++i){
            parallel_for (IndexType j = 0; j < edge_batch_length; ++j){
                IndexType u = vertex_set[i][j];
                vertex_map[u] = vertex_count;
            }
        }
        parallel_for (IndexType i = 0; i < 2; ++i){
            parallel_for (IndexType j = 0; j < edge_batch_length; ++j){
                IndexType u = vertex_set[i][j];
                IndexType off = i * edge_batch_length + j;
                if (IsLegalPush(residual[u], phase_id) && vertex_map[u] == vertex_count){
                    bool success = __sync_bool_compare_and_swap(vertex_map + u, vertex_count, off);
                    if (success){
                        vertex_tag[off] = true;
                        vertex_ind[off] = u;
                    }
                    else{
                        vertex_tag[off] = false;
                    }
                }
                else{
                    vertex_tag[off] = false;
                }
            }
        }
        
        IndexType frontier_count = sequence::pack<IndexType, IndexType>(vertex_ind, global_ft, vertex_tag, 2 * edge_batch_length);
        global_ft_count = frontier_count;
#if defined(VALIDATE)
        memset(vertex_map, -1, sizeof(IndexType) * vertex_count);
        for (IndexType i = 0; i < global_ft_count; ++i){
            IndexType u = global_ft[i];
            assert(IsLegalPush(residual[u], phase_id));
            assert(vertex_map[u] == -1);
            vertex_map[u] = u;
        }
        for (IndexType i = 0; i < 2; ++i){
            for (IndexType j = 0; j < edge_batch_length; ++j){
                IndexType u = vertex_set[i][j];
                if (IsLegalPush(residual[u], phase_id))
                assert(vertex_map[u] >= 0);
            }
        }
#endif
    }

    void Init(){
		parallel_for (IndexType u = 0; u < vertex_count; ++u){
            pagerank[u] = 0.0;
            residual[u] = (source_vertex_id == u) ? 1.0 : 0.0;
        }
        global_ft[0] = source_vertex_id;
        global_ft_count = 1;
	}

    virtual void MainLoop(size_t phase_id = 0){
        std::vector<std::vector<IndexType> > &in_col_ind = graph->in_col_ind;
        std::vector<IndexType> &deg = graph->deg;
#if defined(PROFILE)
        long long traverse_time = 0;
        TimeMeasurer timer;
#endif
        global_ft_count2 = 0;
        while (1){
            IndexType vertex_frontier_count = global_ft_count;
            if (vertex_frontier_count == 0) break;
#if defined(PROFILE)
            std::cout << "iteration_id=" << iteration_id << ",frontier=" << vertex_frontier_count << std::endl;
#endif
            parallel_for (IndexType i = 0; i < vertex_frontier_count; ++i){
                IndexType u = global_ft[i];
                vertex_offset[i] = in_col_ind[u].size(); 
            }
            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(vertex_offset, vertex_offset, vertex_frontier_count);
#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif

            parallel_for (IndexType i = 0; i < vertex_frontier_count; ++i){
                IndexType u = global_ft[i];
                ValueType ru = residual[u];
                global_ft_r[i] = ru;
                pagerank[u] += ALPHA * ru;
                IndexType indegu = in_col_ind[u].size();

                if (indegu < VERTEX_DEGREE_THRESHOLD){
                    for (IndexType j = 0; j < indegu; ++j){
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA)  * ru / (deg[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(prer, phase_id) == false && IsLegalPush(curr, phase_id) == true){
                            is_frontier = true;
                        }
                        if (is_frontier){
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else{
                            edge_flag[off] = false;
                        }
                    }
                }
                else{
                    parallel_for (IndexType j = 0; j < indegu; ++j){
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(prer, phase_id) == false && IsLegalPush(curr, phase_id) == true){
                            is_frontier = true;
                        }
                        if (is_frontier){
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else{
                            edge_flag[off] = false;
                        }
                    }
                }
            }
#if defined(PROFILE)
            Profiler::EndTimer(EXPAND_TIME);
            timer.EndTimer();
            traverse_time += timer.GetElapsedMicroSeconds();
#endif

            IndexType new_frontier_count1 = sequence::pack<IndexType, IndexType>(edge_ind, global_ft2, edge_flag, total_granual);
            IndexType *vertex_ind = edge_ind;
            bool *vertex_flag = edge_flag;
            parallel_for (IndexType i = 0; i < vertex_frontier_count; ++i){
                IndexType u = global_ft[i];
                residual[u] -= global_ft_r[i];
                if (IsLegalPush(residual[u], phase_id)){
                    vertex_flag[i] = true;
                    vertex_ind[i] = u;
                }
                else{
                    vertex_flag[i] = false;
                }
            }
            
            IndexType new_frontier_count2 = sequence::pack<IndexType, IndexType>(vertex_ind, global_ft2 + new_frontier_count1, vertex_flag, vertex_frontier_count);
            global_ft_count2 = new_frontier_count1 + new_frontier_count2;
            
            std::swap(global_ft, global_ft2);
            std::swap(global_ft_count, global_ft_count2);
            ++iteration_id;
        }
#if defined(PROFILE)
        std::cout << "traverse_time=" << traverse_time / 1000.0 << "ms" << std::endl;
#endif
    }

	virtual void Validate(){
        graph->ConstructGraph();
        for (IndexType u = 0; u < vertex_count; ++u){
            assert(residual[u] < gTolerance && residual[u] > -gTolerance);
        }
        PPRCPUPowVec *ppr_pow = new PPRCPUPowVec(graph);
		ppr_pow->CalPPRRev(source_vertex_id);
		double *ans = ppr_pow->pagerank;
		const double bound = gTolerance * 100.0;
        for (IndexType u = 0; u < vertex_count; ++u){
            ValueType err = ans[u] - pagerank[u];
            if (err < 0) err = -err;
            assert(err < bound);
		}
		delete ppr_pow;
		ppr_pow = NULL;
    }
public:
    // to construct frontier queue
    IndexType *edge_ind;
    bool *edge_flag;
    IndexType *vertex_offset;
// for non-DupDetect
    IndexType *status;

    IndexType *global_ft;
    IndexType global_ft_count;
    IndexType *global_ft2;
    IndexType global_ft_count2;
    ValueType *global_ft_r;

};
#endif

