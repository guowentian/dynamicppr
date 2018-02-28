#ifndef __PPR_CPU_MT_CILK_REV_VARIANTS_H__
#define __PPR_CPU_MT_CILK_REV_VARIANTS_H__

#include "PPRCPUMTCilkRev.h"

class PPRCPUMTCilkRevVanilla : public PPRCPUMTCilkRev{
    public: 
        PPRCPUMTCilkRevVanilla(GraphVec *g) : PPRCPUMTCilkRev(g){

        }
    void MainLoop(size_t phase_id = 0){
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
                global_ft_r[i] = residual[u];
                pagerank[u] += ALPHA * residual[u];
                residual[u] = 0.0;
                vertex_offset[i] = in_col_ind[u].size(); 
            }
            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(vertex_offset, vertex_offset, vertex_frontier_count);
#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif
            parallel_for (IndexType i = 0; i < vertex_frontier_count; ++i){
                IndexType u = global_ft[i];
                ValueType ru = global_ft_r[i];
                IndexType indegu = in_col_ind[u].size();
                
                if (indegu < VERTEX_DEGREE_THRESHOLD){
                    for (IndexType j = 0; j < indegu; ++j){
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;
                        if (IsLegalPush(curr, phase_id)){
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp){
                                is_frontier = true;
                            }
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
                        if (IsLegalPush(curr, phase_id)){
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp){
                                is_frontier = true;
                            }
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
            // compact for next frontier
            global_ft_count2 = sequence::pack<IndexType, IndexType>(edge_ind, global_ft2, edge_flag, total_granual);
            
            std::swap(global_ft, global_ft2);
            std::swap(global_ft_count, global_ft_count2);
            ++iteration_id;
        }
#if defined(PROFILE)
        std::cout << "traverse_time=" << traverse_time / 1000.0 << "ms" << std::endl;
#endif
    }
        
};

class PPRCPUMTCilkRevEager : public PPRCPUMTCilkRev{
public: 
    PPRCPUMTCilkRevEager(GraphVec *g) : PPRCPUMTCilkRev(g){

    }
    void MainLoop(size_t phase_id = 0){
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
                status[u] = iteration_id; // avoid put to frontier in neighbor propagation
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

                        if (IsLegalPush(curr, phase_id)){
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp){
                                is_frontier = true;
                            }
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

                        if (IsLegalPush(curr, phase_id)){
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp){
                                is_frontier = true;
                            }
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

};
class PPRCPUMTCilkRevFF : public PPRCPUMTCilkRev{
public: 
    PPRCPUMTCilkRevFF(GraphVec *g) : PPRCPUMTCilkRev(g){

    }
    void MainLoop(size_t phase_id = 0){
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
                global_ft_r[i] = residual[u];
                pagerank[u] += ALPHA * residual[u];
                residual[u] = 0.0;
                vertex_offset[i] = in_col_ind[u].size(); 
            }
            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(vertex_offset, vertex_offset, vertex_frontier_count);
#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif
            parallel_for (IndexType i = 0; i < vertex_frontier_count; ++i){
                IndexType u = global_ft[i];
                ValueType ru = global_ft_r[i];
                IndexType indegu = in_col_ind[u].size();
                
                if (indegu < VERTEX_DEGREE_THRESHOLD){
                    for (IndexType j = 0; j < indegu; ++j){
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
            // compact for next frontier
            global_ft_count2 = sequence::pack<IndexType, IndexType>(edge_ind, global_ft2, edge_flag, total_granual);
            
            std::swap(global_ft, global_ft2);
            std::swap(global_ft_count, global_ft_count2);
            ++iteration_id;
        }
#if defined(PROFILE)
        std::cout << "traverse_time=" << traverse_time / 1000.0 << "ms" << std::endl;
#endif
    }
        

}; 



#endif
