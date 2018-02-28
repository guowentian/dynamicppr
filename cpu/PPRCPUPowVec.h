#ifndef __PPR_CPU_POW_VEC_H__
#define __PPR_CPU_POW_VEC_H__

#include "GraphVec.h"
#include <cstdlib>
#include <cmath>

// for validation
class PPRCPUPowVec{
public:
	PPRCPUPowVec(GraphVec *g) : graph(g){
		vertex_count = g->vertex_count;

        pagerank = NULL;
        for (int i = 0; i < 2; ++i){
        	pr[i] = new double[vertex_count];
        }
	}
	~PPRCPUPowVec(){
        pagerank = NULL;
	    for (int i = 0; i < 2; ++i){
            delete[] pr[i];
            pr[i] = NULL;
        }
    }
    void CalPPRFwd(IndexType svid, ValueType alpha = ALPHA){
        std::vector<std::vector<IndexType> > &in_col_ind = graph->in_col_ind;
        std::vector<std::vector<IndexType> > &col_ind = graph->col_ind;
        for (IndexType u = 0; u < vertex_count; ++u){
            pr[0][u] = (svid == u) ? 1 : 0;
        }
        std::cout << "start power iterative computation" << std::endl;
        size_t iteration_count = 0;
        size_t id = 0;
        while (1){
            bool stop = true;
            size_t oid = 1 - id;
            for (IndexType u = 0; u < vertex_count; ++u){
                pr[oid][u] = 0.0;
                for (size_t j = 0; j < in_col_ind[u].size(); ++j){
                    IndexType v = in_col_ind[u][j];
                    pr[oid][u] += pr[id][v] / (col_ind[v].size() + 1);
                }
                pr[oid][u] = (1.0 - alpha) * pr[oid][u];
                if (u == svid) pr[oid][u] += alpha * 1.0;
                if (std::fabs(pr[oid][u] - pr[id][u]) > 1e-14) stop = false;
            }
            if (stop) break;
            id = (id + 1) % 2;
            ++iteration_count;
        }
        pagerank = pr[id];
        std::cout << "finish power iterative, #iteration=" << iteration_count << std::endl;
    }
    void CalPPRRev(IndexType source_vertex_id, ValueType alpha = ALPHA){
        std::vector<std::vector<IndexType> > &col_ind = graph->col_ind;
        for (IndexType u = 0; u < vertex_count; ++u){
            pr[0][u] = (source_vertex_id == u) ? 1 : 0;
        }
        std::cout << "start power iterative computation" << std::endl;
        size_t iteration_count = 0;
        size_t id = 0;
        while (1){
            bool stop = true;
            size_t oid = 1 - id;
            for (IndexType u = 0; u < vertex_count; ++u){
                pr[oid][u] = 0.0;
                for (size_t j = 0; j < col_ind[u].size(); ++j){
                    IndexType v = col_ind[u][j];
                    pr[oid][u] += pr[id][v] / (col_ind[u].size() + 1);
                }
                
                pr[oid][u] = (1.0 - alpha) * pr[oid][u];
                if (u == source_vertex_id) pr[oid][u] += alpha * 1.0;
                if (std::fabs(pr[oid][u] - pr[id][u]) > 1e-14) stop = false;
            }
            if (stop) break;
            id = (id + 1) % 2;
            ++iteration_count;
        }
        pagerank = pr[id];
        std::cout << "finish power iterative, #iteration=" << iteration_count << std::endl;
    }
    double *GetPRVector(){
        return pagerank;
    }
public:
	GraphVec *graph;
	IndexType vertex_count;

    double *pagerank; // final answer
	double *pr[2];

};



#endif
