#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "Meta.h"
#include "TimeMeasurer.h"
#include <string>
#include <cstring>
#include <cassert>
#include <fstream>
#include <map>
#include <algorithm>
#include <vector>
#include <cstdio>

class Graph{
public:

	Graph(const std::string &filename, bool directed, bool is_window){
		this->directed = directed;
        this->filename = filename;
        // read our .bin file
        if (is_window){
            ReadWindowGraph();
        }
        else{
            ReadFile();
        }
    }
    Graph(){}

    void ReadWindowGraph(){
        std::cout << "read window graph filename=" << filename << std::endl;
        TimeMeasurer timer;
        timer.StartTimer();

        FILE* file_in = NULL;
        size_t file_pos, file_size, res_size;

        file_in = fopen(filename.c_str(), "rb");
        assert(file_in != NULL);
        fseek(file_in, 0L, SEEK_END);
        file_size = ftell(file_in);
        rewind(file_in);
        
        res_size = fread(&vertex_count, sizeof(IndexType), 1, file_in);
	    assert(res_size == 1);
        file_pos = sizeof(IndexType);
        std::cout << "vertex_count=" << vertex_count << std::endl;
    
        deg = new IndexType[vertex_count];
        in_deg = new IndexType[vertex_count];
        memset(deg, 0, sizeof(IndexType) * vertex_count);
        memset(in_deg, 0, sizeof(IndexType) * vertex_count);

        assert((file_size - file_pos) % 2 == 0);
        size_t edge_num = (file_size - file_pos) / (2 * sizeof(IndexType));
        size_t read_edge_num = edge_num * gWindowRatio;
        size_t read_file_size = read_edge_num * 2 * sizeof(IndexType) + file_pos;

        res_size = 0;
        IndexType v1, v2;
        while (file_pos < read_file_size){
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            assert(0 <= v1 && v1 < vertex_count);
            assert(0 <= v2 && v2 < vertex_count);
            deg[v1]++;
            in_deg[v2]++;
            if (!directed){
                deg[v2]++;
                in_deg[v1]++;
            }
            file_pos += sizeof(IndexType) * 2;
        }
        assert(file_pos == read_file_size);
        assert(res_size * sizeof(IndexType) == read_file_size - sizeof(IndexType));
        std::cout << "edge_num=" << edge_num << ",read_edge_num=" << read_edge_num << ",read_file_size=" << read_file_size << std::endl;
        
        fclose(file_in);

        timer.EndTimer();
        std::cout << "read file elapsed time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
    }

    void ReadFile(){
        std::cout << "readfile filename=" << filename << std::endl;
        TimeMeasurer timer;
        timer.StartTimer();

        FILE* file_in = NULL;
        size_t file_pos, file_size, res_size;

        file_in = fopen(filename.c_str(), "rb");
        assert(file_in != NULL);
        fseek(file_in, 0L, SEEK_END);
        file_size = ftell(file_in);
        rewind(file_in);
        
        res_size = fread(&vertex_count, sizeof(IndexType), 1, file_in);
	    assert(res_size == 1);
        file_pos = sizeof(IndexType);
        std::cout << "vertex_count=" << vertex_count << std::endl;
    
        deg = new IndexType[vertex_count];
        in_deg = new IndexType[vertex_count];
        memset(deg, 0, sizeof(IndexType) * vertex_count);
        memset(in_deg, 0, sizeof(IndexType) * vertex_count);

        res_size = 0;
        IndexType v1, v2;
        while (file_pos < file_size){
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            assert(0 <= v1 && v1 < vertex_count);
            assert(0 <= v2 && v2 < vertex_count);
            deg[v1]++;
            in_deg[v2]++;
            if (!directed){
                deg[v2]++;
                in_deg[v1]++;
            }
            file_pos += sizeof(IndexType) * 2;
        }
        assert(file_pos == file_size);
        assert(res_size * sizeof(IndexType) == file_size - sizeof(IndexType));
        
        fclose(file_in);

        timer.EndTimer();
        std::cout << "read file elapsed time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
    }

    // workload    
    void WriteVertexIdToFile(IndexType *ids, const size_t num, std::string &write_file_name){
        std::ofstream file(write_file_name.c_str());
        assert(file.good());
        for (size_t i = 0; i < num; ++i){
            file << ids[i] << std::endl;
        }
        file.close();
    }
    virtual IndexType UniformChooseVertex(){
        return rand() % vertex_count;
    }
    virtual void UniformChooseVertex(IndexType *ids, const size_t num){
        for (size_t i = 0; i < num; ++i){
            while (1){
                IndexType u = rand() % vertex_count;
                bool valid = true;
                for (size_t j = 0; j < i; ++j){
                    if (ids[j] == u){
                        valid = false;
                    }
                }
                if (deg[u] == 0 || in_deg[u] == 0){
                    valid = false;
                }
                if (valid) {
                    ids[i] = u;
                    break;
                }
            }
        }
        for (size_t i = 0; i < num; ++i){
            std::cout << ids[i] << ",deg=" << deg[ids[i]] << std::endl;
        }
    }
    struct VDegFunctor{
        IndexType *deg;
        VDegFunctor(IndexType *d){
            deg = d;
        }
        bool operator()(IndexType a, IndexType b){
            return deg[a] > deg[b];
        }
    };

    virtual void ChooseVertexDegreeRange(IndexType *ids, const size_t num, IndexType rank_st, IndexType rank_ed, const bool is_out_degree){
        // randomly choose vertex with degree in the range of [rank_st, rank_ed)
        // rank_st, rank_ed is 0-based
        if (rank_ed > vertex_count) rank_ed = vertex_count;
        IndexType *idx = new IndexType[vertex_count];  
        for (IndexType i = 0; i < vertex_count; ++i){
            idx[i] = i;
        }
        IndexType *cmp_deg = is_out_degree ? deg : in_deg;
        std::sort(idx, idx + vertex_count, VDegFunctor(cmp_deg));
        std::cout << "finish sort" << std::endl;
        for (IndexType i = 1; i < vertex_count; ++i) assert(cmp_deg[idx[i-1]] >= cmp_deg[idx[i]]);
        assert(rank_ed - rank_st >= num);
        assert(rank_ed <= vertex_count);
        if (rank_ed - rank_st == num){
            for (int i = 0; i < num; ++i){
                IndexType p = rank_st + i;
                IndexType u = idx[p];
                ids[i] = u;
            }
        }
        else{
            for (IndexType i = 0; i < num; ++i){
                while (1){
                    IndexType p = rand() % (rank_ed - rank_st) + rank_st;
                    IndexType u = idx[p];
                    bool valid = true;
                    for (IndexType j = 0; j < i; ++j){
                        if (ids[j] == u){
                            valid = false;
                            break;
                        }
                    }
                    // choose the connected ones
                    if (deg[u] == 0 || in_deg[u] == 0){
                        valid = false;
                    }
                    if (valid){
                        ids[i] = u;
                        break;
                    }
                }
            }
        }
        for (IndexType i = 0; i < num; ++i){
            std::cout << "u=" << ids[i] << ",deg=" << deg[ids[i]] << ",in_deg=" << in_deg[ids[i]] << std::endl;
        }
        delete[] idx;
        idx = NULL;
    }    
public:
    std::string filename;
	bool directed;
	IndexType vertex_count;

    IndexType *deg;
    IndexType *in_deg;
};

#endif

