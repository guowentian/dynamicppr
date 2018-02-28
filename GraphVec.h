#ifndef __GRAPH_VEC_H__
#define __GRAPH_VEC_H__

#include "Meta.h"
#include "TimeMeasurer.h"
#include "EdgeBatch.h"
#include <string>
#include <cstring>
#include <cassert>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

class GraphVec{
public:

	GraphVec(const std::string &filename, bool directed = true){
		this->directed = directed;
        this->filename = filename;
        ConstructGraph();
    }
    GraphVec(){}
    ~GraphVec(){}

	virtual void ConstructGraph(){
        // edge file
		// format: a b  (a and b is vertex id)
		TimeMeasurer timer;
		timer.StartTimer();

        // initialize file
	    FILE* file_ptr = fopen(filename.c_str(), "rb");
        assert(file_ptr != NULL);
        fseek(file_ptr, 0L, SEEK_END);
        size_t file_size = ftell(file_ptr);
        rewind(file_ptr);
        
        // first int indicate vertex_count
        size_t res_size;
        size_t file_pos = 0;
        res_size = fread(&vertex_count, sizeof(IndexType), 1, file_ptr);
        assert(res_size == 1);
        file_pos += sizeof(IndexType);
        std::cout << "vertex_count=" << vertex_count << std::endl;
        
        col_ind.resize(vertex_count);
        in_col_ind.resize(vertex_count);
        deg.resize(vertex_count);

        IndexType v1, v2;
        edge_count = 0;
        while (file_pos < file_size){
            res_size += fread(&v1, sizeof(IndexType), 1, file_ptr);
            res_size += fread(&v2, sizeof(IndexType), 1, file_ptr);

            assert(0 <= v1 && v1 < vertex_count);
            assert(0 <= v2 && v2 < vertex_count);
            
            col_ind[v1].push_back(v2);
            in_col_ind[v2].push_back(v1);
            edge_count++;
            if (!directed){
                col_ind[v2].push_back(v1);
                in_col_ind[v1].push_back(v2);
                edge_count++;
            }
            file_pos += sizeof(IndexType) * 2;
        }
        IndexType edge_stream_count = directed ? edge_count : edge_count / 2;
        assert(res_size / 2 == edge_stream_count);
        std::cout << "edge_stream_count=" << edge_stream_count << std::endl;
        std::cout << "edge_count=" << edge_count << std::endl;
       
        for (IndexType i = 0; i < vertex_count; ++i) deg[i] = col_ind[i].size();
        
        fclose(file_ptr);
        
		timer.EndTimer();
		std::cout << "read file elapsed time=" << timer.GetElapsedMicroSeconds() * 1.0 / 1000 << "ms" << std::endl;
    }
    
	
    void SortCSRColumns(){
        for (IndexType i = 0; i < vertex_count; ++i){
            std::sort(col_ind[i].begin(), col_ind[i].end());
        }
        for (IndexType i = 0; i < vertex_count; ++i){
            std::sort(in_col_ind[i].begin(), in_col_ind[i].end());
        }
    }
    virtual void SerializeEdgeStream(EdgeBatch *edge_stream){
        FILE* file_ptr = fopen(filename.c_str(), "rb");
        assert(file_ptr != NULL);
        fseek(file_ptr, 0L, SEEK_END);
        size_t file_size = ftell(file_ptr);
        rewind(file_ptr);
        
        size_t res_size, file_pos;
        IndexType vcount;
        res_size = fread(&vcount, sizeof(IndexType), 1, file_ptr);
        assert(res_size == 1);
        file_pos = sizeof(IndexType);

        IndexType v1, v2;
        edge_stream->length = 0;
        while (file_pos < file_size){
            res_size += fread(&v1, sizeof(IndexType), 1, file_ptr);
            res_size += fread(&v2, sizeof(IndexType), 1, file_ptr);
            edge_stream->edge1[edge_stream->length] = v1;
            edge_stream->edge2[edge_stream->length] = v2;
            edge_stream->length++;
            file_pos += sizeof(IndexType) * 2;
        }
        fclose(file_ptr); 
    }

    virtual bool StreamUpdates(const size_t stream_count){
		return true;
	}

public:
    std::string filename;
	bool directed;
	IndexType vertex_count;
	IndexType edge_count;

    std::vector<IndexType> deg;
    std::vector<std::vector<IndexType> > col_ind;
    std::vector<std::vector<IndexType> > in_col_ind;

};


#endif
