#ifndef __SLIDING_GRAPH_VEC_H__
#define __SLIDING_GRAPH_VEC_H__

#include "GraphVec.h"
#include "EdgeBatch.h"
#include <vector>
#include <set>

class SlidingGraphVec : public GraphVec{
public:
	SlidingGraphVec(const std::string &fname, bool di = true){
        this->filename = fname;
        this->directed = di;
		PrepareSlidingGraph();
	    out_change_count.resize(vertex_count);
	    in_change_count.resize(vertex_count);
        edge_batch = new EdgeBatch(gStreamUpdateCountPerBatch * 4);
        new_stream = new EdgeBatch(gStreamUpdateCountPerBatch * 2);
    }

    ~SlidingGraphVec(){
        delete edge_batch;
        edge_batch = NULL;
        delete new_stream;
        new_stream = NULL;
    }

	void PrepareSlidingGraph(){
		// edge file
		// format: a b  (a and b is vertex id)
		TimeMeasurer timer;
		timer.StartTimer();

        std::cout << "read filename=" << filename << std::endl;
        file_in = fopen(filename.c_str(), "rb");
        assert(file_in != NULL);
        fseek(file_in, 0L, SEEK_END);
        file_size = ftell(file_in);
        rewind(file_in);
// config parameters
        size_t res_size;
        res_size = fread(&vertex_count, sizeof(IndexType), 1, file_in);
	    assert(res_size == 1);
        file_pos = sizeof(IndexType);
        std::cout << "vertex_count=" << vertex_count << std::endl;
        
        size_t total_edge_stream_length = (file_size - file_pos) / sizeof(IndexType) / 2;
        sliding_window_size = total_edge_stream_length * gWindowRatio;
        if (gWorkloadConfigType == SLIDE_WINDOW_RATIO){
        // the input is gStreamUpdateCountVersusWindowRatio and gStreamBatchCount
        // slide #(gStreamBatchCount) times, each time slide #(gStreamUpdateCountVersusWindowRatio*gWindowRatio*E) edges
            gStreamUpdateCountPerBatch = gStreamUpdateCountVersusWindowRatio * sliding_window_size;
            gStreamUpdateCountTotal = gStreamUpdateCountPerBatch * gStreamBatchCount;
        }
        else if (gWorkloadConfigType == SLIDE_BATCH_SIZE){
        // the input is gStreamUpdateCountPerBatch and gStreamUpdateCountTotal
        // slide #(gStreamUpdateCountPerBatch) edges each update, consume #(gStreamUpdateCountTotal) edges in total
            gStreamBatchCount = (gStreamUpdateCountTotal + gStreamUpdateCountPerBatch - 1) / gStreamUpdateCountPerBatch;
            assert(gStreamBatchCount > 0);
        }
        else{
            assert(false);
        }
        if (gStreamUpdateCountTotal > total_edge_stream_length - sliding_window_size){
            gStreamUpdateCountTotal = total_edge_stream_length - sliding_window_size;
        }
        std::cout << "after workload config: gStreamUpdateCountPerBatch=" << gStreamUpdateCountPerBatch << ",gStreamBatchCount=" << gStreamBatchCount << ",gStreamUpdateCountTotal=" << gStreamUpdateCountTotal << std::endl;
        edge_count = sliding_window_size;
		if (this->directed == false) edge_count *= 2;
        std::cout << "sliding window size=" << sliding_window_size << ",gStreamUpdateCountPerBatch=" << gStreamUpdateCountPerBatch << std::endl;
	    std::cout << "edge_count=" << edge_count << std::endl;

        // init
        deg.resize(vertex_count);
        col_ind.resize(vertex_count);
        in_col_ind.resize(vertex_count);
        for (IndexType i = 0; i < vertex_count; ++i) deg[i] = 0;

		// read initial stream for window
        res_size = 0;
        for (IndexType i = 0, v1, v2; i < sliding_window_size; ++i){
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            deg[v1]++;
            col_ind[v1].push_back(v2);
            in_col_ind[v2].push_back(v1);
            if (!directed){
                deg[v2]++;
                col_ind[v2].push_back(v1);
                in_col_ind[v1].push_back(v2);
            }
            file_pos += 2 * sizeof(IndexType);
        }
        assert(res_size == 2 * sliding_window_size);

    }

    
    void ScratchConstructWindowGraph(){
        // construct the current window from scratch
        // construct both out and in edges
#if defined(PROFILE)
        IndexType window_end = (file_pos - sizeof(IndexType)) / sizeof(IndexType) / 2;
        IndexType window_start = window_end - sliding_window_size ;
        std::cout << "construct scratch in window graph [" << window_start << "," << window_end << ")" << std::endl;
#endif
        TimeMeasurer timer;
        timer.StartTimer();

        // init
        for (IndexType i = 0; i < vertex_count; ++i) in_col_ind[i].clear();
        for (IndexType i = 0; i < vertex_count; ++i) col_ind[i].clear();
        // file read
        size_t window_left_pos = file_pos - sliding_window_size * sizeof(IndexType) * 2;
        fseek(file_in, window_left_pos, SEEK_SET);
        size_t res_size = 0;
        for (size_t cur = window_left_pos; cur < file_pos; cur += sizeof(IndexType) * 2){
            IndexType v1, v2;
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            in_col_ind[v2].push_back(v1);
            col_ind[v1].push_back(v2);
            if (!directed){
                in_col_ind[v1].push_back(v2);
                col_ind[v2].push_back(v1);
            }
        }
        assert(res_size == sliding_window_size * 2);
        assert(ftell(file_in) == file_pos);
        for (IndexType i = 0; i < vertex_count; ++i) deg[i] = col_ind[i].size();
        
        timer.EndTimer();
#if defined(PROFILE)
		std::cout << "construct graph elapsed time=" << timer.GetElapsedMicroSeconds() * 1.0 / 1000 << "ms" << std::endl;
#endif
    }

    /* construct window graph with edge_batch */
    void IncConstructWindowGraph(){
#if defined(PROFILE)
        IndexType window_end = (file_pos - sizeof(IndexType)) / sizeof(IndexType) / 2;
        IndexType window_start = window_end - sliding_window_size ;
        std::cout << "construct incremental in window graph [" << window_start << "," << window_end << ")" << std::endl;
#endif
        TimeMeasurer timer;
        timer.StartTimer();

        // these edges are directed
        // delete
        for (IndexType i = 0; i < edge_batch->length; ++i){
            IndexType v1 = edge_batch->edge1[i];
            IndexType v2 = edge_batch->edge2[i];
            out_change_count[v1] = 0;
            in_change_count[v2] = 0;
        }
        for (IndexType i = 0; i < edge_batch->length; ++i){
            if (edge_batch->is_insert[i] == false){
                IndexType v1 = edge_batch->edge1[i];
                IndexType v2 = edge_batch->edge2[i];
                out_change_count[v1]++;
                in_change_count[v2]++;
            }
        }
        for (IndexType i = 0; i < edge_batch->length; ++i){
            if (edge_batch->is_insert[i] == false){
                IndexType v1 = edge_batch->edge1[i];
                IndexType v2 = edge_batch->edge2[i];
                deg[v1]--;
                if (out_change_count[v1]){
                    col_ind[v1].erase(col_ind[v1].begin(), col_ind[v1].begin() + out_change_count[v1]);
                    out_change_count[v1] = 0;
                }
                if (in_change_count[v2]){
                    in_col_ind[v2].erase(in_col_ind[v2].begin(), in_col_ind[v2].begin() + in_change_count[v2]);
                    in_change_count[v2] = 0;
                }
            }
        }
        
        //insert
        for (IndexType i = 0; i < edge_batch->length; ++i){
            if (edge_batch->is_insert[i]){
                IndexType v1 = edge_batch->edge1[i];
                IndexType v2 = edge_batch->edge2[i];
                deg[v1]++;
                col_ind[v1].push_back(v2);
                in_col_ind[v2].push_back(v1);
            }
        }
       
		timer.EndTimer();
#if defined(PROFILE)
        std::cout << "construct graph elapsed time=" << timer.GetElapsedMicroSeconds() * 1.0 / 1000 << "ms" << std::endl;
#endif
    }

    virtual void ConstructGraph(){
        ScratchConstructWindowGraph();
    }

    virtual void SerializeEdgeStream(EdgeBatch *edge_stream){
        // edge stream in current window
        size_t window_left_pos = file_pos - sliding_window_size * sizeof(IndexType) * 2;
        fseek(file_in, window_left_pos, SEEK_SET);
        size_t res_size = 0;
        edge_stream->length = 0;
        for (size_t cur = window_left_pos; cur < file_pos; cur += sizeof(IndexType) * 2){
            IndexType v1, v2;
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            edge_stream->edge1[edge_stream->length] = v1;
            edge_stream->edge2[edge_stream->length] = v2;
            edge_stream->length++;
        }
        assert(res_size == sliding_window_size * 2);
        assert(ftell(file_in) == file_pos);
    }

	virtual bool StreamUpdates(const size_t stream_count){
        size_t move_pos = stream_count * sizeof(IndexType) * 2;
        if (file_pos + move_pos > file_size) return true;

        size_t res_size = 0;
        IndexType v1, v2;
        // feed new_stream
        fseek(file_in, file_pos, SEEK_SET);
        for (size_t i = 0; i < stream_count; ++i){
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            new_stream->edge1[i] = v1;
            new_stream->edge2[i] = v2;
            new_stream->is_insert[i] = true;
        }
        new_stream->length = stream_count;
       
        // feed edge_batch
        res_size = 0;
        size_t window_left_pos = file_pos - sliding_window_size * sizeof(IndexType) * 2;
        fseek(file_in, window_left_pos, SEEK_SET);
        //delete
        for (size_t i = 0; i < stream_count; ++i){
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            edge_batch->edge1[i] = v1;
            edge_batch->edge2[i] = v2;
            edge_batch->is_insert[i] = false;
        }
        assert(res_size == stream_count * 2);

        fseek(file_in, file_pos, SEEK_SET);
        // insert
        res_size = 0;
        for (size_t i = stream_count; i < 2*stream_count; ++i){
            res_size += fread(&v1, sizeof(IndexType), 1, file_in);
            res_size += fread(&v2, sizeof(IndexType), 1, file_in);
            edge_batch->edge1[i] = v1;
            edge_batch->edge2[i] = v2;
            edge_batch->is_insert[i] = true;
        }
        assert(res_size = stream_count * 2);
        
        file_pos += stream_count * sizeof(IndexType) * 2;
        assert(ftell(file_in) == file_pos);
        edge_batch->length = stream_count * 2;
        
        if (!directed){
            size_t len = edge_batch->length;
            memcpy(edge_batch->edge1 + len, edge_batch->edge2, sizeof(IndexType) * len);
            memcpy(edge_batch->edge2 + len, edge_batch->edge1, sizeof(IndexType) * len);
            memcpy(edge_batch->is_insert + len, edge_batch->is_insert, sizeof(bool) * len);
            edge_batch->length = 2 * len;
        }
        
	    return false;
	}

    
public:
    FILE *file_in;
    size_t file_size, file_pos;

    IndexType sliding_window_size;
    EdgeBatch *edge_batch; // indicate edge direction, edge type
    EdgeBatch *new_stream; // do not indicate edge direction or edge type, used for gpu sliding_graph_builder

    // assist for construct window
    std::vector<IndexType> out_change_count; 
    std::vector<IndexType> in_change_count;
};

#endif
