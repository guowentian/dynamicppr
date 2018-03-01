#ifndef __GRAPH_ENCODER_H__
#define __GRAPH_ENCODER_H__

#include <string>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <algorithm>

typedef int FileIndex;

class GraphEncoder{
    public:
    GraphEncoder(){}
    ~GraphEncoder(){}

    void EncodeSnapToBin(const std::string &input_filename, const bool reverse, bool randomize){
        // transform from SNAP form
        // encode into 0-based vertex ids
        // randomize the orders of edges
        std::vector<FileIndex> edge_stream1;
        std::vector<FileIndex> edge_stream2;

        FileIndex min_vertex_id = 1<<30;
        FileIndex max_vertex_id = 0;
        FileIndex vertex_count, edge_count;
        edge_count = 0;
        {
            FileIndex vid;
            std::fstream file(input_filename.c_str(), std::fstream::in);
            assert(file.good());
            while (1){
                file >> vid;
                if (file.eof()) break;
                if (vid < min_vertex_id) min_vertex_id = vid;
                if (vid > max_vertex_id) max_vertex_id = vid;
                edge_count++;
            }
            file.close();
        }
        vertex_count = max_vertex_id - min_vertex_id + 1;
        std::cout << "vertex_count=" << vertex_count << std::endl;

        edge_count /= 2;
        {
            FileIndex v1, v2;
            std::fstream file(input_filename.c_str(), std::fstream::in);
            assert(file.good());
            while (1){
                file >> v1 >> v2;
                if (file.eof()) break;
                v1 -= min_vertex_id;
                v2 -= min_vertex_id;
                assert(v1 < vertex_count && v2 < vertex_count);
                if (!reverse){
                    edge_stream1.push_back(v1);
                    edge_stream2.push_back(v2);
                }
                else{
                    edge_stream1.push_back(v2);
                    edge_stream2.push_back(v1);
                }
            }
            file.close();
        }
        edge_count = edge_stream1.size();
        std::cout << "edge_count=" << edge_count << std::endl;

        if (randomize){
        std::cout << "RAND_MAX=" << RAND_MAX << ",randomize stream..." << std::endl;
            for (size_t i = 0; i < edge_count; ++i){
                FileIndex pos = rand() % edge_count;
                std::swap(edge_stream1[i], edge_stream1[pos]);
                std::swap(edge_stream2[i], edge_stream2[pos]);
            }
        }

        std::string output_filename = input_filename.substr(input_filename.rfind('/') + 1);
        output_filename = output_filename.substr(0, output_filename.find(".txt"));
        std::string suffix = !reverse ? ".bin" : "_rev.bin";
        output_filename += suffix;
        std::cout << "write to file " << output_filename << std::endl;
        FILE *file_out = fopen(output_filename.c_str(), "wb");
        assert(file_out != NULL);
        size_t res = fwrite(&vertex_count, sizeof(FileIndex), 1, file_out);
        assert(res == 1);
        for (size_t i = 0; i < edge_stream1.size(); ++i){
            res = fwrite(&edge_stream1[i], sizeof(FileIndex), 1, file_out);
            assert(res == 1);
            res = fwrite(&edge_stream2[i], sizeof(FileIndex), 1, file_out);
            assert(res == 1);
        }

        fclose(file_out);
    }
        
    void ReverseBinFile(const std::string &input_filename){
        FILE * file_ptr = fopen(input_filename.c_str(), "rb");
        assert(file_ptr != NULL);
        fseek(file_ptr, 0L, SEEK_END);
        size_t file_size = ftell(file_ptr);
        rewind(file_ptr);
        FileIndex vertex_count;
        fread(&vertex_count, sizeof(FileIndex), 1, file_ptr);
    

        std::string output_filename = input_filename.substr(input_filename.rfind('/') + 1);
        output_filename = output_filename.substr(0, output_filename.find(".bin"));
        output_filename += "_rev.bin";
        std::cout << "write to file " << output_filename << std::endl;
        FILE *file_out = fopen(output_filename.c_str(), "wb");
        assert(file_out != NULL);
        size_t res = fwrite(&vertex_count, sizeof(FileIndex), 1, file_out);
        assert(res == 1);
        
        size_t file_pos = sizeof(FileIndex);
        FileIndex v1, v2;
        while (file_pos < file_size){
            res += fread(&v1, sizeof(FileIndex), 1, file_ptr);
            res += fread(&v2, sizeof(FileIndex), 1, file_ptr);
            res += fwrite(&v2, sizeof(FileIndex), 1, file_out);
            res += fwrite(&v1, sizeof(FileIndex), 1, file_out);
            file_pos += sizeof(FileIndex) * 2;
        }
        assert(file_pos == file_size);
        assert((res - 1) / 2 * sizeof(FileIndex) + sizeof(FileIndex) == file_size);
        
        fclose(file_ptr);
        fclose(file_out);
    }

    void EncodeBinToTSEdgeList(const std::string &input_filename, bool directed){
        // transform from our specialized bin file to timestamped edge list file 
        // timestamp is created in monotonic numbering(0-based)
        FILE * file_ptr = fopen(input_filename.c_str(), "rb");
        assert(file_ptr != NULL);
        fseek(file_ptr, 0L, SEEK_END);
        size_t file_size = ftell(file_ptr);
        rewind(file_ptr);
        FileIndex vertex_count;
        size_t res = fread(&vertex_count, sizeof(FileIndex), 1, file_ptr);
        assert(res == 1);

        std::string output_filename = input_filename.substr(input_filename.rfind('/') + 1);
        output_filename = output_filename.substr(0, output_filename.find(".bin"));
        output_filename += "_ts.txt";
        std::cout << "write to file " << output_filename << "," << (directed ? "directed" : "undirected") << std::endl;
        std::ofstream file_out(output_filename.c_str(), std::fstream::out);

        size_t file_pos = sizeof(FileIndex);
        FileIndex v1, v2;
        FileIndex ts = 0;
        while (file_pos < file_size){
            res += fread(&v1, sizeof(FileIndex), 1, file_ptr);
            res += fread(&v2, sizeof(FileIndex), 1, file_ptr);
            file_out << v1 << " " << v2 << " " << ts << std::endl;
            ts++;
            if (!directed){
                file_out << v2 << " " << v1 << " " << ts << std::endl;
                ts++;
            }
            file_pos += sizeof(FileIndex) * 2;
        }
        assert(res * sizeof(FileIndex) == file_size);

        fclose(file_ptr);
        file_out.close();
    }

    void ValidateReverseBin(const std::string &orig_filename, const std::string &rev_filename){
        FILE *file_ptr1 = fopen(orig_filename.c_str(), "rb");
        FILE *file_ptr2 = fopen(rev_filename.c_str(), "rb");
        assert(file_ptr1 != NULL && file_ptr2 != NULL);
        size_t file_size1, file_size2;
        FileIndex v1, v2, v3, v4;

        fseek(file_ptr1, 0L, SEEK_END);
        file_size1 = ftell(file_ptr1);
        rewind(file_ptr1);
        fread(&v1, sizeof(FileIndex), 1, file_ptr1);

        fseek(file_ptr2, 0L, SEEK_END);
        file_size2 = ftell(file_ptr2);
        rewind(file_ptr2);
        fread(&v2, sizeof(FileIndex), 1, file_ptr2);
        assert(v1 == v2);

        size_t file_pos1 = sizeof(FileIndex);
        size_t file_pos2 = sizeof(FileIndex);
        while (file_pos1 < file_size1){
            fread(&v1, sizeof(FileIndex), 1, file_ptr1);
            fread(&v2, sizeof(FileIndex), 1, file_ptr1);
            fread(&v3, sizeof(FileIndex), 1, file_ptr2);
            fread(&v4, sizeof(FileIndex), 1, file_ptr2);
            assert(v1 == v4 && v2 == v3);
            file_pos1 += sizeof(FileIndex) * 2;
            file_pos2 += sizeof(FileIndex) * 2;
        }

        assert(file_pos1 == file_size1 && file_pos2 == file_size2);

        fclose(file_ptr1);
        fclose(file_ptr2);
    }

    void ValidateBin(const std::string& orig_filename, const std::string& bin_filename){
        std::cout << "start validate" << std::endl;

        std::vector<std::vector<FileIndex> > orig_graph;
        std::vector<std::vector<FileIndex> > bin_graph;

        FILE * file_ptr = fopen(bin_filename.c_str(), "rb");
        assert(file_ptr != NULL);
        fseek(file_ptr, 0L, SEEK_END);
        size_t file_size = ftell(file_ptr);
        rewind(file_ptr);
        FileIndex vertex_count;
        fread(&vertex_count, sizeof(FileIndex), 1, file_ptr);
        
        bin_graph.resize(vertex_count);
        orig_graph.resize(vertex_count);

        size_t file_pos = sizeof(FileIndex);
        while (file_pos < file_size){
            FileIndex v1, v2;
            fread(&v1, sizeof(FileIndex), 1, file_ptr);
            fread(&v2, sizeof(FileIndex), 1, file_ptr);
            bin_graph[v1].push_back(v2);
            file_pos += sizeof(FileIndex) * 2;
        }

        for (FileIndex i = 0; i < vertex_count; ++i){
            std::sort(bin_graph[i].begin(), bin_graph[i].end());
        }

        FileIndex min_vertex_id = 1<<30;
        FileIndex max_vertex_id = 0;
        {
            FileIndex vid;
            std::fstream file(orig_filename.c_str(), std::fstream::in);
            assert(file.good());
            while (1){
                file >> vid;
                if (file.eof()) break;
                if (vid < min_vertex_id) min_vertex_id = vid;
                if (vid > max_vertex_id) max_vertex_id = vid;
            }
            file.close();
        }
        {
            FileIndex v1, v2;
            std::fstream file(orig_filename.c_str(), std::fstream::in);
            assert(file.good());
            while (1){
                file >> v1 >> v2;
                if (file.eof()) break;
                v1 -= min_vertex_id;
                v2 -= min_vertex_id;
                orig_graph[v1].push_back(v2);
            }
            file.close();
        }

        for (FileIndex i = 0; i < vertex_count; ++i){
            std::sort(orig_graph[i].begin(), orig_graph[i].end());
        }
        
        for (FileIndex i = 0; i < vertex_count; ++i){
            for (size_t j = 0; j < orig_graph[i].size(); ++j){
                assert(orig_graph[i][j] == bin_graph[i][j]);
            }
        }
        
        std::cout << "finish validate" << std::endl;
    }
};

#endif
