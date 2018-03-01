#include "Graph.h"
#include <ctime>
#include <cstdlib>
#include <sstream>

std::string IntToString(int a){
    std::ostringstream ss;
    ss << a;
    return ss.str();
}
std::string GetOutputFileName(const std::string &input_filename, const std::string &feature){
    return input_filename + "_" + feature + ".txt";
}
void GenerateTopN(Graph *g, std::string &output_filename, bool is_out_degree, size_t gen_num, IndexType rank_st, IndexType rank_ed){
    IndexType *source_vertex_ids = new IndexType[gen_num];
    g->ChooseVertexDegreeRange(source_vertex_ids, gen_num, rank_st, rank_ed, is_out_degree);
    g->WriteVertexIdToFile(source_vertex_ids, gen_num, output_filename);
    delete[] source_vertex_ids;
    source_vertex_ids = NULL;
}
void GenerateRandom(Graph *g, std::string &output_filename, size_t gen_num){
    IndexType *source_vertex_ids = new IndexType[gen_num];
    g->UniformChooseVertex(source_vertex_ids, gen_num);
    g->WriteVertexIdToFile(source_vertex_ids, gen_num, output_filename);
    delete[] source_vertex_ids;
    source_vertex_ids = NULL;
}

int main(int argc, char* argv[]){
    if (argc != 5){
        std::cout << "./workload filename directed is_window is_choose_outdegree" << std::endl;
        return -1;
    } 
    std::string input_filename(argv[1]);
    int directed = atoi(argv[2]);
    int is_window = atoi(argv[3]);
    int is_out_degree = atoi(argv[4]);
   
    std::cout << "is_directed=" << directed << ",window=" << is_window << ",is_outdegree=" << is_out_degree << std::endl;
    Graph *g = new Graph(input_filename, directed, is_window);
    std::string window_prefix = is_window ? "window" : "";
    std::string degree_prefix = is_out_degree ? "" : "rev";

    input_filename = input_filename.substr(input_filename.rfind("/") + 1);
    std::string top10_file = GetOutputFileName(input_filename, "top" + window_prefix + degree_prefix + IntToString(10));
    std::cout << "top10 filename=" << top10_file << std::endl;
    GenerateTopN(g, top10_file, is_out_degree, 10, 0, 10);

    std::string top1000_file = GetOutputFileName(input_filename, "top" + window_prefix + degree_prefix + IntToString(1000));
    std::cout << "top1000 filename=" << top1000_file << std::endl;
    GenerateTopN(g, top1000_file, is_out_degree, 10, 10, 1000);

    std::string top1000000_file = GetOutputFileName(input_filename, "top" + window_prefix + degree_prefix + IntToString(1000000));
    std::cout << "top1000000 filename=" << top1000000_file << std::endl;
    GenerateTopN(g, top1000000_file, is_out_degree, 10, 1000, 1000000);


    /*std::string rand_file = GetOutputFileName(input_filename, std::string("random"));
    std::cout << "random filename=" << rand_file << std::endl;
    GenerateRandom(g, rand_file, 10);
*/
    return 0;
}
