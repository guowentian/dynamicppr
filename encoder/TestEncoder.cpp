#include "GraphEncoder.h"
#include <string>
#include <iostream>
#include "GraphVec.h"
#include <vector>

int main(int argc, char*argv[]){
    if (argc != 2 && argc != 3){
        std::cout << "./encoder input_filename reverse" << std::endl;
        return -1;
    }
    std::string filename = std::string(argv[1]);
    bool reverse = false;
    if (argc == 3){
        reverse = atoi(argv[2]);
    }
    GraphEncoder *encoder = new GraphEncoder();
    encoder->EncodeSnapToBin(filename, reverse, true);
    //encoder->ValidateBin(filename, "test.bin");
    //encoder->EncodeBinToTSEdgeList(filename, directed);

    return 0;
}
