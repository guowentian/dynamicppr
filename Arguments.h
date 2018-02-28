#ifndef __ARGUMENTS_H__
#define __ARGUMENTS_H__

#include <iostream>
#include <cassert>
#include "Meta.h"
#include "CommandLine.h"

static void PrintUsage() {
	std::cout << "==========[USAGE]==========" << std::endl;
    std::cout << "-d: gDataFileName" << std::endl;
    std::cout << "-a: gAppType" << std::endl;
    std::cout << REVERSE_PUSH << ":rev push" << std::endl; 
    std::cout << "-i: gIsDirected" << std::endl;
    std::cout << "-y: gIsDynamic" << std::endl;
    
    std::cout << "-w: gWindowRatio" << std::endl;
    std::cout << "-n: gWorkloadConfigType" << std::endl;
    std::cout << SLIDE_WINDOW_RATIO << ": SLIDE_WINDOW_RATIO, " << SLIDE_BATCH_SIZE << ": SLIDE_BATCH_SIZE" << std::endl;
    std::cout << "-r: gStreamUpdateCountVersusWindowRatio" << std::endl;
    std::cout << "-b: gStreamBatchCount" << std::endl;
    std::cout << "-c: gStreamUpdateCountPerBatch" << std::endl;
    std::cout << "-l: gStreamUpdateCountTotal" << std::endl;
    
    std::cout << "-s: gSourceVertexId" << std::endl;
    std::cout << "-t: gThreadNum" << std::endl;
    std::cout << "-o: gVariant" << std::endl;
    std::cout << OPTIMIZED << ": optimized, " << FAST_FRONTIER << ": fast frontier, " << EAGER << ": eager, " << VANILLA << ": VANILLA" << std::endl; 
    std::cout << "-e: error tolerance" << std::endl;
    std::cout << "EXAMPLE: ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 1 -w 0.1 -n 0 -r 0.01 -b 1000 -s 1" << std::endl;
    std::cout << "EXAMPLE: ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 1 -w 0.1 -n 1 -c 100 -l 10000 -s 1" << std::endl;
    std::cout << "deprected EXAMPLE: ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 0" << std::endl;
}
static void PrintArguments(){
    std::cout << "gAppType=" << gAppType << ",gIsDirected=" << gIsDirected << ",gIsDynamic=" << gIsDynamic << std::endl;
    std::cout << "gWindowRatio=" << gWindowRatio << ",gWorkloadConfigType=" << gWorkloadConfigType << ",gStreamUpdateCountVersusWindowRatio=" << gStreamUpdateCountVersusWindowRatio << ",gStreamBatchCount=" << gStreamBatchCount << ",gStreamUpdateCountPerBatch=" << gStreamUpdateCountPerBatch << ",gStreamUpdateCountTotal=" << gStreamUpdateCountTotal << std::endl;
    std::cout << "gSourceVertexId=" << gSourceVertexId << std::endl;
    std::cout << "gThreadNum=" << gThreadNum << ",gVariant=" << gVariant << std::endl;
    std::cout << "error=" << gTolerance << ",ALPHA=" << ALPHA << std::endl;
}

static void ArgumentsChecker() {
	bool valid = true;
    if (gAppType < 0 || gAppType > kAlgoTypeSize){
        valid = false;
    }
    if (gIsDirected < 0 || gIsDynamic < 0 || gDataFileName == ""){
        valid = false;
    }
    if (gWorkloadConfigType == SLIDE_WINDOW_RATIO){
        if (gStreamUpdateCountVersusWindowRatio < 0.0 || gStreamBatchCount == 0) valid = false;
    }
    else if (gWorkloadConfigType == SLIDE_BATCH_SIZE){
        if (gStreamUpdateCountPerBatch == 0 || gStreamUpdateCountTotal == 0) valid = false;
    }
    else{
        valid = false;
    }
    if (!valid){
        std::cout << "invalid arguments" << std::endl;
        PrintUsage();
        exit(-1);
    }
}

static void ArgumentsParser(int argc, char *argv[]) {
	CommandLine commandline(argc, argv);
    
    gDataFileName = commandline.GetOptionValue("-d", "");
    gAppType = commandline.GetOptionIntValue("-a", 0);
    gIsDirected = commandline.GetOptionIntValue("-i", -1);
    gIsDynamic = commandline.GetOptionIntValue("-y", -1);
    
    gWindowRatio = commandline.GetOptionDoubleValue("-w", 0.1);
    gWorkloadConfigType = commandline.GetOptionIntValue("-n", SLIDE_WINDOW_RATIO);
    gStreamUpdateCountVersusWindowRatio = commandline.GetOptionDoubleValue("-r", -1.0);
    gStreamBatchCount = commandline.GetOptionIntValue("-b", 0);
    gStreamUpdateCountPerBatch = commandline.GetOptionIntValue("-c", 0);
    gStreamUpdateCountTotal = commandline.GetOptionIntValue("-l", 0);
    gSourceVertexId = commandline.GetOptionIntValue("-s", 1);
    gThreadNum = commandline.GetOptionIntValue("-t", 1);
    gVariant = commandline.GetOptionIntValue("-o", 0);
    gTolerance = commandline.GetOptionDoubleValue("-e", 0.000000001);
    
	ArgumentsChecker();
}

#endif
