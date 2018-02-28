#include "Meta.h"
#include "GraphVec.h"
#include "SlidingGraphVec.h"
#include "PPRRevPushGPU.cuh"
#include "Arguments.h"
#include "PPRRevPushGPUVariants.cuh"

int main(int argc, char* argv[]){
	ArgumentsParser(argc, argv);
    PrintArguments();

	GraphVec *graph = NULL;
	if (gIsDynamic){
		graph = new SlidingGraphVec(gDataFileName, gIsDirected);
	}
	else{
		graph = new GraphVec(gDataFileName, gIsDirected);
	}

    GPUProfiler::InitProfiler();
    
    assert(gAppType == REVERSE_PUSH);
    PPRGPU *ppr_gpu = NULL;
    if (gVariant == OPTIMIZED) ppr_gpu = new PPRRevPushGPU(graph);
    else if (gVariant == FAST_FRONTIER) ppr_gpu = new PPRRevPushGPUFF(graph);
    else if (gVariant == EAGER) ppr_gpu = new PPRRevPushGPUEager(graph);
    else if (gVariant == VANILLA) ppr_gpu = new PPRRevPushGPUVanilla(graph);

    if (gIsDynamic){
        ppr_gpu->DynamicExecute();
    }
    else{
        ppr_gpu->Execute();
    }

    GPUProfiler::ReportProfile();

	return 0;
}
