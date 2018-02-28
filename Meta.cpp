#include "Meta.h"


std::string gDataFileName = "";
int gAppType = -1;
int gIsDirected = -1;
int gIsDynamic = -1;

int gWorkloadConfigType = SLIDE_WINDOW_RATIO; 
double gWindowRatio = 0.1;
double gStreamUpdateCountVersusWindowRatio = -1.0;
size_t gStreamBatchCount = 0;
size_t gStreamUpdateCountTotal = 0;
size_t gStreamUpdateCountPerBatch = 0;

int gSourceVertexId = 1;

int gThreadNum = 1;
int gVariant = 0;

ValueType gTolerance = 0.000000001;

