#include "GPUProfiler.cuh"

GPUTimer *GPUProfiler::timer = NULL;
float *GPUProfiler::elapsed_time = NULL;
long long *GPUProfiler::accum_count = NULL;
