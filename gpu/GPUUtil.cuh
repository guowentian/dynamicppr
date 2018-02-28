#ifndef __GPU_UTIL_H__
#define __GPU_UTIL_H__

#include <cstdio>
#include "cusparse.h"

static void HandleError(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

#define CUDA_ERROR( err ) (HandleError( err, __FILE__, __LINE__))
#define CUSPARSE_ERROR(status) \
    if (status != CUSPARSE_STATUS_SUCCESS){ \
        printf("cusparse error in %s at line %d\n", __FILE__, __LINE__);\
        exit(-1); \
    }

__device__ static double atomicAdd(double* address, double val) { 
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __longlong_as_double(old); 
}
__device__ static double atomicMul(double *address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __longlong_as_double(old);
}

template<typename T>
__device__ T MIN(T a, T b){
    if (a < b) return a;
    return b;
}

#endif
