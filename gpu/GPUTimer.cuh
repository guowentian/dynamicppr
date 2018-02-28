#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#include "GPUUtil.cuh"

class GPUTimer{
    public:
        GPUTimer(){
            CUDA_ERROR(cudaEventCreate(&start));
            CUDA_ERROR(cudaEventCreate(&stop));
        }
        ~GPUTimer(){
            CUDA_ERROR(cudaEventDestroy(start));
            CUDA_ERROR(cudaEventDestroy(stop));
        }
        void StartTimer(){
            CUDA_ERROR(cudaEventRecord(start, 0));
        }
        void EndTimer(){
            CUDA_ERROR(cudaEventRecord(stop, 0));
        }
        float GetElapsedMilliSeconds(){
            float ret;
            CUDA_ERROR(cudaEventSynchronize(stop));
            CUDA_ERROR(cudaEventElapsedTime(&ret, start, stop));
            return ret;
        }
    private:
        cudaEvent_t start;
        cudaEvent_t stop;
};

#endif
