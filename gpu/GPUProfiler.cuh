#ifndef __GPU_PROFILER_H__
#define __GPU_PROFILER_H__

#include "GPUTimer.cuh"
#include "ProfileCommon.h"
#include <iostream>
#include <iomanip>

class GPUProfiler{
public:
	static void InitProfiler(){
#if defined(PROFILE)
		timer = new GPUTimer[PROFILE_PHASE_NUM];
        elapsed_time = new float[PROFILE_PHASE_NUM];
        memset(elapsed_time, 0, sizeof(float) * PROFILE_PHASE_NUM);
        accum_count = new long long[PROFILE_COUNT_TYPE_NUM];
        memset(accum_count, 0, sizeof(long long) * PROFILE_COUNT_TYPE_NUM);
#endif
	}

    static void StartTimer(ProfilePhase phase_type){
#if defined(PROFILE)
        timer[phase_type].StartTimer();
#endif
    }
    static void EndTimer(ProfilePhase phase_type){
#if defined(PROFILE)
        timer[phase_type].EndTimer();
        elapsed_time[phase_type] += timer[phase_type].GetElapsedMilliSeconds();
#endif
    }
    static void AggCount(ProfileCount count_type, int cnt){
#if defined(PROFILE)
        accum_count[count_type] += cnt;
#endif
    }
    static float GetPhaseTime(ProfilePhase phase_type){
        return elapsed_time[phase_type];
    }
    
    static void ReportProfile(){
#if defined(PROFILE)
		std::cout << "****************** profile time **********************" << std::endl;
        for (size_t j = 0; j < PROFILE_PHASE_NUM; ++j){
            std::cout << "[" << GetPhaseString(j) << "]=" << elapsed_time[j] << "ms ";
        }
        std::cout << std::endl;

        for (size_t j = 0; j < PROFILE_COUNT_TYPE_NUM; ++j){
            std::cout << "[" << GetCountString(j) << "]=" << accum_count[j] << " ";
        }
        std::cout << std::endl;
    	std::cout << "****************** end profile  **********************" << std::endl;
#endif
	}

public:
	static GPUTimer *timer;
	static float *elapsed_time;
    static long long *accum_count;
};

#endif
