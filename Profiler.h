#ifndef __PROFILER_H__
#define __PROFILER_H__

#include "TimeMeasurer.h"
#include "ProfileCommon.h"
#include <iostream>
#include <cstring>
#include <string>
#include <cassert>
#include <iomanip>

class Profiler{
public:
	static void InitProfiler(size_t thd_count, size_t prof_phase_num, size_t prof_count_type_num){
        thread_count = thd_count;
        profile_phase_num = prof_phase_num;
        profile_count_type_num = prof_count_type_num;

		timer = new TimeMeasurer*[thread_count];
		elapsed_time = new long long*[thread_count];
        accum_count = new long long*[thread_count];
		for (size_t i = 0; i < thread_count; ++i){
			timer[i] = new TimeMeasurer[profile_phase_num];
			elapsed_time[i] = new long long[profile_phase_num];
			memset(elapsed_time[i], 0, sizeof(long long) * profile_phase_num);
		    accum_count[i] = new long long[profile_count_type_num];
            memset(accum_count[i], 0, sizeof(long long) * profile_count_type_num);
        }
	}

	static void StartTimer(size_t thread_id, ProfilePhase phase_type){
#if defined(PROFILE)
		timer[thread_id][phase_type].StartTimer();
#endif
	}

	static void EndTimer(size_t thread_id, ProfilePhase phase_type){
#if defined(PROFILE)
		timer[thread_id][phase_type].EndTimer();
		elapsed_time[thread_id][phase_type] += timer[thread_id][phase_type].GetElapsedMicroSeconds();
#endif
	}

    static void AggCount(size_t thread_id, ProfileCount count_type, int cnt){
#if defined(PROFILE)
        accum_count[thread_id][count_type] += cnt;
#endif
    }

    static void StartTimer(ProfilePhase phase_type){
#if defined(PROFILE)
        StartTimer(0, phase_type);
#endif
    }
    static void EndTimer(ProfilePhase phase_type){
#if defined(PROFILE)
        EndTimer(0, phase_type);
#endif
    }
    static void AggCount(ProfileCount count_type, int cnt){
#if defined(PROFILE)
        AggCount(0, count_type, cnt);
#endif
    }
    
    static long long GetPhaseTime(size_t thread_id, ProfilePhase phase_type){
        // in microseconds
        return elapsed_time[thread_id][phase_type];
    }
    static long long GetPhaseTime(ProfilePhase phase_type){
        return GetPhaseTime(0, phase_type);
    }
    
    static void ReportProfile(){
#if defined(PROFILE)
		std::cout << "****************** profile time **********************" << std::endl;
		for (size_t i = 0; i < thread_count; ++i){
			std::cout << "thread_id=" << i << ",";
			for (size_t j = 0; j < profile_phase_num; ++j){
				std::cout << "[" << GetPhaseString(j) << "]=" << elapsed_time[i][j] * 1.0 / 1000.0 << "ms ";
			}
			std::cout << std::endl;
        }

        for (size_t i = 0; i < thread_count; ++i){
            std::cout << "thread_id=" << i << ",";
            for (size_t j = 0; j < profile_count_type_num; ++j){
                std::cout << "[" << GetCountString(j) << "]=" << accum_count[i][j] << " ";
            }
            std::cout << std::endl;
        }
        
		std::cout << "****************** end profile  **********************" << std::endl;
#endif
	}

public:
	static TimeMeasurer **timer;
	static long long **elapsed_time;
    static long long **accum_count;
	static size_t thread_count;
    static size_t profile_phase_num;
    static size_t profile_count_type_num;
};

#endif
