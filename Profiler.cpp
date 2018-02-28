#include "Profiler.h"

TimeMeasurer **Profiler::timer = NULL;
long long **Profiler::elapsed_time = NULL;
size_t Profiler::thread_count = 0;
long long **Profiler::accum_count = NULL;
size_t Profiler::profile_phase_num = 0;
size_t Profiler::profile_count_type_num = 0;

