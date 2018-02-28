#ifndef __PAPI_PROFILER_H__
#define __PAPI_PROFILER_H__

#if defined(PAPI_PROFILE)
#include <papi.h>
#include <iostream>

class PapiProfiler{
    public:
        static void InitPapiProfiler(){
// l1_dcm(conflict with lst_ins), l2_dcm, l2_dca, l2_tcm, l2_tca, l3_tcm, l3_tca / res_stl,tot_cyc
// L2 cache miss ratio: PAPI_L2_TCM / PAPI_L2_TCA
// graduated instructions per cycle = PAPI_TOT_INS / PAPI_TOT_CYC
// percentage of cycles stalled on any resource = PAPI_RES_STL / PAPI_TOT_CYC
            InitPapiProfiler1(); 
        }
        static void InitPapiProfiler1(){
            papi_events[0] = PAPI_L2_DCM;
            papi_events[1] = PAPI_L2_DCA;
            papi_events[2] = PAPI_L3_TCM;
            papi_events[3] = PAPI_L3_TCA;
        }
        static void InitPapiProfiler2(){
            papi_events[0] = PAPI_TOT_INS;
            papi_events[1] = PAPI_TOT_CYC;
            papi_events[2] = PAPI_RES_STL;
        }
        static void BeginProfile(){
            if (PAPI_start_counters(papi_events, kPapiEventsNum) != PAPI_OK){
                std::cerr << "PAPI start counters fail!" << std::endl;
                exit(-1);
            }
        }
        static void EndProfile(){
            if (PAPI_stop_counters(papi_temp_values, kPapiEventsNum) != PAPI_OK){
                std::cerr << "PAPI end counters fail!" << std::endl;
                exit(-1);
            }
            for (size_t i = 0; i < kPapiEventsNum; ++i){
                papi_values[i] += papi_temp_values[i];
            }
        }
        static void ReportProfile(){
            ReportProfile1();
        }
        static void ReportProfile1(){
            std::cout << "L2_data_cache_miss " <<  papi_values[0] << " L2_data_cache_hit " << papi_values[1] << std::endl;
            std::cout << "L2_data_cache_miss_rate " << papi_values[0] * 1.0 / (papi_values[0]+papi_values[1]) << std::endl;
            std::cout << "L3_cache_miss " << papi_values[2] << " L3_cache_hit " << papi_values[3] << std::endl;
            std::cout << "L3_cache_miss_rate " << papi_values[2] * 1.0 / (papi_values[2]+papi_values[3]) << std::endl; 
        }
        static void ReportProfile2(){
            std::cout << "total_instruction " << papi_values[0] << " total_cycles " << papi_values[1] << " resource_stall_cycles " << papi_values[2] << std::endl; \
            std::cout << "instruction_per_cycle " << papi_values[0] * 1.0 / papi_values[1] << std::endl;\
            std::cout << "percentage_resource_stall_cycles " << papi_values[2] * 1.0 / papi_values[1] << std::endl;
        }
    public:
        static const size_t kPapiEventsNum = 4;
        //const static kPapiEventsNum = 3;
        
        static int papi_events[kPapiEventsNum];
        static long_long papi_temp_values[kPapiEventsNum];
        static long_long papi_values[kPapiEventsNum];

};
#endif

#endif
