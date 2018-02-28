#include "PapiProfiler.h"

#if defined(PAPI_PROFILE)
long_long PapiProfiler::papi_values[PapiProfiler::kPapiEventsNum] = {0};
long_long PapiProfiler::papi_temp_values[PapiProfiler::kPapiEventsNum] = {0};
int PapiProfiler::papi_events[kPapiEventsNum] = {0};
#endif
