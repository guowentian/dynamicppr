#ifndef __PROFILE_COMMON_H__
#define __PROFILE_COMMON_H__

#include <string>

enum ProfilePhase{
	INSPECT_TIME,
    EXPAND_TIME,
    INIT_GRAPH_CALC_TIME,
    DYNA_GRAPH_CALC_TIME,
    EXCLUDE_GRAPH_UPDATE_TIME,
    SORT_TIME,
    REDUCE_TIME,
    REPAIR_FRONTIER_TIME,
    INC_UPDATE_TIME,
    PUSH_TIME,
    TOTAL_TIME,
    PPR_TIME,
    PPR_UPDATE_TIME,
    PPR_QUERY_TIME,
    PROFILE_PHASE_NUM
};

enum ProfileCount{
    TRAVERSE_COUNT,
    EXPAND_COUNT,
    UPDATE_POS_RESIDUAL_COUNT,
    UPDATE_NEG_RESIDUAL_COUNT,
    UPDATE_RANDOM_WALK_COUNT,
    PROFILE_COUNT_TYPE_NUM
};

static std::string GetPhaseString(size_t p){
    if (p == INSPECT_TIME){
        return "inspect_time";
    }
    else if (p == EXPAND_TIME){
        return "expand_time";
    }
    else if (p == INIT_GRAPH_CALC_TIME){
        return "init_graph_calculation_time";
    }
    else if (p == DYNA_GRAPH_CALC_TIME){
        return "dynamic_graph_calculation_time";
    }
    else if (p == INC_UPDATE_TIME){
        return "inc_update_time";
    }
    else if (p == PUSH_TIME){
        return "push_time";
    }
    else if (p == SORT_TIME){
        return "sort_time";
    }
    else if (p == REDUCE_TIME){
        return "reduce_time";
    }
    else if (p == REPAIR_FRONTIER_TIME){
        return "repair_frontier_time";
    }
    else if (p == EXCLUDE_GRAPH_UPDATE_TIME){
        return "exclude_graph_update_time";
    }
    else if (p == TOTAL_TIME){
        return "total_time";
    }
    else if (p == PPR_TIME){
        return "ppr_time";
    }
    else if (p == PPR_UPDATE_TIME){
        return "ppr_update_time";
    }
    else if (p == PPR_QUERY_TIME){
        return "ppr_query_time";
    }
    return "";
}
static std::string GetCountString(size_t p){
    if (p == TRAVERSE_COUNT){
        return "traverse_count";
    }
    else if (p == EXPAND_COUNT){
        return "expand_count";
    }
    else if (p == UPDATE_POS_RESIDUAL_COUNT){
        return "update_pos_residual_count";
    }
    else if (p == UPDATE_NEG_RESIDUAL_COUNT){
        return "update_neg_residual_count";
    }
    else if (p == UPDATE_RANDOM_WALK_COUNT){
        return "update_random_walk_count";
    }
    return "";
}


#endif
