WORK_DIR=../gpu
LOG_DIR=log
DATA_DIR=../../data #specify your own directory storing the data sets
WINDOW_RATIO=0.1
APP_TYPE=0 # algorithm type, 0 for reverse push
mkdir ${LOG_DIR}

DATA_SETS=("com-youtube.ungraph.bin" "soc-pokec-relationships.bin" "com-orkut.ungraph.bin" "twitter-2010_rev.bin") # a list of data sets for test
DIRECTIONS=(0 1 0 1) # 1 for directed and 0 for undirected graph, the order should be consistent with DATA_SETS
SOURCE_FEATURES=(top10) # top10 means the vertices with top 10 out-degree, other source features are top1000, top1000000 etc.
SOURCE_VERTEX_FILE_DIR=./exp_vids # exp_vids contains the pre-generated source vertices with specified source features
# for vary_batch_size: batch size is #edges for each batch, run_edge is the total number of incoming edges
BATCH_SIZES=(1 10 100 1000 10000 100000 1000000)
RUN_EDGES=(1000 100000 10000 100000 1000000 5000000 50000000)

#SOURCES_START and SOURCES_END indicate the range of source vertices to use, by default use 10 source vertices
SOURCES_START=3
SOURCES_END=4
#SOURCES_START=0
#SOURCES_END=10


vary_batch_size () {
for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    DIRECTION=${DIRECTIONS[$di]}

    for ((batchidx=0; batchidx<${#BATCH_SIZES[@]}; batchidx++)); do
        BATCH_SIZE=${BATCH_SIZES[$batchidx]}
        TOTAL_EDGE_NUM=${RUN_EDGES[$batchidx]}

        for SOURCE_FEATURE in ${SOURCE_FEATURES[@]}; do
            SOURCE_VERTEX_FILE=${SOURCE_VERTEX_FILE_DIR}/${DATA_SET}_${SOURCE_FEATURE}.txt 
            echo "source_vertex_file=${SOURCE_VERTEX_FILE}"
            SOURCES=$(cat ${SOURCE_VERTEX_FILE} | tr "\\n" " ")
            SOURCES=($SOURCES)
            for ((sidx=${SOURCES_START}; sidx<${SOURCES_END}; sidx++)); do
                sid=${SOURCES[$sidx]}
                OUTPUT_FILE="batch_size_${DATA_SET}_${BATCH_SIZE}_${sid}.txt"

                COMMAND="${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 1 -c ${BATCH_SIZE} -l ${TOTAL_EDGE_NUM} -s ${sid} > ${LOG_DIR}/${OUTPUT_FILE}"
                echo "command=${COMMAND}"
                
                ${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 1 -c ${BATCH_SIZE} -l ${TOTAL_EDGE_NUM} -s ${sid} > ${LOG_DIR}/${OUTPUT_FILE}                                                                                                                                                                  
            done
        done
    done
done
}

vary_variant (){
OPT_METHODS=(0 1 2 3)
OUTPUT_FILE_PREFIX=op_gpu
BATCH_RATIO=0.01
BATCH_COUNT=100
for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    DIRECTION=${DIRECTIONS[$di]}

    for variant in ${OPT_METHODS[@]};
    do
        for SOURCE_FEATURE in ${SOURCE_FEATURES[@]}; do
            SOURCE_VERTEX_FILE=${SOURCE_VERTEX_FILE_DIR}/${DATA_SET}_${SOURCE_FEATURE}.txt 
            echo "source_vertex_file=${SOURCE_VERTEX_FILE}"
            SOURCES=$(cat ${SOURCE_VERTEX_FILE} | tr "\\n" " ")
            SOURCES=($SOURCES)

            for ((sidx=${SOURCES_START}; sidx<${SOURCES_END}; sidx++)); do
                sid=${SOURCES[$sidx]}
                OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${variant}_${DATA_SET}_${sid}.txt"

                COMMAND="${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} -o ${variant} > ${LOG_DIR}/${OUTPUT_FILE}"
                echo "command=${COMMAND}"
                ${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} -o ${variant} > ${LOG_DIR}/${OUTPUT_FILE}
                
            done
        done
    done
done
}

vary_epsilon (){
ERRORS=(0.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001)
ERRORS_PREFIX=(1e5 1e6 1e7 1e8 1e9 1e10)
OUTPUT_FILE_PREFIX=error_gpu
BATCH_RATIO=0.01
BATCH_COUNT=100
for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    DIRECTION=${DIRECTIONS[$di]}

    for ((errorid=0; errorid<${#ERRORS[@]}; errorid++));
    do
        error=${ERRORS[$errorid]}
        error_prefix=${ERRORS_PREFIX[$errorid]}
        for SOURCE_FEATURE in ${SOURCE_FEATURES[@]}; do
            SOURCE_VERTEX_FILE=${SOURCE_VERTEX_FILE_DIR}/${DATA_SET}_${SOURCE_FEATURE}.txt 
            echo "source_vertex_file=${SOURCE_VERTEX_FILE}"
            SOURCES=$(cat ${SOURCE_VERTEX_FILE} | tr "\\n" " ")
            SOURCES=($SOURCES)

            for ((sidx=${SOURCES_START}; sidx<${SOURCES_END}; sidx++)); do
                sid=${SOURCES[$sidx]}
                OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${error_prefix}_${DATA_SET}_${sid}.txt"

                COMMAND="${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} -e ${error} > ${LOG_DIR}/${OUTPUT_FILE}"
                echo "command=${COMMAND}"
                ${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} -e ${error} > ${LOG_DIR}/${OUTPUT_FILE}
                
            done
        done
    done
done
}

vary_source_features (){
SOURCE_FEATURES=(top10 top1000 top1000000)
OUTPUT_FILE_PREFIX=source_feature
BATCH_RATIO=0.01
BATCH_COUNT=100
for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    DIRECTION=${DIRECTIONS[$di]}

    for SOURCE_FEATURE in ${SOURCE_FEATURES[@]}; do
        SOURCE_VERTEX_FILE=${SOURCE_VERTEX_FILE_DIR}/${DATA_SET}_${SOURCE_FEATURE}.txt 
        echo "source_vertex_file=${SOURCE_VERTEX_FILE}"
        SOURCES=$(cat ${SOURCE_VERTEX_FILE} | tr "\\n" " ")
        SOURCES=($SOURCES)

        for ((sidx=${SOURCES_START}; sidx<${SOURCES_END}; sidx++)); do
            sid=${SOURCES[$sidx]}
            OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${SOURCE_FEATURE}_${DATA_SET}_${sid}.txt"

            COMMAND="${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} > ${LOG_DIR}/${OUTPUT_FILE}"
            echo "command=${COMMAND}"
            ${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} > ${LOG_DIR}/${OUTPUT_FILE}
            
        done
    done
done
}

vary_batch_ratios (){
OUTPUT_FILE_PREFIX=batch_ratio
BATCH_RATIOS=(0.01 0.001 0.0001)
BATCH_COUNT=100
for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    DIRECTION=${DIRECTIONS[$di]}

    for BATCH_RATIO in ${BATCH_RATIOS[@]};
    do
        for SOURCE_FEATURE in ${SOURCE_FEATURES[@]}; do
            SOURCE_VERTEX_FILE=${SOURCE_VERTEX_FILE_DIR}/${DATA_SET}_${SOURCE_FEATURE}.txt 
            echo "source_vertex_file=${SOURCE_VERTEX_FILE}"
            SOURCES=$(cat ${SOURCE_VERTEX_FILE} | tr "\\n" " ")
            SOURCES=($SOURCES)

            for ((sidx=${SOURCES_START}; sidx<${SOURCES_END}; sidx++)); do
                sid=${SOURCES[$sidx]}
                OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${BATCH_RATIO}_${DATA_SET}_${sid}.txt"

                COMMAND="${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} > ${LOG_DIR}/${OUTPUT_FILE}"
                echo "command=${COMMAND}"
                ${WORK_DIR}/pagerank -d ${DATA_DIR}/${DATA_SET} -a ${APP_TYPE} -i ${DIRECTION} -y 1 -n 0 -r ${BATCH_RATIO} -b ${BATCH_COUNT} -s ${sid} > ${LOG_DIR}/${OUTPUT_FILE}
                
            done
        done
    done
done
}


vary_batch_size
vary_variant
vary_epsilon
vary_source_features
vary_batch_ratios

