# choose source vertex ids to run
# you need to specify DATA_DIR, DATA_SETS, DIRECTIONS, IS_OUT_DEGREE
DATA_DIR=/data1/wentian/data
DATA_SETS=(soc-pokec-relationships.bin com-youtube.ungraph.bin com-lj.ungraph.bin)
DIRECTIONS=(1 0 0)
IS_OUT_DEGREE=1 # set to 1 if choosing based on out-degree, set to 0 if based on in-degree

WORK_DIR=../workload
BIN_FILE=workload

for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    DIRECTION=${DIRECTIONS[$di]}
    
    COMMAND="${WORK_DIR}/${BIN_FILE} ${DATA_DIR}/${DATA_SET} ${DIRECTION} 0 ${IS_OUT_DEGREE}"    
    echo "command=${COMMAND}"
    ${WORK_DIR}/${BIN_FILE} ${DATA_DIR}/${DATA_SET} ${DIRECTION} 0 ${IS_OUT_DEGREE} 

done

# move all source vertex ids files to 'vids' for later usage
SOURCE_VERTEX_FILE_DIR=vids
mkdir ${SOURCE_VERTEX_FILE_DIR}
mv *.txt ${SOURCE_VERTEX_FILE_DIR}
