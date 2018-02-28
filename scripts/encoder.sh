# transform data graph file from SNAP form to our BIN form
# you need to specify DATA_DIR, DATA_SETS, IS_REVERSES 
DATA_DIR=/data1/wentian/data
DATA_SETS=(soc-pokec-relationships.txt com-youtube.ungraph.txt com-lj.ungraph.txt)
IS_REVERSES=(0 0 0)

WORK_DIR=../encoder
BIN_FILE=encoder

for ((di=0; di<${#DATA_SETS[@]}; di++)); do
    DATA_SET=${DATA_SETS[$di]}
    IS_REVERSE=${IS_REVERSES[$di]}
    
    COMMAND="${WORK_DIR}/${BIN_FILE} ${DATA_DIR}/${DATA_SET} ${IS_REVERSE}"    
    echo "command=${COMMAND}"
    ${WORK_DIR}/${BIN_FILE} ${DATA_DIR}/${DATA_SET} ${IS_REVERSE} 

done

# move all generated *.bin to the data folder
#mv *.bin ${DATA_DIR}
