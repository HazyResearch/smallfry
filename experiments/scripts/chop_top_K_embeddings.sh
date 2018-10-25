FULL_VOCAB=$1
TOP_K=$2
head ${FULL_VOCAB} -n ${TOP_K} > ${FULL_VOCAB}.chopped_top_${TOP_K}
