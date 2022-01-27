#!/bin/bash

# Localtion of the data
DATA_PATH="../../../local_data/"
SAVEPOINT="./md5all.txt"

TESTS=("1st_test" "2nd_test" "3rd_test")

for test in ${TESTS[@]}; do
  DIR=${DATA_PATH}${test}/
  echo "${test} : [" >> $SAVEPOINT
  declare -i cont_line=0
  for file in ${DIR}/*; do
    cont_line+=1
    md5=($(md5sum ${file}))
    printf "\"${md5}\", " >> $SAVEPOINT
    if [[ $cont_line == 5 ]]; then
      printf "\n" >> $SAVEPOINT
      cont_line=0
    fi
  done
  echo "]," >> $SAVEPOINT
done