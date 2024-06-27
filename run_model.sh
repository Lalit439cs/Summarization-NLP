#!/bin/bash

# bash run_model.sh train <path_to_data> <path_to_save>
# bash run_model.sh test <path_to_data> <path_to_model> <path_to_result>

train_model() {
    local data_path=$1
    local save_path=$2
    python flant5-finetune.py $data_path $save_path
}

test_model() {
    local data_path=$1
    local model_path=$2
    local result_path=$3
    python flant5-inference.py $data_path $model_path $result_path
}

if [ "$1" == "train" ]; then
    train_model "$2" "$3"
elif [ "$1" == "test" ]; then
    test_model "$2" "$3" "$4"
fi
