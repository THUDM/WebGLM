#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_webglm.sh"

DATA_PATH="data/web_questions.jsonl"

run_cmd="python ${main_dir}/evaluate.py \
       --webglm_ckpt_path $GENERATOR_CHECKPOINT_PATH \
       --task web_questions \
       --evaluate_task_data_path $DATA_PATH"

eval ${run_cmd}