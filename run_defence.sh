export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_DIR=./dataset/beavertails-qwen
export ROLLOUT_TP_SIZE=2
export PROJECTNAME=tokenbuncher_defense
export EXPERIMENT_NAME=tokenbuncher-qwen-2.5-3b
export OUTPUT_DIR=./outputs/tokenbuncher-qwen-2.5-3b
unset VLLM_ATTENTION_BACKEND # unnecessary for high version vllm

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi
bash ./configs/defence_grpo.sh
